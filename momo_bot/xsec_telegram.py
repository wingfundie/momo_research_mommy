from __future__ import annotations

import asyncio
import html
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from telegram import InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

import trading_bot as positions_api
from momo_bot.chart_theme import (
    changes_figure,
    export_figures,
    performance_figures,
    portfolio_figure,
    risk_figure,
    signal_distribution_figure,
    ticker_figures,
)
from momo_bot.costs import BacktestCostConfig
from momo_bot.exchanges.binance_futures import BinanceFuturesExchange
from momo_bot.xsec20 import (
    CONFIG_ID,
    GROSS_CAP,
    MODEL,
    PORTFOLIO_VALUE,
    SLIPPAGE_BPS,
    TARGET_VOL,
    TICKER_RISK_CAP,
    VOLATILITY_WINDOW,
    XSecSnapshot,
    annual_metrics,
    compare_returns,
    get_xsec20_service,
    normalize_symbol,
    round_quantity,
)


PERIODS = {
    "1d": 1,
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "180d": 180,
    "1y": 365,
    "all": None,
}
_LOOKBACK_PATTERN = re.compile(r"^(\d{1,4})([dwmy])$")
_LOOKBACK_MULTIPLIERS = {"d": 1, "w": 7, "m": 30, "y": 365}


def lookback_token(value: str) -> tuple[str, int | None] | None:
    token = str(value).strip().lower()
    if token == "all":
        return "all", None
    match = _LOOKBACK_PATTERN.fullmatch(token)
    if match is None:
        return None
    amount = int(match.group(1))
    days = amount * _LOOKBACK_MULTIPLIERS[match.group(2)]
    if amount < 1 or days > 3650:
        return None
    return token, days


def parse_lookback(
    args: Iterable[str],
    *,
    default: str | None,
) -> tuple[str | None, int | None]:
    for value in args:
        parsed = lookback_token(str(value))
        if parsed is not None:
            return parsed
    if default is None:
        return None, None
    parsed_default = lookback_token(default)
    if parsed_default is None:
        raise ValueError(f"Invalid default lookback: {default}")
    return parsed_default


def _window(frame, days: int | None):
    if days is None or frame.empty:
        return frame
    start = frame.index.max() - pd.Timedelta(days=days - 1)
    return frame.loc[frame.index >= start]


def _standalone_sharpe(snapshot: XSecSnapshot, days: int | None) -> pd.Series:
    if days is None:
        return snapshot.standalone_sr
    returns = snapshot.ticker_history["standalone_return"].unstack("symbol")
    returns = _window(returns, days)
    minimum = min(20, max(3, int(days * 0.5)))

    def score(values: pd.Series) -> float:
        clean = values.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < minimum or clean.std(ddof=1) <= 0:
            return np.nan
        return float(clean.mean() / clean.std(ddof=1) * np.sqrt(365))

    return returns.apply(score, axis=0).rename("standalone_sr")


def wants_legacy(args: Iterable[str]) -> bool:
    return any(str(arg).lower() == "legacy" for arg in args)


def result_count(args: Iterable[str], default: int = 10) -> int:
    for arg in args:
        try:
            return min(max(int(arg), 1), 25)
        except (TypeError, ValueError):
            continue
    return default


def _fmt(value, kind: str = "number") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if kind == "percent":
        return f"{value:+.1%}"
    if kind == "rank":
        return f"{value:.0%}"
    if kind == "money":
        return f"${value:,.0f}"
    return f"{value:+.2f}"


def _table(frame: pd.DataFrame, title: str | None = None) -> str:
    if frame.empty:
        body = "No rows available."
    else:
        body = frame.to_string(index=False)
    prefix = f"<b>{html.escape(title)}</b>\n" if title else ""
    return prefix + f"<pre>{html.escape(body)}</pre>"


async def _reply_frame(update: Update, frame: pd.DataFrame, title: str | None = None) -> None:
    message = _table(frame, title)
    if len(message) <= 3900:
        await update.message.reply_text(message, parse_mode=ParseMode.HTML)
        return
    plain = frame.to_string(index=False)
    for start in range(0, len(plain), 3600):
        await update.message.reply_text(
            f"<pre>{html.escape(plain[start:start + 3600])}</pre>",
            parse_mode=ParseMode.HTML,
        )


async def _snapshot(
    update: Update,
    *,
    announce: bool = True,
    warn_stale: bool = True,
) -> XSecSnapshot:
    if announce:
        await update.message.reply_text("Loading corrected XSec20 snapshot…")
    service = get_xsec20_service()
    snapshot = await asyncio.to_thread(service.get_snapshot)
    if warn_stale and service.is_stale(snapshot):
        await update.message.reply_text(
            f"⚠️ STALE SNAPSHOT — data cutoff {snapshot.data_cutoff:%Y-%m-%d}. "
            "Values below are the latest stored model state."
        )
    return snapshot


async def _send_figures(update: Update, figures, stem: str) -> None:
    with tempfile.TemporaryDirectory(prefix="momo_xsec_chart_") as tmp:
        paths = await asyncio.to_thread(export_figures, figures, Path(tmp), stem)
        with ExitStack() as stack:
            files = [stack.enter_context(path.open("rb")) for path in paths]
            if len(files) == 1:
                await update.message.reply_photo(photo=files[0])
            else:
                await update.message.reply_media_group(
                    media=[InputMediaPhoto(media=file) for file in files]
                )


def _signal_rows(latest: pd.DataFrame) -> pd.DataFrame:
    frame = latest.reset_index().rename(columns={"index": "symbol"})
    return pd.DataFrame(
        {
            "Coin": frame["symbol"].str.removesuffix("USDT"),
            "Trend": frame["trend_side"],
            "XSec": frame["xsec_side"],
            "Fcast": frame["absolute_forecast"].map(lambda x: _fmt(x)),
            "Rank": frame["rank"].map(lambda x: _fmt(x, "rank")),
            "Target": frame["target_weight"].map(lambda x: _fmt(x, "percent")),
            "SR": frame["standalone_sr"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}"),
            "_symbol": frame["symbol"],
            "_trend": frame["trend_side"],
            "_xsec": frame["xsec_signal"],
            "_rank": frame["rank"],
            "_sr": frame["standalone_sr"],
        }
    )


async def signals_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    sort_by: str,
    sector: str | None = None,
) -> None:
    snapshot = await _snapshot(update)
    period, days = parse_lookback(context.args, default="all")
    latest = snapshot.latest.copy()
    latest["standalone_sr"] = _standalone_sharpe(snapshot, days).reindex(latest.index)
    if sector:
        allowed = get_xsec20_service().sector_symbols(sector)
        latest = latest.loc[latest.index.intersection(sorted(allowed))]
    rows = _signal_rows(latest)
    count = result_count(context.args)
    if sort_by == "sr":
        longs = rows[rows["_trend"].eq("LONG")].sort_values("_sr", ascending=False).head(count)
        shorts = rows[rows["_trend"].eq("SHORT")].sort_values("_sr", ascending=False).head(count)
        title = f"Standalone trend Sharpe · {period}"
        long_label = "POSITIVE TREND"
        short_label = "NEGATIVE TREND"
    else:
        longs = rows[rows["_xsec"].gt(0)].sort_values("_rank", ascending=False).head(count)
        shorts = rows[rows["_xsec"].lt(0)].sort_values("_rank", ascending=True).head(count)
        title = f"XSec20 relative strength · SR {period}"
        long_label = "PORTFOLIO LONG"
        short_label = "PORTFOLIO SHORT"
    visible = ["Coin", "Trend", "XSec", "Fcast", "Rank", "Target", "SR"]
    await _reply_frame(update, longs[visible], f"{title} — {long_label}")
    await _reply_frame(update, shorts[visible], f"{title} — {short_label}")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update)
    m = snapshot.manifest
    text = (
        f"<b>Corrected XSec20 momentum</b>\n"
        f"Model: <code>{MODEL}</code>\nConfig: <code>{CONFIG_ID}</code>\n"
        f"Cutoff: {html.escape(str(m['data_cutoff']))}\n"
        f"Risk: {TARGET_VOL:.0%} vol · {GROSS_CAP:.1f}× gross · {TICKER_RISK_CAP:.0%} ticker cap\n"
        f"Execution: daily · next open · 100% taker · {SLIPPAGE_BPS:.0f} bps slippage\n"
        f"Validation Sharpe: {m.get('validation_net_sharpe', float('nan')):.3f}\n"
        f"2026 holdout Sharpe: {m.get('holdout_net_sharpe', float('nan')):.3f}\n"
        "Construction: raw-rank neutral; inverse-vol sizing makes final holdings non-neutral.\n"
        f"2026 held |net|: mean {m.get('holdout_mean_absolute_net_exposure', float('nan')):.1%} · "
        f"max {m.get('holdout_max_absolute_net_exposure', float('nan')):.1%}"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def _rounded_portfolio(snapshot: XSecSnapshot) -> pd.DataFrame:
    latest = snapshot.latest.reset_index()
    exchange = BinanceFuturesExchange(
        cost_config=BacktestCostConfig(
            enabled=True,
            source="default",
            taker_share=1.0,
            taker_fee=0.0004,
            maker_fee=0.0002,
            slippage_bps=SLIPPAGE_BPS,
            include_funding=True,
            funding_mode="historical",
        )
    )

    def apply_rules(row):
        rules = exchange.get_symbol_rules(row.symbol)
        target_quantity = round_quantity(
            row.estimated_quantity if abs(row.target_notional) >= 1e-8 else 0.0,
            step_size=rules.step_size,
            min_qty=rules.min_qty,
        )
        held_quantity = round_quantity(
            row.held_notional / row.price if row.price and np.isfinite(row.price) else 0.0,
            step_size=rules.step_size,
            min_qty=rules.min_qty,
        )
        if rules.min_notional and abs(target_quantity * row.price) < rules.min_notional:
            target_quantity = 0.0
        return pd.Series(
            {
                "quantity": target_quantity,
                "held_quantity": held_quantity,
                "step_size": rules.step_size,
                "min_notional": rules.min_notional,
            }
        )

    rounded = await asyncio.to_thread(lambda: latest.apply(apply_rules, axis=1))
    latest = pd.concat([latest, rounded], axis=1)
    return latest


def _portfolio_side_lines(active: pd.DataFrame, side: str) -> list[str]:
    """Build compact rows for one side of the one-message portfolio view."""
    side_frame = active.loc[active["xsec_side"].eq(side)].copy()
    side_frame = side_frame.reindex(
        side_frame["target_weight"].abs().sort_values(ascending=False).index
    )
    notional = float(side_frame["target_notional"].sum())
    lines = [
        f"{side}S {len(side_frame)} | ${abs(notional) / 1_000:.2f}k",
        "Coin      $k    Qty |   Str  Rk   SR",
    ]

    def compact_quantity(value: float) -> str:
        value = abs(value)
        if value >= 1_000:
            return f"{value:.0f}"
        if value >= 100:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        if value >= 1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")

    for row in side_frame.itertuples(index=False):
        coin = str(row.symbol).removesuffix("USDT")
        strength = "n/a" if not np.isfinite(row.absolute_forecast) else f"{row.absolute_forecast:+.1f}"
        rank = "n/a" if not np.isfinite(row.rank) else f"{row.rank * 100:.0f}"
        sharpe = "n/a" if not np.isfinite(row.standalone_sr) else f"{row.standalone_sr:+.1f}"
        allocation = f"{abs(row.target_notional) / 1_000:.2f}"
        quantity = "n/a" if not np.isfinite(row.quantity) else compact_quantity(row.quantity)
        lines.append(f"{coin:<6} {allocation:>5} {quantity:>6} | {strength:>5} {rank:>3} {sharpe:>4}")
    return lines


def _portfolio_message(
    active: pd.DataFrame,
    state: pd.Series,
    *,
    stale: bool = False,
) -> str:
    body = "\n".join(
        _portfolio_side_lines(active, "LONG")
        + [""]
        + _portfolio_side_lines(active, "SHORT")
    )
    stale_line = "⚠️ Latest stored snapshot · " if stale else ""
    message = (
        "<b>XSec20 · $100k target</b>\n"
        f"{stale_line}Held L {state.long_exposure:.1%} · S {abs(state.short_exposure):.1%} · "
        f"G {state.gross_exposure:.1%} · N {state.net_exposure:+.1%}\n"
        "$k and Qty are absolute sizes within each side\n"
        "Str −20…20 · Rk percentile · SR standalone Sharpe\n"
        f"<pre>{html.escape(body)}</pre>"
    )
    if len(message) > 4096:
        raise ValueError(f"Portfolio message is too long for Telegram: {len(message)} characters")
    return message


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update, announce=False, warn_stale=False)
    unit = "usd" if any(str(x).lower() == "usd" for x in context.args) else "share"
    latest = await _rounded_portfolio(snapshot)
    active = latest[latest["target_weight"].abs().gt(1e-8)].copy()
    active = active.reindex(active["target_weight"].abs().sort_values(ascending=False).index)
    state = snapshot.portfolio.iloc[-1]
    await update.message.reply_text(
        _portfolio_message(
            active,
            state,
            stale=get_xsec20_service().is_stale(snapshot),
        ),
        parse_mode=ParseMode.HTML,
    )
    figures = [portfolio_figure(latest, unit=unit)]
    period, days = parse_lookback(context.args, default=None)
    if period is not None:
        figures.append(risk_figure(_window(snapshot.portfolio, days), unit=unit))
    await _send_figures(update, figures, "portfolio")


async def rebalance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update)
    latest = await _rounded_portfolio(snapshot)
    latest["target_quantity"] = latest["quantity"].fillna(0.0)
    latest["trade_quantity"] = latest["target_quantity"] - latest["held_quantity"]
    latest["trade_notional"] = latest["trade_quantity"] * latest["price"]
    executable = latest["trade_notional"].abs().gt(1.0)
    executable &= latest["min_notional"].isna() | latest["trade_notional"].abs().ge(latest["min_notional"])
    latest = latest[executable]
    latest = latest.reindex(latest["trade_notional"].abs().sort_values(ascending=False).index)
    action = np.select(
        [
            latest["held_quantity"].eq(0),
            latest["target_quantity"].eq(0),
            latest["held_quantity"].mul(latest["target_quantity"]).lt(0),
            latest["target_quantity"].abs().lt(latest["held_quantity"].abs()),
        ],
        ["OPEN", "CLOSE", "FLIP", "REDUCE"],
        default="INCREASE",
    )
    display = pd.DataFrame(
        {
            "Coin": latest["symbol"].str.removesuffix("USDT"),
            "Action": action,
            "Trade $": latest["trade_notional"].map(lambda x: f"{x:+,.0f}"),
            "Trade Qty": latest["trade_quantity"].map(lambda x: f"{x:+g}"),
            "Target %": latest["target_weight"].map(lambda x: f"{x:+.2%}"),
        }
    )
    await _reply_frame(update, display, "Read-only $100k model rebalance")


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update)
    unit = "usd" if any(str(x).lower() == "usd" for x in context.args) else "share"
    period, days = parse_lookback(context.args, default="1y")
    state = snapshot.portfolio.iloc[-1]
    await update.message.reply_text(
        f"Gross {state.gross_exposure:.1%} · Net {state.net_exposure:+.1%} · "
        f"Long {state.long_exposure:.1%} · Short {state.short_exposure:.1%} · "
        f"60d realized vol {state.realized_volatility_60d:.1%}"
    )
    await _send_figures(
        update,
        [risk_figure(_window(snapshot.portfolio, days), unit=unit)],
        f"risk_{period}",
    )


def _performance_args(args: Iterable[str]) -> tuple[str, int | None, str | None]:
    period, days = parse_lookback(args, default="1y")
    asset = None
    for raw in args:
        value = str(raw).lower()
        if lookback_token(value) is None and value not in {"legacy"}:
            asset = normalize_symbol(value)
    return str(period), days, asset


async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update)
    period, days, asset = _performance_args(context.args)
    returns = _window(snapshot.portfolio["net_return"].dropna(), days)
    benchmark = None
    comparison = None
    if asset:
        benchmark = await asyncio.to_thread(
            get_xsec20_service().benchmark_returns,
            asset,
            start=returns.index.min() - pd.Timedelta(days=1),
            end=returns.index.max(),
        )
        aligned, comparison = compare_returns(returns, benchmark)
        returns = aligned["portfolio"]
        benchmark = aligned["benchmark"]
    metrics = annual_metrics(returns)
    rows = [
        {"Series": "Model", "Return": f"{metrics['cumulative_return']:.1%}", "Vol": f"{metrics['annual_volatility']:.1%}", "Sharpe": f"{metrics['sharpe']:.2f}", "Max DD": f"{metrics['max_drawdown']:.1%}"}
    ]
    if benchmark is not None and comparison is not None:
        bm = comparison["benchmark"]
        rows.append({"Series": asset.removesuffix("USDT"), "Return": f"{bm['cumulative_return']:.1%}", "Vol": f"{bm['annual_volatility']:.1%}", "Sharpe": f"{bm['sharpe']:.2f}", "Max DD": f"{bm['max_drawdown']:.1%}"})
    await _reply_frame(update, pd.DataFrame(rows), f"Performance · {period}")
    if comparison is not None:
        await update.message.reply_text(
            f"Excess return {comparison['excess_cumulative_return']:+.1%} · "
            f"Correlation {comparison['correlation']:.2f} · Beta {comparison['beta']:.2f} · "
            f"{comparison['start']:%Y-%m-%d} to {comparison['end']:%Y-%m-%d}"
        )
    figures = performance_figures(
        returns,
        benchmark_returns=benchmark,
        benchmark_name=asset.removesuffix("USDT") if asset else None,
    )
    await _send_figures(update, figures, "performance")


async def changes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update)
    timestamps = snapshot.ticker_history.index.get_level_values("timestamp").unique().sort_values()
    if len(timestamps) < 2:
        await update.message.reply_text("No earlier XSec20 snapshot is available for comparison.")
        return
    period, days = parse_lookback(context.args, default="1d")
    current_stamp = timestamps[-1]
    if days is None:
        previous_stamp = timestamps[0]
    else:
        target = current_stamp - pd.Timedelta(days=days)
        candidates = timestamps[timestamps <= target]
        previous_stamp = candidates[-1] if len(candidates) else timestamps[0]
    current = snapshot.ticker_history.xs(current_stamp, level="timestamp")
    previous = snapshot.ticker_history.xs(previous_stamp, level="timestamp").reindex(current.index)
    changes = pd.DataFrame(
        {
            "symbol": current.index,
            "forecast_change": current["absolute_forecast"] - previous["absolute_forecast"],
            "rank_change": current["rank"] - previous["rank"],
            "target_change": current["target_weight"] - previous["target_weight"],
        }
    )
    count = result_count(context.args)
    changes = changes.reindex(changes["target_change"].abs().sort_values(ascending=False).index).head(count)
    display = pd.DataFrame(
        {
            "Coin": changes["symbol"].str.removesuffix("USDT"),
            "Forecast Δ": changes["forecast_change"].map(lambda x: f"{x:+.2f}"),
            "Rank Δ": changes["rank_change"].map(lambda x: f"{x:+.1%}"),
            "Target Δ": changes["target_change"].map(lambda x: f"{x:+.2%}"),
        }
    )
    await _reply_frame(
        update,
        display,
        f"Largest changes · {previous_stamp:%Y-%m-%d} to {current_stamp:%Y-%m-%d} ({period})",
    )
    await _send_figures(update, [changes_figure(changes)], "changes")


async def distribution_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update)
    period, days = parse_lookback(context.args, default="180d")
    forecasts = snapshot.ticker_history["absolute_forecast"].unstack("symbol")
    forecasts = _window(forecasts, days)
    current = forecasts.iloc[-1].dropna()
    if current.empty:
        await update.message.reply_text("No current XSec20 forecasts are available.")
        return
    positive_breadth = float(current.gt(0).mean())
    await update.message.reply_text(
        f"Universe {len(current)} · Average {current.mean():+.2f} · "
        f"P25 {current.quantile(.25):+.2f} · P75 {current.quantile(.75):+.2f} · "
        f"Positive breadth {positive_breadth:.0%}"
    )
    await _send_figures(
        update,
        [signal_distribution_figure(forecasts, period_label=str(period))],
        "signal_distribution",
    )


MANUAL_SECTIONS = (
    (
        "<b>XSec20 Telegram bot manual</b>\n\n"
        "Commands work with or without a slash. Example: <code>mc SOL 6m</code> and "
        "<code>/mc SOL 6m</code> are equivalent.\n\n"
        "<b>Lookbacks</b>\n"
        "Use any value from 1 to 3650 days with these suffixes:\n"
        "• <code>d</code> days: <code>45d</code>\n"
        "• <code>w</code> weeks: <code>12w</code>\n"
        "• <code>m</code> 30-day months: <code>6m</code>\n"
        "• <code>y</code> 365-day years: <code>2y</code>\n"
        "• <code>all</code> full available history\n\n"
        "Lookbacks change the reporting window. They do not alter the pinned EWMAC, IC20, "
        "vol60, target-vol or portfolio-construction parameters.\n\n"
        "<b>Signals</b>\n"
        "<code>mom_sig_str [N] [lookback]</code> — current XSec strength; selected-window SR is shown.\n"
        "<code>mom_sig_sr [N] [lookback]</code> — rank by standalone annualized Sharpe over the window.\n"
        "<code>mom_l1 [N] [lookback]</code> — Layer-1 subset.\n"
        "<code>mom_eth_beta [N] [lookback]</code> — Ethereum ecosystem subset.\n"
        "<code>mom_sol_beta [N] [lookback]</code> — Solana ecosystem subset.\n"
        "<code>mom_meme [N] [lookback]</code> — meme subset.\n"
        "Append <code>legacy</code> to a momentum command to use the old per-ticker engine."
    ),
    (
        "<b>Charts and analytics</b>\n"
        "<code>mc TICKER [lookback]</code> — ticker returns, price/forecast and rank/weight charts.\n"
        "<code>distribution [lookback]</code> — all forecast paths, P25/P75, mean and breadth.\n"
        "<code>performance [lookback] [ASSET]</code> — portfolio metrics and optional benchmark.\n"
        "<code>risk [share|usd] [lookback]</code> — long, short, gross and net exposure history.\n"
        "<code>changes [N] [lookback]</code> — changes from the selected historical date to now.\n\n"
        "<b>Portfolio</b>\n"
        "<code>portfolio [share|usd] [lookback]</code> — current sizing table and composition; adding a "
        "lookback also includes exposure history.\n"
        "<code>rebalance</code> — exchange-rounded model target changes on the fixed $100k base.\n"
        "<code>portfolio_momo</code> — compare actual Binance positions with model signals.\n\n"
        "<b>Model and operations</b>\n"
        "<code>model</code> — pinned parameters, validation and holdout diagnostics.\n"
        "<code>health</code> — data cutoff, universe and refresh status.\n"
        "<code>manual</code> or <code>help</code> — show this guide.\n\n"
        "<b>Breakout</b>\n"
        "<code>breakout_sig_sr</code>, <code>breakout_sig_str</code>, <code>breakout_l1</code>, "
        "<code>breakout_eth_beta</code>, <code>breakout_sol_beta</code>, <code>breakout_meme</code>, "
        "<code>breakout_sr_100</code>, <code>breakout_str_100</code>, <code>breakout_pairs</code>, "
        "<code>portfolio_breakout</code>, and <code>bc TICKER [lookback]</code>.\n\n"
        "All portfolio and rebalance outputs are analytical and read-only; they do not place orders."
    ),
)


async def manual_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    for section in MANUAL_SECTIONS:
        await update.message.reply_text(section, parse_mode=ParseMode.HTML)


async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update, announce=False)
    service = get_xsec20_service()
    m = snapshot.manifest
    text = (
        f"<b>XSec20 health</b>\n"
        f"Status: {'⚠️ STALE' if service.is_stale(snapshot) else '✅ FRESH'}\n"
        f"Data cutoff: {snapshot.data_cutoff:%Y-%m-%d}\n"
        f"Generated: {html.escape(str(m.get('generated_at')))}\n"
        f"Universe: {m.get('universe_assets')} assets · as of {html.escape(str(m.get('universe_as_of')))}\n"
        f"Config: <code>{CONFIG_ID}</code>\n"
        f"Refresh fallback used: {'yes' if m.get('refresh_error') else 'no'}"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def ticker_chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    period, days = parse_lookback(context.args, default="all")
    args = [
        arg
        for arg in context.args
        if str(arg).lower() != "legacy" and lookback_token(str(arg)) is None
    ]
    if not args:
        await update.message.reply_text("Usage: /mc SOL [lookback] [legacy]")
        return
    symbol = normalize_symbol(args[0])
    snapshot = await _snapshot(update)
    try:
        history = _window(snapshot.ticker(symbol), days)
    except KeyError as exc:
        await update.message.reply_text(str(exc))
        return
    await _send_figures(
        update,
        ticker_figures(history, symbol),
        f"ticker_{symbol.lower()}_{period}",
    )


async def account_momentum_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    snapshot = await _snapshot(update)
    latest = snapshot.latest.reset_index().set_index("symbol")
    longs, shorts = positions_api.gen_ls_frame(positions_api.get_futures_positions())
    longs = longs.iloc[:-1].copy()
    shorts = shorts.iloc[:-1].copy()
    longs["Account"] = "LONG"
    shorts["Account"] = "SHORT"
    positions = pd.concat([longs, shorts], ignore_index=True)
    if positions.empty:
        await update.message.reply_text("No open Binance futures positions were found.")
        return
    positions = positions.set_index("symbol")
    supported = positions.join(latest, how="inner")
    if not supported.empty:
        display = pd.DataFrame(
            {
                "Coin": supported.index.str.removesuffix("USDT"),
                "Account": supported["Account"],
                "Notional": supported["notional"].astype(float).map(lambda x: f"${x:,.0f}"),
                "Trend": supported["trend_side"],
                "XSec": supported["xsec_side"],
                "Rank": supported["rank"].map(lambda x: f"{x:.0%}"),
                "SR": supported["standalone_sr"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}"),
            }
        )
        display["Aligned"] = np.where(display["Account"].eq(display["XSec"]), "YES", "NO")
        await _reply_frame(update, display, "Binance account momentum exposure")
    unsupported = positions.loc[~positions.index.isin(latest.index)]
    if not unsupported.empty:
        display = pd.DataFrame(
            {
                "Coin": unsupported.index.str.removesuffix("USDT"),
                "Account": unsupported["Account"],
                "Notional": unsupported["notional"].astype(float).map(lambda x: f"${x:,.0f}"),
                "Status": "NO MODEL DATA",
            }
        )
        await _reply_frame(update, display, "Unsupported account positions")
