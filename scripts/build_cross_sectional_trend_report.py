from __future__ import annotations

import hashlib
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_theme.report_theme import figure_html, finding, hero, metric, render_page, style_plotly
from scripts.execute_complete_crypto_study import load_full_price_panel
from scripts.execute_full_crypto_study import funding_coefficients
from scripts.execute_production_like_walkforward import (
    INPUTS,
    delayed_positions,
    funding_tradability_mask,
    rich_metrics,
    simulation_from_held,
)


RESULTS = ROOT / "data_store/crypto_momentum_research/cross_sectional_trend_v1"
OUTPUT = ROOT / "reports/crypto_cross_sectional_trend_study_20260921.html"
COMPOSITION_OUTPUT = ROOT / "reports/crypto_cross_sectional_trend_composition_20260921.html"

FAMILY_LABELS = {
    "xs_momentum": "Cross-sectional EWMAC momentum",
    "xs_breakout": "Cross-sectional Carver5 breakout",
}
STRATEGY_LABELS = {
    "xs_momentum_ensemble": "Momentum ensemble",
    "xs_breakout_ensemble": "Breakout ensemble",
    "xs_combined_50_50": "50/50 diagnostic combination",
}


def pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def num(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2f}"


def chart_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def table_html(frame: pd.DataFrame) -> str:
    output = frame.copy()
    for column in output:
        key = str(column).lower()
        if any(token in key for token in (
            "return", "volatility", "drawdown", "funding", "fees", "slippage",
            "cvar", "concentration",
        )) or key.endswith("_rate"):
            output[column] = output[column].map(pct)
        elif any(token in key for token in ("sharpe", "sortino", "calmar", "beta", "correlation")):
            output[column] = output[column].map(num)
        elif "turnover" in key:
            output[column] = output[column].map(lambda value: "—" if pd.isna(value) else f"{value:,.1f}×")
        elif key in {"gross_cap", "gross_exposure", "net_exposure"}:
            output[column] = output[column].map(lambda value: "—" if pd.isna(value) else f"{value:.2f}×")
        elif pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda value: "—" if pd.isna(value) else f"{value:,.2f}")
    return '<div class="table-wrap" tabindex="0">' + output.to_html(index=False, border=0, escape=True) + "</div>"


def standard_equity_figure(state: pd.DataFrame) -> go.Figure:
    subset = state[state.timestamp.ge("2024-01-01")].copy()
    subset["family_label"] = subset.family.map(FAMILY_LABELS)
    subset["growth"] = subset.groupby("family").net_return.transform(lambda values: (1 + values.fillna(0)).cumprod())
    fig = px.line(subset, x="timestamp", y="growth", color="family_label")
    fig.add_vline(x=pd.Timestamp("2026-01-01").timestamp() * 1000, line_dash="dash", line_color="#bd4e4e")
    fig.add_annotation(x="2026-01-01", y=1.02, yref="paper", text="Untouched holdout begins", showarrow=False)
    style_plotly(fig, "Standard validation and untouched historical holdout", 650)
    fig.update_yaxes(title="Growth of $1")
    return fig


def standard_scatter(selection: pd.DataFrame) -> go.Figure:
    frame = selection.copy()
    frame["family_label"] = frame.family.map(FAMILY_LABELS)
    fig = px.scatter(
        frame, x="validation_net_sharpe", y="holdout_net_sharpe", color="family_label",
        hover_data=["model", "target_vol", "gross_cap", "ticker_risk_cap", "rebalance"],
        opacity=.65,
    )
    lo = float(np.nanmin([frame.validation_net_sharpe.min(), frame.holdout_net_sharpe.min()]))
    hi = float(np.nanmax([frame.validation_net_sharpe.max(), frame.holdout_net_sharpe.max()]))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#999", dash="dot"))
    fig.add_hline(y=0, line_color="#777", line_width=1)
    style_plotly(fig, "Validation performance versus later holdout", 650)
    fig.update_xaxes(title="2024–25 validation net Sharpe")
    fig.update_yaxes(title="2026 YTD holdout net Sharpe")
    return fig


def rolling_equity_figure(state: pd.DataFrame) -> go.Figure:
    names = list(STRATEGY_LABELS)
    subset = state[state.strategy.isin(names) & state.timestamp.ge("2022-01-01")].copy()
    subset["strategy_label"] = subset.strategy.map(STRATEGY_LABELS)
    subset["growth"] = subset.groupby("strategy").net_return.transform(lambda values: (1 + values.fillna(0)).cumprod())
    fig = px.line(subset, x="timestamp", y="growth", color="strategy_label")
    style_plotly(fig, "Stitched production-like outer-fold growth", 680)
    fig.update_yaxes(title="Growth of $1")
    return fig


def fold_figure(folds: pd.DataFrame) -> go.Figure:
    frame = folds[folds.strategy.isin(STRATEGY_LABELS)].copy()
    frame["strategy_label"] = frame.strategy.map(STRATEGY_LABELS)
    fig = px.bar(frame, x="fold", y="net_sharpe", color="strategy_label", barmode="group")
    fig.add_hline(y=0, line_color="#777", line_width=1)
    style_plotly(fig, "Net Sharpe in each untouched outer fold", 620)
    fig.update_yaxes(title="Net Sharpe")
    return fig


def rolling_sharpe_figure(rolling: pd.DataFrame) -> go.Figure:
    frame = rolling[
        rolling.strategy.isin(STRATEGY_LABELS) & rolling.window_days.eq(365)
    ].dropna(subset=["sharpe"]).copy()
    frame["strategy_label"] = frame.strategy.map(STRATEGY_LABELS)
    fig = px.line(frame, x="timestamp", y="sharpe", color="strategy_label")
    fig.add_hline(y=0, line_color="#777", line_width=1)
    style_plotly(fig, "Trailing 365-day net Sharpe", 620)
    fig.update_yaxes(title="Net Sharpe")
    return fig


def annual_figure(state: pd.DataFrame) -> go.Figure:
    frame = state[state.strategy.isin(STRATEGY_LABELS) & state.timestamp.ge("2022-01-01")].copy()
    frame["year"] = frame.timestamp.dt.year
    annual = frame.groupby(["strategy", "year"], as_index=False).agg(
        net_return=("net_return", lambda x: (1 + x).prod() - 1),
        funding=("funding", "sum"), fees=("fee", "sum"), slippage=("slippage", "sum"),
    )
    annual["strategy_label"] = annual.strategy.map(STRATEGY_LABELS)
    fig = px.bar(annual, x="year", y="net_return", color="strategy_label", barmode="group")
    fig.add_hline(y=0, line_color="#777", line_width=1)
    style_plotly(fig, "Calendar-year net return", 620)
    fig.update_yaxes(title="Net return", tickformat=".0%")
    return fig


def baseline_figure(state: pd.DataFrame) -> go.Figure:
    frame = state[state.timestamp.ge("2022-01-01")].copy()
    frame["growth"] = frame.groupby("baseline").net_return.transform(lambda values: (1 + values.fillna(0)).cumprod())
    fig = px.line(frame, x="timestamp", y="growth", color="baseline")
    style_plotly(fig, "Reference portfolios under the same execution accounting", 620)
    fig.update_yaxes(title="Growth of $1")
    return fig


def build_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_prices = load_full_price_panel()
    universe = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    members = universe.loc[universe.member.astype(str).str.lower().isin(["true", "1"]), "symbol"]
    symbols = [symbol for symbol in members if symbol in all_prices]
    prices = all_prices[symbols]
    index = prices.index
    open_prices = pd.read_parquet(INPUTS / "portfolio_open_prices.parquet").reindex(index=index, columns=symbols)
    open_returns = open_prices.shift(-1).div(open_prices).sub(1).fillna(0.0)
    events = pd.read_parquet(INPUTS / "funding_events.parquet")
    events["funding_time"] = pd.to_datetime(events.funding_time, utc=True)
    same, midnight = funding_coefficients(prices, events)
    tradable = funding_tradability_mask(index, symbols, events)
    commissions = pd.read_csv(INPUTS / "commission_rates.csv").set_index("symbol")
    maker = commissions.maker.reindex(symbols).fillna(0.0002)
    taker = commissions.taker.reindex(symbols).fillna(0.0004)
    available = prices.notna() & open_prices.notna() & tradable
    equal = available.astype(float).div(available.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fdv = pd.to_numeric(
        universe.set_index("symbol").fully_diluted_valuation.reindex(symbols), errors="coerce"
    ).fillna(0.0)
    fdv_weight = available.astype(float).mul(fdv, axis=1)
    fdv_weight = fdv_weight.div(fdv_weight.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    btc = pd.DataFrame(0.0, index=index, columns=symbols)
    if "BTCUSDT" in btc:
        btc["BTCUSDT"] = available["BTCUSDT"].astype(float)
    rng = np.random.default_rng(20260921)
    random_sign = pd.DataFrame(rng.choice([-1.0, 1.0], size=prices.shape), index=index, columns=symbols)
    random_sign = random_sign.where(available, 0.0)
    random_sign = random_sign.div(random_sign.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    definitions = {
        "Equal-weight eligible basket": equal,
        "Current-FDV-weighted basket": fdv_weight,
        "BTC perpetual buy/hold": btc,
        "Deterministic no-skill signs": random_sign,
    }
    masks = {
        "validation": pd.Series((index >= "2024-01-01") & (index < "2026-01-01"), index=index),
        "holdout_2026": pd.Series(index >= "2026-01-01", index=index),
        "rolling_2022_2026": pd.Series(index >= "2022-01-01", index=index),
    }
    btc_returns = open_returns.get("BTCUSDT", pd.Series(0.0, index=index))
    metric_rows = []
    state_rows = []
    for name, desired in definitions.items():
        held = pd.DataFrame(
            delayed_positions(desired.to_numpy(float), index, "daily"), index=index, columns=symbols
        )
        simulation = simulation_from_held(held, open_returns, same, midnight, maker, taker)
        for period, mask in masks.items():
            metric_rows.append({"baseline": name, "period": period, **rich_metrics(simulation, mask, btc_returns)})
        state_rows.append(pd.DataFrame({
            "timestamp": index, "baseline": name, "net_return": simulation.net,
            "gross_return": simulation.gross_return, "fee": simulation.fee,
            "slippage": simulation.slippage, "funding": simulation.funding,
            "turnover": simulation.turnover,
        }))
    metrics = pd.DataFrame(metric_rows)
    state = pd.concat(state_rows, ignore_index=True)
    metrics.to_parquet(RESULTS / "baseline_metrics.parquet", index=False)
    state.to_parquet(RESULTS / "baseline_daily_state.parquet", index=False)
    return metrics, state


def strategy_dashboard(state: pd.DataFrame, rolling: pd.DataFrame, strategy: str) -> go.Figure:
    subset = state[state.strategy.eq(strategy) & state.timestamp.ge("2022-01-01")].sort_values("timestamp")
    diagnostics = rolling[rolling.strategy.eq(strategy) & rolling.timestamp.ge("2022-01-01")]
    equity = (1 + subset.net_return.fillna(0)).cumprod()
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=.07,
        subplot_titles=("Stitched equity", "Gross / net exposure", "Portfolio breadth", "Rolling net Sharpe"),
    )
    fig.add_trace(go.Scatter(x=subset.timestamp, y=equity, name="Growth of $1", line=dict(color="#7658c9")), row=1, col=1)
    fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.gross_exposure, name="Gross", line=dict(color="#4c94df")), row=2, col=1)
    fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.net_exposure, name="Net", line=dict(color="#268a87")), row=2, col=1)
    fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.long_count, name="Longs", line=dict(color="#268a87")), row=3, col=1)
    fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.short_count, name="Shorts", line=dict(color="#bd4e4e")), row=3, col=1)
    fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.top5_concentration, name="Top-5 concentration", line=dict(color="#ce9a48")), row=3, col=1)
    for window, color in ((90, "#ce9a48"), (365, "#66717e")):
        block = diagnostics[diagnostics.window_days.eq(window)]
        fig.add_trace(go.Scatter(x=block.timestamp, y=block.sharpe, name=f"{window}d Sharpe", line=dict(color=color)), row=4, col=1)
    style_plotly(fig, escape(STRATEGY_LABELS.get(strategy, strategy)), 1040)
    fig.update_yaxes(title="Growth", row=1, col=1)
    fig.update_yaxes(title="Weight", row=2, col=1)
    fig.update_yaxes(title="Count / share", row=3, col=1)
    fig.update_yaxes(title="Sharpe", row=4, col=1)
    return fig


def holdings_heatmap(positions: pd.DataFrame, strategy: str) -> go.Figure:
    subset = positions[positions.strategy.eq(strategy) & positions.timestamp.ge("2022-01-01")]
    pivot = subset.pivot(index="timestamp", columns="symbol", values="held_weight").fillna(0.0)
    top = pivot.abs().mean().nlargest(18).index.tolist()
    other = [symbol for symbol in pivot if symbol not in top]
    weekly = pivot[top].resample("W").last()
    if other:
        weekly["OTHER_NET"] = pivot[other].sum(axis=1).resample("W").last()
    fig = go.Figure(go.Heatmap(
        z=weekly.T.to_numpy(), x=weekly.index, y=weekly.columns,
        colorscale=[[0, "#bd4e4e"], [.5, "#f5f6f9"], [1, "#268a87"]], zmid=0,
        colorbar=dict(title="Held weight"),
    ))
    style_plotly(fig, f"Signed weekly holdings · {STRATEGY_LABELS.get(strategy, strategy)}", 680)
    fig.update_yaxes(title="")
    return fig


def selected_standard_table(selection: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family", "model", "validation_net_sharpe", "holdout_net_sharpe", "target_vol",
        "gross_cap", "ticker_risk_cap", "rebalance", "config_id",
    ]
    frame = selection[selection.selected][columns].copy()
    frame["family"] = frame.family.map(FAMILY_LABELS)
    return frame


def selected_rolling_table(selection: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fold", "family", "rank", "model", "selection_net_sharpe", "selection_annual_turnover",
        "target_vol", "gross_cap", "ticker_risk_cap", "rebalance", "config_id",
    ]
    frame = selection[selection.selected][columns].sort_values(["fold", "family", "rank"]).copy()
    frame["family"] = frame.family.map(FAMILY_LABELS)
    return frame


def roster_churn(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    chosen = selection[selection.selected].sort_values(["family", "selection_cutoff", "rank"])
    for family, group in chosen.groupby("family"):
        prior: set[str] = set()
        for fold, block in group.groupby("fold", sort=False):
            current = set(block.model)
            rows.append({
                "family": FAMILY_LABELS[family],
                "fold": fold,
                "retained_model_names": len(current & prior) if prior else np.nan,
                "new_model_names": len(current - prior) if prior else len(current),
                "roster_size": len(current),
            })
            prior = current
    return pd.DataFrame(rows)


def family_robustness(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in selection.groupby("family"):
        rows.append({
            "family": FAMILY_LABELS[family],
            "distinct_models": int(len(group)),
            "median_validation_sharpe": group.validation_net_sharpe.median(),
            "median_holdout_sharpe": group.holdout_net_sharpe.median(),
            "positive_holdout_share": group.holdout_net_sharpe.gt(0).mean(),
            "validation_holdout_rank_correlation": group.validation_net_sharpe.corr(
                group.holdout_net_sharpe, method="spearman"
            ),
            "selected_holdout_sharpe": group.loc[group.selected, "holdout_net_sharpe"].iloc[0],
        })
    return pd.DataFrame(rows)


def conclusions(
    standard_metrics: pd.DataFrame,
    standard_selection: pd.DataFrame,
    stitched: pd.DataFrame,
    folds: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> list[tuple[str, str]]:
    holdout = standard_metrics[standard_metrics.period.eq("holdout_2026")].set_index("family")
    ensemble = stitched[stitched.strategy.isin(["xs_momentum_ensemble", "xs_breakout_ensemble"])].set_index("strategy")
    winner = ensemble.net_sharpe.idxmax()
    loser = "xs_breakout_ensemble" if winner == "xs_momentum_ensemble" else "xs_momentum_ensemble"
    positive_folds = folds[folds.strategy.eq(winner)].net_sharpe.gt(0).sum()
    momentum_candidates = standard_selection[standard_selection.family.eq("xs_momentum")]
    breakout_candidates = standard_selection[standard_selection.family.eq("xs_breakout")]
    combined = stitched.set_index("strategy").loc["xs_combined_50_50"]
    breakout = stitched.set_index("strategy").loc["xs_breakout_ensemble"]
    baselines = baseline_metrics[baseline_metrics.period.eq("rolling_2022_2026")].set_index("baseline")
    fdv = baselines.loc["Current-FDV-weighted basket"]
    btc = baselines.loc["BTC perpetual buy/hold"]
    return [
        (
            "The standard holdout is a clean comparison, not a rolling deployment",
            f"Momentum recorded {num(holdout.loc['xs_momentum', 'net_sharpe'])} net Sharpe and breakout "
            f"{num(holdout.loc['xs_breakout', 'net_sharpe'])} in 2026 YTD after selection on 2024–25 only.",
        ),
        (
            "The 2024–25 momentum winner did not generalize",
            f"Its 2026 Sharpe was {num(holdout.loc['xs_momentum', 'net_sharpe'])}, even though the median momentum "
            f"construction delivered {num(momentum_candidates.holdout_net_sharpe.median())}. Breakout’s selected "
            f"{num(holdout.loc['xs_breakout', 'net_sharpe'])} was close to its family median of "
            f"{num(breakout_candidates.holdout_net_sharpe.median())}; model-selection stability is therefore a central result.",
        ),
        (
            f"{STRATEGY_LABELS[winner]} led the production-like test",
            f"Its stitched 2022–2026 YTD net Sharpe was {num(ensemble.loc[winner, 'net_sharpe'])}, versus "
            f"{num(ensemble.loc[loser, 'net_sharpe'])} for {STRATEGY_LABELS[loser].lower()}, and it was positive in "
            f"{positive_folds} of {folds.fold.nunique()} outer folds.",
        ),
        (
            "The 50/50 blend reduced drawdown, not improved Sharpe",
            f"The diagnostic blend produced {num(combined.net_sharpe)} Sharpe with {pct(combined.max_drawdown)} maximum "
            f"drawdown, versus breakout’s {num(breakout.net_sharpe)} Sharpe and {pct(breakout.max_drawdown)} drawdown. "
            "That is useful diversification, but not a superior return-to-volatility result.",
        ),
        (
            "Breakout did not clear the passive FDV benchmark on Sharpe",
            f"Breakout’s {num(breakout.net_sharpe)} stitched Sharpe was below the current-FDV basket’s "
            f"{num(fdv.net_sharpe)} and only modestly above BTC’s {num(btc.net_sharpe)}. Its practical advantage was "
            f"risk compression: {pct(breakout.annual_volatility)} volatility and {pct(breakout.max_drawdown)} drawdown, "
            f"versus {pct(fdv.annual_volatility)} and {pct(fdv.max_drawdown)} for the FDV basket.",
        ),
        (
            "Portfolio construction matters as much as the raw signal",
            "The search compares signed, dollar-neutral, ranked baskets, long-only, BTC-hedged, beta-constrained, "
            "residualized and buffered forms under the same risk, fees, slippage and funding engine.",
        ),
        (
            "This is historical causal evidence, not prospective proof",
            "Selections and internal refits are causal, but the top-100 universe is a current-FDV fallback and the large "
            "grid creates material winner’s-curse risk. Keep any deployment in shadow mode first.",
        ),
    ]


def main() -> None:
    manifest = json.loads((RESULTS / "study_manifest.json").read_text(encoding="utf-8"))
    standard_selection = pd.read_parquet(RESULTS / "standard_selection_ledger.parquet")
    standard_metrics = pd.read_parquet(RESULTS / "standard_selected_metrics.parquet")
    standard_state = pd.read_parquet(RESULTS / "standard_daily_state.parquet")
    rolling_selection = pd.read_parquet(RESULTS / "rolling_selection_ledger.parquet")
    stitched = pd.read_parquet(RESULTS / "rolling_stitched_metrics.parquet")
    folds = pd.read_parquet(RESULTS / "rolling_fold_metrics.parquet")
    state = pd.read_parquet(RESULTS / "rolling_daily_state.parquet")
    rolling = pd.read_parquet(RESULTS / "rolling_metrics.parquet")
    positions = pd.read_parquet(RESULTS / "rolling_daily_positions.parquet")
    baseline_metrics, baseline_state = build_baselines()
    for frame in (standard_state, state, rolling, positions):
        frame["timestamp"] = pd.to_datetime(frame.timestamp)
    baseline_state["timestamp"] = pd.to_datetime(baseline_state.timestamp)

    headline = stitched[stitched.strategy.isin(STRATEGY_LABELS)].set_index("strategy")
    standard_holdout = standard_metrics[standard_metrics.period.eq("holdout_2026")].set_index("family")
    combined = headline.loc["xs_combined_50_50"]
    worst_fold = folds[folds.strategy.eq("xs_combined_50_50")].sort_values("net_sharpe").iloc[0]
    findings = conclusions(standard_metrics, standard_selection, stitched, folds, baseline_metrics)

    body = hero(
        "ACAUSAL CAPITAL · CROSS-SECTIONAL TREND STUDY",
        "Breakout joins momentum,",
        "then faces a production-like rolling test.",
        "A causal portfolio study of cross-sectional EWMAC momentum and corrected Carver5 breakout, with a fixed historical holdout, annual outer folds, explicit roster selection and complete holdings reconstruction.",
        [
            "2024–25 validation / 2026 holdout", "Outer folds 2022–2026 YTD",
            "Top 3 models per family", "97-asset portfolio universe", "Next-open net returns",
        ],
    )
    body += '<nav><a href="#conclusion">Conclusion</a><a href="#standard">Standard holdout</a><a href="#rolling">Rolling test</a><a href="#selection">Selections</a><a href="#accounting">Accounting</a><a href="#method">Method</a><a href="#downloads">Downloads</a></nav>'
    body += '<section class="metrics">'
    body += metric("Momentum 2026 holdout Sharpe", num(standard_holdout.loc["xs_momentum", "net_sharpe"]), "Selected on 2024–25 only")
    body += metric("Breakout 2026 holdout Sharpe", num(standard_holdout.loc["xs_breakout", "net_sharpe"]), "Selected on 2024–25 only")
    body += metric("Rolling momentum Sharpe", num(headline.loc["xs_momentum_ensemble", "net_sharpe"]), "Stitched 2022–2026 YTD")
    body += metric("Rolling breakout Sharpe", num(headline.loc["xs_breakout_ensemble", "net_sharpe"]), "Stitched 2022–2026 YTD")
    body += metric("50/50 combined Sharpe", num(combined.net_sharpe), "Diagnostic, stitched outer folds")
    body += metric("Worst combined fold", num(worst_fold.net_sharpe), str(worst_fold.fold))
    body += "</section>"
    body += '<aside class="callout warning"><strong>Not a prospective track record.</strong> The 2026 segment is an untouched historical holdout and every rolling fold is selected causally, but the universe remains the documented current top-100 FDV fallback. Results are research/shadow evidence only.</aside>'
    body += '<section id="conclusion"><h2>Investment conclusion</h2><div class="findings">'
    for number, (title, text) in enumerate(findings, 1):
        body += finding(number, title, text)
    body += "</div></section>"

    standard_table = standard_metrics[standard_metrics.period.isin(["validation", "holdout_2026"])][[
        "family", "model", "period", "annual_return", "annual_volatility", "net_sharpe", "sortino",
        "calmar", "max_drawdown", "cumulative_return", "worst_month", "cvar_95", "positive_day_rate",
        "annual_turnover", "fees", "slippage", "funding", "btc_beta",
    ]].copy()
    standard_table["family"] = standard_table.family.map(FAMILY_LABELS)
    body += '<section id="standard"><h2>1. Standard holdout test</h2><p>The model and portfolio settings are selected once using net performance from 1 January 2024 through 31 December 2025. They are then frozen as a selection decision and measured on 1 January 2026 through 18 September 2026. Internal quarterly signal calibration remains causal and can refit using information available at the time; the holdout is excluded from model selection.</p>'
    body += figure_html(chart_html(standard_equity_figure(standard_state)), "Validation-to-holdout equity", "The dashed line marks the first untouched historical holdout observation.", "../data_store/crypto_momentum_research/cross_sectional_trend_v1/standard_daily_state.parquet")
    body += figure_html(chart_html(standard_scatter(standard_selection)), "All model constructions: validation versus holdout", "Each point is the validation-best risk configuration for one distinct signal/construction model. Wide dispersion is direct evidence of selection uncertainty.", "../data_store/crypto_momentum_research/cross_sectional_trend_v1/standard_selection_ledger.parquet")
    body += "<h3>Family-wide robustness, not only the winner</h3>" + table_html(family_robustness(standard_selection))
    body += "<h3>Selected results</h3>" + table_html(standard_table)
    body += "<h3>Selected exact configurations</h3>" + table_html(selected_standard_table(standard_selection)) + "</section>"

    stitched_table = stitched[stitched.strategy.isin(STRATEGY_LABELS)][[
        "strategy", "annual_return", "annual_volatility", "net_sharpe", "sortino", "calmar",
        "max_drawdown", "cumulative_return", "worst_month", "cvar_95", "positive_day_rate",
        "annual_turnover", "fees", "slippage", "funding", "btc_beta", "btc_correlation",
    ]].copy()
    stitched_table["strategy"] = stitched_table.strategy.map(STRATEGY_LABELS)
    fold_table = folds[folds.strategy.isin(STRATEGY_LABELS)][[
        "strategy", "fold", "annual_return", "annual_volatility", "net_sharpe", "max_drawdown",
        "annual_turnover", "funding", "btc_beta",
    ]].copy()
    fold_table["strategy"] = fold_table.strategy.map(STRATEGY_LABELS)
    body += '<section id="rolling"><h2>2. Production-like rolling selection</h2><p>At each 31 December cutoff, the complete candidate matrix is scored on the trailing 730 calendar days only. The top three distinct model names per family are locked for the next calendar fold. Their declared quarterly IC/scalar updates continue causally, activate on the next candle and themselves use the same trailing-history rule. Stitched metrics contain only the subsequent outer-fold returns.</p>'
    body += figure_html(chart_html(rolling_equity_figure(state)), "Stitched outer-fold equity", "No calibration-window return is included in these paths.", "../data_store/crypto_momentum_research/cross_sectional_trend_v1/rolling_daily_state.parquet")
    body += figure_html(chart_html(fold_figure(folds)), "Fold-by-fold stability", "The five folds reveal regime dependence hidden by one aggregate Sharpe.", "../data_store/crypto_momentum_research/cross_sectional_trend_v1/rolling_fold_metrics.parquet")
    body += figure_html(chart_html(rolling_sharpe_figure(rolling)), "Rolling performance diagnostics", "Trailing 365-day values are descriptive diagnostics, not a selection input.", "../data_store/crypto_momentum_research/cross_sectional_trend_v1/rolling_metrics.parquet")
    body += figure_html(chart_html(annual_figure(state)), "Annual return decomposition", "Calendar 2026 is year-to-date through the data cutoff.")
    body += figure_html(chart_html(baseline_figure(baseline_state)), "Reference portfolios", "These are comparison baselines, not members of the model-selection contest. They use next-open activation and the same fee, slippage, funding and futures-eligibility rules.", "../data_store/crypto_momentum_research/cross_sectional_trend_v1/baseline_daily_state.parquet")
    body += "<h3>Stitched metrics</h3>" + table_html(stitched_table)
    rolling_baselines = baseline_metrics[baseline_metrics.period.eq("rolling_2022_2026")][[
        "baseline", "annual_return", "annual_volatility", "net_sharpe", "max_drawdown", "cumulative_return",
        "annual_turnover", "fees", "slippage", "funding", "btc_beta",
    ]]
    body += "<h3>Reference portfolio metrics</h3>" + table_html(rolling_baselines)
    body += "<h3>Every outer fold</h3>" + table_html(fold_table) + "</section>"

    body += '<section id="selection"><h2>3. What the rolling process actually selected</h2><p>The table below is the production-like roster: three distinct momentum configurations and three distinct breakout configurations at every annual cutoff. Scores shown are prior-window net Sharpes; they are never outer-fold performance. The exact configuration JSON contains all IC horizons, construction choices, risk settings and execution assumptions.</p>'
    body += table_html(selected_rolling_table(rolling_selection))
    body += '<h3>Roster churn</h3><p>No model name is forced to persist. A zero in “retained model names” means all three selected signal/construction identities changed from the prior annual fold—direct evidence of selection instability, even when the outer-fold portfolio remains profitable.</p>' + table_html(roster_churn(rolling_selection))
    body += '<p><a href="crypto_cross_sectional_trend_composition_20260921.html"><strong>Open the complete portfolio-composition companion →</strong></a></p></section>'

    decomposition = state[state.strategy.isin(STRATEGY_LABELS) & state.timestamp.ge("2022-01-01")].groupby("strategy", as_index=False).agg(
        gross_pnl=("gross_return", "sum"), fees=("fee", "sum"), slippage=("slippage", "sum"),
        funding=("funding", "sum"), mean_gross_exposure=("gross_exposure", "mean"),
        mean_abs_net_exposure=("net_exposure", lambda x: x.abs().mean()),
        mean_top5_concentration=("top5_concentration", "mean"),
    )
    decomposition["strategy"] = decomposition.strategy.map(STRATEGY_LABELS)
    body += '<section id="accounting"><h2>4. Portfolio accounting, funding and concentration</h2><p>All portfolios execute at the next daily open, pay current symbol-specific Binance taker commission, incur five basis points of one-way slippage and settle archived historical funding events against the pre-rebalance position at coincident midnight events. Ensemble trades and costs are recalculated after cross-model netting.</p>'
    body += table_html(decomposition) + '<p>The companion report shows gross and net exposure, long/short breadth, top-five concentration and signed ticker weights for every model slot and ensemble through time.</p></section>'

    body += '<section id="method"><h2>5. Signal and test methodology</h2><h3>Cross-sectional EWMAC momentum</h3><p>Five causal EWMAC components—2/8, 4/16, 8/32, 16/64 and 32/128—are volatility normalized, causally scaled, capped at ±20 and combined using quarterly rank-IC estimates. The study compares 1-, 5-, 20- and equal 5/20-day IC targets.</p><h3>Cross-sectional Carver5 breakout</h3><p>Five complete-window breakouts—10, 20, 40, 80 and 160 days—use h/4 EWMA smoothing, causal crypto forecast scaling, component ±20 caps and Carver’s per-ticker cost-speed eligibility filter. There is no 320-day rule. The same four rank-IC targets then combine the breakout components before cross-sectional ranking.</p><h3>Portfolio constructions</h3><p>Each signal family is tested as signed unhedged, explicitly dollar-neutral continuous rank, top/bottom 10%, 20% and 30% baskets, positive-signal long-only, BTC-hedged, directly beta-constrained and BTC-residualized. The equal 5/20-day IC model also receives five- and ten-percentile weekly rank buffers. The corrected continuous-rank implementation demeans each finite-universe row before gross normalization; this removes the unintended net-long bias in the older shared helper.</p><h3>Risk matrix</h3><p>Every construction is crossed with 60/90/180/360-day risk windows, 15–40% annual volatility targets, 1–3× gross caps, 5–40% single-ticker risk caps and daily/weekly/monthly rebalancing. Buffered models are weekly only. All forecasts, weights, universe states and risk estimates are causal.</p><h3>Limitations</h3><ul><li>The top-100 portfolio universe is based on current FDV membership because complete point-in-time market-cap history is not available; this introduces survivor and availability bias.</li><li>The exhaustive grid creates multiple-testing and winner’s-curse risk. Raw net Sharpe is the requested ranking statistic, not an adjusted significance measure.</li><li>Historical funding is event-level, but slippage is a fixed sensitivity rather than order-book replay. Capacity estimates are not equivalent to executable fill guarantees.</li><li>Outer folds are causal historical simulations. They are stronger than a single holdout but remain weaker than an untouched prospective shadow record.</li><li>The 50/50 momentum/breakout portfolio is a diagnostic diversification view and is not separately optimized.</li></ul></section>'

    body += '<section id="downloads"><h2>6. Reproduction and complete artifacts</h2><p>'
    links = [
        ("Selected exact configurations", "selected_configurations.json"),
        ("Standard exhaustive configurations", "standard_holdout_tested_configurations.jsonl.gz"),
        ("Rolling exhaustive configurations", "rolling_730_tested_configurations.jsonl.gz"),
        ("Standard candidate ledger", "standard_holdout_candidate_scores.parquet"),
        ("Rolling candidate ledger", "rolling_730_candidate_scores.parquet"),
        ("Rolling selection ledger", "rolling_selection_ledger.parquet"),
        ("Daily portfolio state", "rolling_daily_state.parquet"),
        ("Daily ticker positions", "rolling_daily_positions.parquet"),
        ("Study manifest", "study_manifest.json"),
    ]
    body += " · ".join(
        f'<a href="../data_store/crypto_momentum_research/cross_sectional_trend_v1/{escape(file_name)}">{escape(label)}</a>'
        for label, file_name in links
    )
    body += '</p><pre>python scripts/execute_cross_sectional_trend_study.py&#10;python scripts/build_cross_sectional_trend_report.py</pre></section>'
    OUTPUT.write_text(render_page("Crypto Cross-Sectional Trend Study", body, plotly=True, accent="purple"), encoding="utf-8")

    composition_body = hero(
        "ACAUSAL CAPITAL · CROSS-SECTIONAL PORTFOLIO COMPOSITION",
        "Every rolling strategy,",
        "ticker by ticker through time.",
        "Exposure, breadth, concentration, rolling performance and signed weekly holdings for the six selected model slots, the two family ensembles and the diagnostic combination.",
        ["2022–2026 YTD", "Annual roster changes", "90/365-day Sharpe", "Top-18 ticker heatmaps", "Net of costs and funding"],
    )
    composition_body += '<nav><a href="#directory">Strategy directory</a><a href="#reading">How to read</a></nav>'
    strategy_order = [
        "xs_momentum_slot_1", "xs_momentum_slot_2", "xs_momentum_slot_3",
        "xs_breakout_slot_1", "xs_breakout_slot_2", "xs_breakout_slot_3",
        "xs_momentum_ensemble", "xs_breakout_ensemble", "xs_combined_50_50",
    ]
    for strategy in strategy_order:
        STRATEGY_LABELS.setdefault(strategy, strategy.replace("xs_", "").replace("_", " ").title())
    composition_body += '<section id="directory"><h2>Strategy directory</h2><p>Slot identity may change at each annual boundary; the main report’s selection table identifies the exact active model. Family ensembles equal-risk the three selected slots before portfolio-level cost reconciliation.</p>'
    metrics_by_strategy = stitched.set_index("strategy")
    for strategy in strategy_order:
        summary = metrics_by_strategy.loc[strategy]
        section_id = strategy.replace("_", "-")
        composition_body += f'<section id="{section_id}"><h2>{escape(STRATEGY_LABELS[strategy])}</h2><div class="metrics">'
        composition_body += metric("Net Sharpe", num(summary.net_sharpe), "Stitched outer folds")
        composition_body += metric("Annual return", pct(summary.annual_return), "Arithmetic annualized")
        composition_body += metric("Maximum drawdown", pct(summary.max_drawdown), "Stitched outer folds")
        composition_body += metric("Annual turnover", f"{summary.annual_turnover:,.1f}×", "Two-sided portfolio turnover")
        composition_body += metric("Funding", pct(summary.funding), "Cumulative return contribution") + "</div>"
        composition_body += figure_html(chart_html(strategy_dashboard(state, rolling, strategy)), "Performance, exposure and breadth", "Annual vertical transitions can reflect a new selected model as well as changing signals.")
        composition_body += figure_html(chart_html(holdings_heatmap(positions, strategy)), "Signed weekly holdings", "Top 18 tickers by mean absolute held weight plus the net weight of all remaining assets.", "../data_store/crypto_momentum_research/cross_sectional_trend_v1/rolling_daily_positions.parquet")
        contribution = positions[positions.strategy.eq(strategy) & positions.timestamp.ge("2022-01-01")].groupby("symbol", as_index=False).agg(
            net_contribution=("net_return_contribution", "sum"), mean_abs_weight=("held_weight", lambda x: x.abs().mean())
        ).sort_values("net_contribution", ascending=False)
        extremes = pd.concat([contribution.head(10), contribution.tail(10)]).drop_duplicates("symbol")
        composition_body += "<h3>Largest cumulative ticker contributors and detractors</h3>" + table_html(extremes) + "</section>"
    composition_body += '</section><section id="reading"><h2>How to read the composition charts</h2><ul><li>Green heatmap cells are long; red cells are short. White means near-zero held weight, not missing P&amp;L.</li><li>Heatmaps use weekly last observations for readability; all accounting uses daily held positions.</li><li>Top-five concentration is the share of gross exposure in the five largest absolute positions.</li><li>A slot can switch model identity at a new outer fold, while family ensembles absorb all three slot changes.</li></ul></section>'
    COMPOSITION_OUTPUT.write_text(render_page("Cross-Sectional Trend Portfolio Composition", composition_body, plotly=True, accent="blue"), encoding="utf-8")

    source_names = [
        "study_manifest.json", "standard_selection_ledger.parquet", "standard_selected_metrics.parquet",
        "standard_daily_state.parquet", "rolling_selection_ledger.parquet", "rolling_stitched_metrics.parquet",
        "rolling_fold_metrics.parquet", "rolling_daily_state.parquet", "rolling_metrics.parquet",
        "rolling_daily_positions.parquet", "selected_configurations.json",
        "baseline_metrics.parquet", "baseline_daily_state.parquet",
    ]
    report_manifest = {
        "schema_version": 1,
        "reports": [str(OUTPUT), str(COMPOSITION_OUTPUT)],
        "generator": "scripts/build_cross_sectional_trend_report.py",
        "rebuild": "python scripts/build_cross_sectional_trend_report.py",
        "study_methodology": manifest["methodology_version"],
        "inputs": {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in source_names},
    }
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(COMPOSITION_OUTPUT)


if __name__ == "__main__":
    main()
