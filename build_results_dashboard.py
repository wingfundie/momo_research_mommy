from __future__ import annotations

"""Plotly Dash dashboard for static and walk-forward backtest bundles."""

import argparse
import datetime as dt
import webbrowser
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from plotly.subplots import make_subplots

GRID_COLOR = "rgba(15, 23, 42, 0.10)"
DEFAULT_PORT = 8050
SOURCE_FULL = "full"
SOURCE_STATIC_OOS = "static_oos"
SOURCE_WFO = "wfo"

SORT_OPTIONS = [
    {"label": "WFO Sharpe", "value": "wfo_sharpe"},
    {"label": "WFO Return", "value": "wfo_return"},
    {"label": "Max Drawdown", "value": "wfo_mdd"},
    {"label": "Return / Max DD", "value": "return_over_mdd"},
    {"label": "Cost Drag", "value": "cost_drag_pct"},
    {"label": "Turnover", "value": "turnover"},
    {"label": "Windows", "value": "wfo_windows"},
    {"label": "Robust Score", "value": "robust_score"},
]

TABLE_COLUMNS = [
    {"name": "Ticker", "id": "ticker"},
    {"name": "WFO status", "id": "wfo_status"},
    {"name": "WFO windows", "id": "wfo_windows", "type": "numeric"},
    {"name": "First WFO date", "id": "first_wfo_date"},
    {"name": "Last WFO date", "id": "last_wfo_date"},
    {"name": "Static OOS Sharpe", "id": "static_oos_sharpe", "type": "numeric"},
    {"name": "WFO Sharpe", "id": "wfo_sharpe", "type": "numeric"},
    {"name": "Sharpe delta", "id": "sharpe_delta", "type": "numeric"},
    {"name": "Static OOS return", "id": "static_oos_return", "type": "numeric"},
    {"name": "WFO return", "id": "wfo_return", "type": "numeric"},
    {"name": "Return delta", "id": "return_delta", "type": "numeric"},
    {"name": "Static max DD", "id": "static_mdd", "type": "numeric"},
    {"name": "WFO max DD", "id": "wfo_mdd", "type": "numeric"},
    {"name": "Return / DD", "id": "return_over_mdd", "type": "numeric"},
    {"name": "CAGR", "id": "cagr", "type": "numeric"},
    {"name": "Deflated Sharpe", "id": "deflated_sharpe", "type": "numeric"},
    {"name": "T-stat", "id": "t_stat", "type": "numeric"},
    {"name": "Skew", "id": "skew", "type": "numeric"},
    {"name": "Lower 5% tail", "id": "lower_tail", "type": "numeric"},
    {"name": "Upper 95% tail", "id": "upper_tail", "type": "numeric"},
    {"name": "Avg exposure", "id": "avg_exposure", "type": "numeric"},
    {"name": "Max exposure", "id": "max_exposure", "type": "numeric"},
    {"name": "Turnover", "id": "turnover", "type": "numeric"},
    {"name": "Total fees", "id": "total_fees", "type": "numeric"},
    {"name": "Total cost", "id": "total_cost", "type": "numeric"},
    {"name": "Cost drag %", "id": "cost_drag_pct", "type": "numeric"},
    {"name": "Max gap days", "id": "max_gap_days", "type": "numeric"},
    {"name": "Days stale", "id": "days_stale", "type": "numeric"},
    {"name": "Robust score", "id": "robust_score", "type": "numeric"},
    {"name": "Quality flag", "id": "quality_flag"},
]


def _periods_per_year(freq: str) -> int:
    """Return the number of periods per year for a data frequency string."""
    return {"1d": 365, "4h": 365 * 6, "1h": 365 * 24}.get(freq, 365)


def _safe_float(value: Any) -> float:
    """Coerce a value to float, returning NaN on failure."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_sum(series: pd.Series | None) -> float:
    if series is None or series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).sum())


def _pandas_display_freq(freq: str) -> str:
    if freq.endswith("h"):
        return freq
    return "D"


def _fmt_pct(value: Any) -> str:
    value = _safe_float(value)
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def _fmt_num(value: Any) -> str:
    value = _safe_float(value)
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def _fmt_currency(value: Any) -> str:
    value = _safe_float(value)
    return "n/a" if pd.isna(value) else f"${value:,.0f}"


def _fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    value = _safe_float(value)
    return None if pd.isna(value) else round(value, digits)


def _annualized_sharpe(returns: pd.Series, freq: str) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    std = returns.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(returns.mean() / std * np.sqrt(_periods_per_year(freq)))


def _rolling_sharpe(returns: pd.Series, freq: str, window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(index=returns.index, dtype=float)

    def _roll(x: pd.Series) -> float:
        std = x.std()
        if std == 0 or np.isnan(std):
            return float("nan")
        return float(x.mean() / std * np.sqrt(_periods_per_year(freq)))

    return returns.rolling(window).apply(_roll, raw=False)


def _compute_metrics(returns: pd.Series, freq: str) -> Dict[str, float]:
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "sharpe": float("nan"),
            "volatility": float("nan"),
            "max_drawdown": float("nan"),
        }
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    periods = _periods_per_year(freq)
    cagr = float((1.0 + total_return) ** (periods / max(1, len(returns))) - 1.0) if total_return > -1 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": _annualized_sharpe(returns, freq),
        "volatility": float(returns.std() * np.sqrt(periods)),
        "max_drawdown": float(drawdown.min()),
    }


def _find_latest_bundle(data_dir: Path) -> Path | None:
    """Return the newest WFO bundle first, then the newest root data-store bundle."""
    wfo_dir = data_dir / "wfo_runs"
    if wfo_dir.exists():
        wfo_bundles = sorted(wfo_dir.glob("backtest_results_bundle_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if wfo_bundles:
            return wfo_bundles[0]
    bundles = sorted(data_dir.glob("backtest_results_bundle_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return bundles[0] if bundles else None


def get_wfo_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    wf = payload.get("walkforward")
    return wf if isinstance(wf, dict) else {}


def extract_metric(payload: Dict[str, Any], source: str, metric_name: str) -> float:
    if source == SOURCE_WFO:
        metrics = get_wfo_payload(payload).get("metrics_oos") or {}
    elif source == SOURCE_STATIC_OOS:
        metrics = payload.get("metrics_oos") or {}
    else:
        metrics = payload.get("metrics") or {}
    return _safe_float(metrics.get(metric_name))


def _series_for_source(payload: Dict[str, Any], source: str, first_oos_ts: Any = None) -> pd.DataFrame:
    if source == SOURCE_WFO:
        series = get_wfo_payload(payload).get("series")
        return series.copy() if isinstance(series, pd.DataFrame) else pd.DataFrame()

    series = payload.get("series")
    if not isinstance(series, pd.DataFrame):
        return pd.DataFrame()
    series = series.copy()
    if source == SOURCE_STATIC_OOS and first_oos_ts is not None and "returns" in series:
        series = series.loc[series.index >= pd.Timestamp(first_oos_ts)]
    return series


def _returns_for_source(payload: Dict[str, Any], source: str, first_oos_ts: Any = None) -> pd.Series:
    series = _series_for_source(payload, source, first_oos_ts)
    if series.empty or "returns" not in series:
        return pd.Series(dtype=float)
    return pd.to_numeric(series["returns"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _rebase_equity(equity: pd.Series) -> pd.Series:
    equity = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if equity.empty:
        return equity
    first = equity.iloc[0]
    return equity / first if first != 0 else equity


def _calendarize_series(series: pd.Series, freq: str) -> pd.Series:
    """Reindex a series to its expected display calendar so charts show real gaps."""
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.dropna(how="all")
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return series
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=_pandas_display_freq(freq))
    out = series.reindex(full_index)
    out.index.name = series.index.name
    return out


def _coverage_stats(series: pd.DataFrame, global_last_ts: pd.Timestamp | None) -> Dict[str, float]:
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return {"max_gap_days": float("nan"), "days_stale": float("nan")}
    idx = pd.DatetimeIndex(series.index).sort_values()
    diffs = idx.to_series().diff().dropna()
    max_gap_days = float(diffs.max().days) if not diffs.empty else 0.0
    days_stale = float((global_last_ts - idx.max()).days) if global_last_ts is not None else float("nan")
    return {"max_gap_days": max_gap_days, "days_stale": days_stale}


def _raw_price_coverage_series(payload: Dict[str, Any]) -> pd.DataFrame:
    series = payload.get("series")
    if not isinstance(series, pd.DataFrame) or series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return pd.DataFrame()
    if "price" in series:
        price = pd.to_numeric(series["price"], errors="coerce").dropna()
        return price.to_frame("price")
    return series.dropna(how="all")


def compute_drawdown(returns_or_equity: pd.Series) -> pd.Series:
    """Compute drawdown from either a returns series or an equity series."""
    series = pd.Series(returns_or_equity).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return pd.Series(dtype=float)
    first = _safe_float(series.iloc[0])
    looks_like_equity = pd.notna(first) and first > 0.5
    equity = series if looks_like_equity else (1.0 + series.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def compute_cost_drag(series: pd.DataFrame, portfolio_value: float) -> float:
    if series.empty or "total_cost" not in series or portfolio_value == 0:
        return float("nan")
    return _safe_sum(series["total_cost"]) / float(portfolio_value)


def compute_exposure_stats(series: pd.DataFrame) -> Dict[str, float]:
    if series.empty or "positions_usd" not in series:
        return {"avg_exposure": float("nan"), "max_exposure": float("nan")}
    exposure = pd.to_numeric(series["positions_usd"], errors="coerce").abs().replace([np.inf, -np.inf], np.nan)
    return {"avg_exposure": float(exposure.mean()), "max_exposure": float(exposure.max())}


def compute_turnover_stats(series: pd.DataFrame) -> Dict[str, float]:
    if series.empty or "turnover_notional" not in series:
        return {"turnover": float("nan")}
    return {"turnover": _safe_sum(series["turnover_notional"])}


def _quality_flag(
    ticker: str,
    wfo_status: str,
    wfo_windows: int,
    wfo_mdd: float,
    max_gap_days: float,
    days_stale: float,
) -> str:
    flags: List[str] = []
    if wfo_status != "success":
        flags.append(wfo_status)
    if wfo_status == "success" and wfo_windows < 3:
        flags.append("low sample")
    if pd.notna(max_gap_days) and max_gap_days > 7:
        flags.append(f"data gap {max_gap_days:.0f}d")
    if pd.notna(days_stale) and days_stale > 30:
        flags.append(f"stale {days_stale:.0f}d")
    if pd.notna(wfo_mdd) and wfo_mdd <= -0.75:
        flags.append("extreme DD")
    if ticker.upper().startswith(("USDC", "FDUSD", "TUSD", "USDP")):
        flags.append("stablecoin-like")
    return ", ".join(flags) if flags else "ok"


def build_universe_summary(bundle: Dict[str, Any], strategy: str = "breakout") -> pd.DataFrame:
    """Build one analysis row per ticker with WFO/static comparison fields."""
    section = bundle.get(strategy, {}) or {}
    results = section.get("results", {}) or {}
    first_oos_ts = section.get("first_oos_ts")
    portfolio_value = _safe_float(bundle.get("meta", {}).get("portfolio_value"))
    portfolio_value = portfolio_value if pd.notna(portfolio_value) and portfolio_value != 0 else 1.0
    rows: List[Dict[str, Any]] = []
    global_last_ts: pd.Timestamp | None = None
    for payload in results.values():
        coverage_series = _raw_price_coverage_series(payload)
        if not coverage_series.empty:
            max_ts = pd.Timestamp(coverage_series.index.max())
            global_last_ts = max(max_ts, global_last_ts) if global_last_ts is not None else max_ts

    for ticker, payload in sorted(results.items()):
        wf = get_wfo_payload(payload)
        wf_status = str(wf.get("status", "missing"))
        wf_metrics = wf.get("metrics_oos") or {}
        static_metrics = payload.get("metrics_oos") or {}
        source_series = _series_for_source(payload, SOURCE_WFO, first_oos_ts)
        if source_series.empty:
            source_series = _series_for_source(payload, SOURCE_STATIC_OOS, first_oos_ts)

        exposure = compute_exposure_stats(source_series)
        turnover = compute_turnover_stats(source_series)
        cost_drag = compute_cost_drag(source_series, portfolio_value)
        coverage = _coverage_stats(_raw_price_coverage_series(payload), global_last_ts)
        total_fees = _safe_sum(source_series["fees"]) if "fees" in source_series else float("nan")
        total_cost = _safe_sum(source_series["total_cost"]) if "total_cost" in source_series else float("nan")

        wfo_sharpe = _safe_float(wf_metrics.get("sharpe"))
        wfo_return = _safe_float(wf_metrics.get("total_return"))
        wfo_mdd = _safe_float(wf_metrics.get("max_drawdown"))
        static_sharpe = _safe_float(static_metrics.get("sharpe"))
        static_return = _safe_float(static_metrics.get("total_return"))
        static_mdd = _safe_float(static_metrics.get("max_drawdown"))
        wfo_windows = int(wf.get("n_windows", 0) or 0)
        return_over_mdd = wfo_return / abs(wfo_mdd) if pd.notna(wfo_return) and pd.notna(wfo_mdd) and wfo_mdd != 0 else float("nan")
        robust_score = (
            wfo_sharpe + 0.5 * wfo_return - abs(wfo_mdd) - 0.1 * cost_drag
            if wf_status == "success" and all(pd.notna(x) for x in (wfo_sharpe, wfo_return, wfo_mdd, cost_drag))
            else float("nan")
        )

        rows.append(
            {
                "ticker": ticker,
                "static_status": (payload.get("params") or {}).get("status", "missing"),
                "wfo_status": wf_status,
                "wfo_windows": wfo_windows,
                "first_wfo_date": _fmt_date(wf.get("first_oos_ts")),
                "last_wfo_date": _fmt_date(wf.get("last_oos_ts")),
                "static_oos_sharpe": static_sharpe,
                "wfo_sharpe": wfo_sharpe,
                "sharpe_delta": wfo_sharpe - static_sharpe if pd.notna(wfo_sharpe) and pd.notna(static_sharpe) else float("nan"),
                "static_oos_return": static_return,
                "wfo_return": wfo_return,
                "return_delta": wfo_return - static_return if pd.notna(wfo_return) and pd.notna(static_return) else float("nan"),
                "static_mdd": static_mdd,
                "wfo_mdd": wfo_mdd,
                "return_over_mdd": return_over_mdd,
                "cagr": _safe_float(wf_metrics.get("cagr")),
                "deflated_sharpe": _safe_float(wf_metrics.get("deflated_sharpe")),
                "t_stat": _safe_float(wf_metrics.get("t_stat")),
                "skew": _safe_float(wf_metrics.get("skew")),
                "lower_tail": _safe_float(wf_metrics.get("lower_tail")),
                "upper_tail": _safe_float(wf_metrics.get("upper_tail")),
                "avg_exposure": exposure["avg_exposure"],
                "max_exposure": exposure["max_exposure"],
                "turnover": turnover["turnover"],
                "total_fees": total_fees,
                "total_cost": total_cost,
                "cost_drag_pct": cost_drag,
                "max_gap_days": coverage["max_gap_days"],
                "days_stale": coverage["days_stale"],
                "robust_score": robust_score,
                "quality_flag": _quality_flag(
                    ticker,
                    wf_status,
                    wfo_windows,
                    wfo_mdd,
                    coverage["max_gap_days"],
                    coverage["days_stale"],
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """Backward-compatible summary used by older tests and rankings."""
    rows = []
    for ticker, payload in results.items():
        metrics = payload.get("metrics") or {}
        total_return = _safe_float(metrics.get("total_return"))
        sharpe = _safe_float(metrics.get("sharpe"))
        rows.append(
            {
                "ticker": ticker,
                "total_return": total_return,
                "abs_return": abs(total_return) if pd.notna(total_return) else float("nan"),
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows).set_index("ticker").sort_index()


def _empty_figure(message: str, *, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        autosize=False,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=20, t=40, b=30),
        font=dict(family="Space Grotesk, sans-serif", color="#0f172a", size=12),
    )
    return fig


def _style_figure(fig: go.Figure, title: str, *, height: int = 340, y_tickformat: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        autosize=False,
        height=height,
        margin=dict(l=45, r=25, t=55, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="Space Grotesk, sans-serif", color="#0f172a", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1.0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, tickformat=y_tickformat)
    return fig


def _bar_figure(df: pd.DataFrame, value_col: str, title: str, *, n: int = 20, ascending: bool = False, pct: bool = False) -> go.Figure:
    if df.empty or value_col not in df:
        return _empty_figure("No data available.")
    data = df.dropna(subset=[value_col]).sort_values(value_col, ascending=ascending).head(n)
    if data.empty:
        return _empty_figure("No data available.")
    colors = ["#15803d" if value >= 0 else "#b91c1c" for value in data[value_col]]
    fig = go.Figure(
        go.Bar(
            x=data[value_col],
            y=data["ticker"],
            orientation="h",
            marker=dict(color=colors),
            text=[_fmt_pct(v) if pct else _fmt_num(v) for v in data[value_col]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _style_figure(fig, title, height=420, y_tickformat=".0%" if pct else None)


def _histogram(df: pd.DataFrame, col: str, title: str, *, pct: bool = False) -> go.Figure:
    values = pd.to_numeric(df.get(col), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return _empty_figure("No data available.")
    fig = go.Figure(go.Histogram(x=values, nbinsx=45, marker=dict(color="#2563eb"), opacity=0.85))
    return _style_figure(fig, title, y_tickformat=None, height=320).update_xaxes(tickformat=".0%" if pct else ".2f")


def _scatter_return_drawdown(df: pd.DataFrame) -> go.Figure:
    data = df[df["wfo_status"] == "success"].dropna(subset=["wfo_return", "wfo_mdd", "wfo_sharpe"])
    if data.empty:
        return _empty_figure("No WFO successes available.")
    size = (pd.to_numeric(data["wfo_windows"], errors="coerce").fillna(1).clip(lower=1) * 3 + 8).tolist()
    fig = go.Figure(
        go.Scatter(
            x=data["wfo_mdd"],
            y=data["wfo_return"],
            mode="markers",
            text=data["ticker"],
            marker=dict(size=size, color=data["wfo_sharpe"], colorscale="RdYlGn", showscale=True, colorbar=dict(title="Sharpe")),
            hovertemplate="%{text}<br>Return %{y:.2%}<br>Max DD %{x:.2%}<br>Sharpe %{marker.color:.2f}<extra></extra>",
        )
    )
    return _style_figure(fig, "WFO Return vs Max Drawdown", height=390).update_xaxes(tickformat=".0%").update_yaxes(tickformat=".0%")


def _scatter_static_vs_wfo(df: pd.DataFrame) -> go.Figure:
    data = df.dropna(subset=["static_oos_sharpe", "wfo_sharpe"])
    if data.empty:
        return _empty_figure("No static/WFO comparison available.")
    lo = float(np.nanmin([data["static_oos_sharpe"].min(), data["wfo_sharpe"].min()]))
    hi = float(np.nanmax([data["static_oos_sharpe"].max(), data["wfo_sharpe"].max()]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["static_oos_sharpe"],
            y=data["wfo_sharpe"],
            text=data["ticker"],
            mode="markers",
            marker=dict(color=data["sharpe_delta"], colorscale="RdYlGn", showscale=True, colorbar=dict(title="Delta")),
            hovertemplate="%{text}<br>Static %{x:.2f}<br>WFO %{y:.2f}<extra></extra>",
            name="Ticker",
        )
    )
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#64748b", dash="dash"), name="Equal"))
    return _style_figure(fig, "Static OOS Sharpe vs WFO Sharpe", height=390)


def _build_kpis(summary: pd.DataFrame, portfolio: Dict[str, Any]) -> html.Div:
    success = summary[summary["wfo_status"] == "success"]
    cards = [
        ("Total tickers", f"{len(summary):,}"),
        ("WFO success", f"{len(success):,}"),
        ("Skipped", f"{int((summary['wfo_status'] == 'skipped_insufficient_data').sum()):,}"),
        ("Median WFO Sharpe", _fmt_num(success["wfo_sharpe"].median())),
        ("Median WFO Return", _fmt_pct(success["wfo_return"].median())),
        ("Median WFO Max DD", _fmt_pct(success["wfo_mdd"].median())),
        ("EW WFO Return", _fmt_pct(portfolio.get("metrics", {}).get("total_return"))),
        ("EW WFO Sharpe", _fmt_num(portfolio.get("metrics", {}).get("sharpe"))),
        ("EW WFO Max DD", _fmt_pct(portfolio.get("metrics", {}).get("max_drawdown"))),
    ]
    return html.Div(
        className="kpi-grid",
        children=[
            html.Div([html.Span(label, className="kpi-label"), html.Strong(value, className="kpi-value")], className="kpi-card")
            for label, value in cards
        ],
    )


def _format_metrics_row(
    label: str,
    metrics: Dict[str, float] | None,
    *,
    information_ratio: float | None = None,
) -> Dict[str, str]:
    metrics = metrics or {}
    return {
        "Segment": label,
        "Sharpe": _fmt_num(metrics.get("sharpe")),
        "Deflated Sharpe": _fmt_num(metrics.get("deflated_sharpe")),
        "CAGR": _fmt_pct(metrics.get("cagr")),
        "Max Drawdown": _fmt_pct(metrics.get("max_drawdown")),
        "Total Return": _fmt_pct(metrics.get("total_return")),
        "Skew": _fmt_num(metrics.get("skew")),
        "Lower Tail (5%)": _fmt_pct(metrics.get("lower_tail")),
        "Upper Tail (95%)": _fmt_pct(metrics.get("upper_tail")),
        "T-Stat": _fmt_num(metrics.get("t_stat")),
        "Information Ratio": _fmt_num(information_ratio),
    }


def _build_detail_figure(series: pd.DataFrame, ticker: str, strategy: str, freq: str) -> go.Figure:
    if series.empty:
        return _empty_figure("No ticker series available.", height=680)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
        subplot_titles=("Price and Signal", "Position USD", "Net Returns"),
    )
    if "price" in series:
        price = _calendarize_series(series["price"], freq)
        fig.add_trace(
            go.Scatter(
                x=price.index,
                y=price,
                name="Price",
                line=dict(color="#f97316", width=1.8),
                connectgaps=False,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    if "signal" in series:
        signal = _calendarize_series(series["signal"], freq)
        fig.add_trace(
            go.Scatter(
                x=signal.index,
                y=signal,
                name="Signal",
                line=dict(color="#1d4ed8", width=1.5),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
    if "positions_usd" in series:
        positions = _calendarize_series(series["positions_usd"], freq)
        fig.add_trace(
            go.Scatter(
                x=positions.index,
                y=positions,
                name="Position USD",
                fill="tozeroy",
                line=dict(color="#0f766e", width=1.5),
                connectgaps=False,
            ),
            row=2,
            col=1,
        )
    if "returns" in series:
        returns = _calendarize_series(series["returns"], freq)
        fig.add_trace(go.Bar(x=returns.index, y=returns, name="Returns", marker=dict(color="#64748b")), row=3, col=1)
    return _style_figure(fig, f"{strategy.title()} | {ticker} | Price, Signal, Position", height=720)


def _build_equity_comparison(
    payload: Dict[str, Any],
    ticker: str,
    benchmark_returns: pd.Series | None,
    *,
    first_oos_ts: Any,
    freq: str,
) -> go.Figure:
    fig = go.Figure()
    static = _series_for_source(payload, SOURCE_FULL)
    static_oos = _series_for_source(payload, SOURCE_STATIC_OOS, first_oos_ts)
    wfo = _series_for_source(payload, SOURCE_WFO)
    if "equity" in static:
        equity = _calendarize_series(_rebase_equity(static["equity"]), freq)
        fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Static full", line=dict(color="#64748b", width=1.6), connectgaps=False))
    if "equity" in static_oos:
        equity = _calendarize_series(_rebase_equity(static_oos["equity"]), freq)
        fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Static OOS", line=dict(color="#2563eb", width=2), connectgaps=False))
    if "equity" in wfo:
        equity = _calendarize_series(_rebase_equity(wfo["equity"]), freq)
        fig.add_trace(go.Scatter(x=equity.index, y=equity, name="WFO OOS", line=dict(color="#16a34a", width=2.2), connectgaps=False))
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_eq = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
        bench_eq = _calendarize_series(_rebase_equity(bench_eq), freq)
        fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, name="Benchmark", line=dict(color="#f97316", width=1.8, dash="dot"), connectgaps=False))
    if not fig.data:
        return _empty_figure("No equity data available.")
    return _style_figure(fig, f"{ticker} Equity Comparison", height=390)


def _build_drawdown_figure(payload: Dict[str, Any], ticker: str, *, first_oos_ts: Any, freq: str) -> go.Figure:
    static_oos = _series_for_source(payload, SOURCE_STATIC_OOS, first_oos_ts)
    wfo = _series_for_source(payload, SOURCE_WFO)
    fig = go.Figure()
    if "equity" in static_oos:
        dd = _calendarize_series(compute_drawdown(static_oos["equity"]), freq)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Static OOS", line=dict(color="#2563eb", width=2), connectgaps=False))
        if not dd.empty:
            fig.add_trace(go.Scatter(x=[dd.idxmin()], y=[dd.min()], name="Static max DD", mode="markers", marker=dict(color="#1e40af", size=9)))
    if "equity" in wfo:
        dd = _calendarize_series(compute_drawdown(wfo["equity"]), freq)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name="WFO OOS", line=dict(color="#16a34a", width=2), connectgaps=False))
        if not dd.empty:
            fig.add_trace(go.Scatter(x=[dd.idxmin()], y=[dd.min()], name="WFO max DD", mode="markers", marker=dict(color="#15803d", size=9)))
    if not fig.data:
        return _empty_figure("No drawdown data available.")
    return _style_figure(fig, f"{ticker} Drawdown", height=340).update_yaxes(tickformat=".0%")


def _build_return_distribution(payload: Dict[str, Any], ticker: str, *, first_oos_ts: Any) -> go.Figure:
    static_returns = _returns_for_source(payload, SOURCE_STATIC_OOS, first_oos_ts)
    wfo_returns = _returns_for_source(payload, SOURCE_WFO)
    fig = go.Figure()
    if not static_returns.empty:
        fig.add_trace(go.Histogram(x=static_returns, name="Static OOS", nbinsx=40, opacity=0.55, marker=dict(color="#2563eb")))
    if not wfo_returns.empty:
        fig.add_trace(go.Histogram(x=wfo_returns, name="WFO OOS", nbinsx=40, opacity=0.65, marker=dict(color="#16a34a")))
        for label, value, color in [
            ("Mean", wfo_returns.mean(), "#0f172a"),
            ("5%", wfo_returns.quantile(0.05), "#b91c1c"),
            ("95%", wfo_returns.quantile(0.95), "#15803d"),
        ]:
            fig.add_vline(x=value, line=dict(color=color, dash="dash"), annotation_text=label)
    if not fig.data:
        return _empty_figure("No return distribution available.")
    fig.update_layout(barmode="overlay")
    return _style_figure(fig, f"{ticker} Return Distribution", height=340).update_xaxes(tickformat=".2%")


def _build_rolling_performance(payload: Dict[str, Any], ticker: str, freq: str, *, first_oos_ts: Any) -> go.Figure:
    returns = _returns_for_source(payload, SOURCE_WFO, first_oos_ts)
    label = "WFO"
    if returns.empty:
        returns = _returns_for_source(payload, SOURCE_STATIC_OOS, first_oos_ts)
        label = "Static OOS"
    if returns.empty:
        return _empty_figure("No returns available.")
    window = 180 if freq == "1d" else max(20, int(_periods_per_year(freq) * 0.5))
    rolling_sharpe = _rolling_sharpe(returns, freq, window)
    rolling_vol = returns.rolling(window).std() * np.sqrt(_periods_per_year(freq))
    rolling_return = (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0
    rolling_sharpe = _calendarize_series(rolling_sharpe, freq)
    rolling_vol = _calendarize_series(rolling_vol, freq)
    rolling_return = _calendarize_series(rolling_return, freq)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            name=f"{label} Sharpe",
            line=dict(color="#2563eb", width=2),
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            name="Vol",
            line=dict(color="#f97316", width=1.6),
            yaxis="y2",
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_return.index,
            y=rolling_return,
            name="Return",
            line=dict(color="#16a34a", width=1.6),
            yaxis="y3",
            connectgaps=False,
        )
    )
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False),
        yaxis3=dict(overlaying="y", side="right", position=0.94, tickformat=".0%", showgrid=False),
    )
    return _style_figure(fig, f"{ticker} Rolling Performance ({window} bars)", height=360)


def _build_cost_turnover(payload: Dict[str, Any], ticker: str, portfolio_value: float, *, first_oos_ts: Any, freq: str) -> go.Figure:
    series = _series_for_source(payload, SOURCE_WFO, first_oos_ts)
    if series.empty:
        series = _series_for_source(payload, SOURCE_STATIC_OOS, first_oos_ts)
    if series.empty:
        return _empty_figure("No cost data available.")
    fig = go.Figure()
    if "fees" in series:
        fees = _calendarize_series(series["fees"].fillna(0.0).cumsum(), freq)
        fig.add_trace(go.Scatter(x=fees.index, y=fees, name="Cumulative fees", line=dict(color="#2563eb", width=2), connectgaps=False))
    if "total_cost" in series:
        cumulative = _calendarize_series(series["total_cost"].fillna(0.0).cumsum(), freq)
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative, name="Cumulative total cost", line=dict(color="#b91c1c", width=2), connectgaps=False))
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative / portfolio_value, name="Cost drag", line=dict(color="#f97316", width=1.8), yaxis="y2", connectgaps=False))
    if "turnover_notional" in series:
        turnover = _calendarize_series(series["turnover_notional"], freq)
        fig.add_trace(go.Bar(x=turnover.index, y=turnover, name="Turnover", marker=dict(color="rgba(100,116,139,0.35)")))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".1%", showgrid=False))
    return _style_figure(fig, f"{ticker} Costs and Turnover", height=360)


def _build_weight_stability(weights: pd.DataFrame | None) -> go.Figure:
    if weights is None or weights.empty:
        return _empty_figure("No walk-forward weights available.")
    fig = go.Figure()
    palette = ["#2563eb", "#14b8a6", "#f97316", "#e11d48", "#7c3aed", "#10b981"]
    for idx, column in enumerate(weights.columns):
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[column],
                name=str(column),
                stackgroup="one",
                line=dict(width=0.7, color=palette[idx % len(palette)]),
            )
        )
    return _style_figure(fig, "Parameter Stability (WFO Breakout Weights)", height=340).update_yaxes(tickformat=".0%")


def _wfo_window_rows(payload: Dict[str, Any], freq: str) -> List[Dict[str, Any]]:
    wf = get_wfo_payload(payload)
    weights = wf.get("weights")
    series = wf.get("series")
    if not isinstance(weights, pd.DataFrame) or weights.empty or not isinstance(series, pd.DataFrame) or series.empty:
        return []
    dates = list(pd.to_datetime(weights.index))
    rows: List[Dict[str, Any]] = []
    for idx, start in enumerate(dates):
        end = dates[idx + 1] if idx + 1 < len(dates) else pd.Timestamp(series.index.max()) + pd.Timedelta(days=1)
        window_series = series.loc[(series.index >= start) & (series.index < end)]
        returns = window_series["returns"] if "returns" in window_series else pd.Series(dtype=float)
        metrics = _compute_metrics(returns, freq)
        weight_row = weights.iloc[idx]
        dominant = str(weight_row.astype(float).idxmax()) if not weight_row.empty else ""
        rows.append(
            {
                "Refit date": _fmt_date(start),
                "Test start": _fmt_date(window_series.index.min()) if not window_series.empty else _fmt_date(start),
                "Test end": _fmt_date(window_series.index.max()) if not window_series.empty else "",
                "Return": _fmt_pct(metrics["total_return"]),
                "Sharpe": _fmt_num(metrics["sharpe"]),
                "Max DD": _fmt_pct(metrics["max_drawdown"]),
                "Dominant horizon": dominant,
                "Turnover": _fmt_currency(_safe_sum(window_series["turnover_notional"]) if "turnover_notional" in window_series else float("nan")),
            }
        )
    return rows


def build_equal_weight_portfolio(
    selected_tickers: Iterable[str],
    source: str,
    results: Dict[str, Any],
    *,
    first_oos_ts: Any = None,
    portfolio_value: float = 1.0,
    freq: str = "1d",
) -> Dict[str, Any]:
    """Build an equal-weight research basket from selected ticker return streams."""
    return_map: Dict[str, pd.Series] = {}
    cost_map: Dict[str, pd.Series] = {}
    for ticker in selected_tickers:
        payload = results.get(ticker)
        if not payload:
            continue
        series = _series_for_source(payload, source, first_oos_ts)
        if series.empty or "returns" not in series:
            continue
        return_map[ticker] = pd.to_numeric(series["returns"], errors="coerce")
        if "total_cost" in series:
            cost_map[ticker] = pd.to_numeric(series["total_cost"], errors="coerce") / float(portfolio_value or 1.0)

    if not return_map:
        return {
            "returns": pd.Series(dtype=float),
            "equity": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float),
            "cost_drag": pd.Series(dtype=float),
            "contributions": pd.Series(dtype=float),
            "metrics": _compute_metrics(pd.Series(dtype=float), freq),
            "avg_pairwise_corr": float("nan"),
        }

    returns_df = pd.DataFrame(return_map).replace([np.inf, -np.inf], np.nan)
    returns = returns_df.mean(axis=1, skipna=True).dropna()
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    costs = pd.DataFrame(cost_map).replace([np.inf, -np.inf], np.nan).mean(axis=1, skipna=True).fillna(0.0).cumsum() if cost_map else pd.Series(dtype=float)
    contributions = (1.0 + returns_df.fillna(0.0)).prod() - 1.0
    corr = returns_df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    return {
        "returns": returns,
        "equity": _calendarize_series(equity, freq),
        "drawdown": _calendarize_series(compute_drawdown(equity), freq),
        "cost_drag": _calendarize_series(costs, freq) if not costs.empty else costs,
        "contributions": contributions.sort_values(ascending=False),
        "metrics": _compute_metrics(returns, freq),
        "avg_pairwise_corr": float(upper.mean()) if not upper.empty else float("nan"),
    }


def _prepare_strategies(bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    strategies: Dict[str, Dict[str, Any]] = {}
    for strategy in ("momentum", "breakout"):
        section = bundle.get(strategy, {}) or {}
        results = section.get("results", {}) or {}
        if results:
            strategies[strategy] = {
                "summary": build_universe_summary(bundle, strategy),
                "legacy_summary": _build_summary(results),
                "results": results,
                "meta": section,
                "benchmark": section.get("benchmark"),
            }
    return strategies


def _filter_summary(summary: pd.DataFrame, status: str, min_windows: int, search: str | None, sort_by: str) -> pd.DataFrame:
    data = summary.copy()
    if status == "success":
        data = data[data["wfo_status"] == "success"]
    elif status == "skipped":
        data = data[data["wfo_status"] == "skipped_insufficient_data"]
    if min_windows > 0:
        data = data[(data["wfo_status"] != "success") | (data["wfo_windows"] >= min_windows)]
    if search:
        data = data[data["ticker"].str.contains(search.upper(), case=False, na=False)]
    ascending = sort_by in {"wfo_mdd", "cost_drag_pct"}
    if sort_by in data:
        data = data.sort_values(sort_by, ascending=ascending, na_position="last")
    return data


def _table_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = []
    for row in df.to_dict("records"):
        clean = {}
        for key, value in row.items():
            clean[key] = _round_or_none(value) if isinstance(value, (float, np.floating)) else value
        records.append(clean)
    return records


def _selected_ticker(dropdown_value: str | None, table_data: List[Dict[str, Any]] | None, selected_rows: List[int] | None) -> str | None:
    if selected_rows and table_data:
        idx = selected_rows[0]
        if 0 <= idx < len(table_data):
            return table_data[idx].get("ticker") or dropdown_value
    return dropdown_value


def _metadata_rows(bundle: Dict[str, Any], source_path: Path, strategies: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    meta = bundle.get("meta", {})
    rows = [
        ("Bundle path", str(source_path)),
        ("Timestamp", str(meta.get("timestamp", ""))),
        ("Portfolio value", _fmt_currency(meta.get("portfolio_value"))),
        ("Target vol", _fmt_pct(meta.get("target_vol_annual"))),
        ("OOS fraction", _fmt_pct(meta.get("oos_fraction"))),
        ("Cost config", str(meta.get("cost_config", {}))),
    ]
    for name, item in strategies.items():
        section = item["meta"]
        summary = item["summary"]
        rows.extend(
            [
                (f"{name} rows", str(len(summary))),
                (f"{name} WFO train/test/step", f"{section.get('breakout_wfo_train_window', '')}/{section.get('breakout_wfo_test_window', '')}/{section.get('breakout_wfo_step_window', '')}"),
                (f"{name} WFO workers/trials", f"{section.get('breakout_wfo_workers', '')}/{section.get('breakout_wfo_n_trials', '')}"),
                (f"{name} calibration", f"scalars={section.get('breakout_scalar_source', '')}, dm={section.get('breakout_dm_source', '')}"),
                (f"{name} status counts", str(summary["wfo_status"].value_counts(dropna=False).to_dict())),
                (f"{name} data warnings", str((section.get("data_validation") or {}).get("warnings", []))),
            ]
        )
    return [{"Field": field, "Value": value} for field, value in rows]


def _datatable(id_: str, columns: List[Dict[str, Any]], *, page_size: int = 20, row_selectable: str | bool = False) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=id_,
        columns=columns,
        data=[],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        row_selectable=row_selectable,
        selected_rows=[],
        style_table={"overflowX": "auto", "minWidth": "100%"},
        style_cell={
            "padding": "8px",
            "fontFamily": "Space Grotesk, sans-serif",
            "fontSize": "12px",
            "backgroundColor": "rgba(255, 255, 255, 0.92)",
            "color": "#0f172a",
            "border": "1px solid rgba(15, 23, 42, 0.08)",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "88px",
        },
        style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
        style_data_conditional=[
            {"if": {"filter_query": "{wfo_return} > 0", "column_id": "wfo_return"}, "color": "#15803d", "fontWeight": "600"},
            {"if": {"filter_query": "{wfo_return} < 0", "column_id": "wfo_return"}, "color": "#b91c1c", "fontWeight": "600"},
            {"if": {"filter_query": "{wfo_sharpe} > 0", "column_id": "wfo_sharpe"}, "color": "#15803d", "fontWeight": "600"},
            {"if": {"filter_query": "{wfo_sharpe} < 0", "column_id": "wfo_sharpe"}, "color": "#b91c1c", "fontWeight": "600"},
            {"if": {"filter_query": "{sharpe_delta} < 0", "column_id": "sharpe_delta"}, "backgroundColor": "#fef3c7"},
            {"if": {"filter_query": "{quality_flag} contains 'low sample'"}, "backgroundColor": "#fff7ed"},
            {"if": {"filter_query": "{quality_flag} contains 'data gap'"}, "backgroundColor": "#fef3c7"},
            {"if": {"filter_query": "{quality_flag} contains 'stale'"}, "backgroundColor": "#e0f2fe"},
            {"if": {"filter_query": "{quality_flag} contains 'extreme DD'"}, "backgroundColor": "#fee2e2"},
        ],
    )


def _build_app(bundle: Dict[str, Any], source_path: Path) -> Dash:
    meta = bundle.get("meta", {})
    timestamp = meta.get("timestamp", dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    strategies = _prepare_strategies(bundle)
    if not strategies:
        raise ValueError("No strategy results found in bundle.")

    strategy_order = [name for name in ("breakout", "momentum") if name in strategies]
    default_strategy = "breakout" if "breakout" in strategy_order else strategy_order[0]
    default_summary = strategies[default_strategy]["summary"]
    default_filtered = _filter_summary(default_summary, "success", 3, "", "robust_score")
    default_coin = default_filtered["ticker"].iloc[0] if not default_filtered.empty else default_summary["ticker"].iloc[0]
    default_basket = default_filtered.dropna(subset=["robust_score"]).sort_values("robust_score", ascending=False)["ticker"].head(12).tolist()

    benchmark_options = [{"label": "BTCUSDT", "value": "BTCUSDT"}]
    app = Dash(__name__, title="Breakout WFO Dashboard")

    app.layout = html.Div(
        className="app-shell",
        children=[
            html.Div(
                className="hero",
                children=[
                    html.Div(
                        className="hero-text",
                        children=[
                            html.H1("Breakout WFO Research Dashboard"),
                            html.P(f"Bundle timestamp: {timestamp}"),
                            html.P(str(source_path)),
                        ],
                    ),
                    html.Div(
                        className="hero-meta",
                        children=[
                            html.Div([html.Span("Portfolio", className="meta-label"), html.Span(_fmt_currency(meta.get("portfolio_value")), className="meta-value")], className="meta-chip"),
                            html.Div([html.Span("Target Vol", className="meta-label"), html.Span(_fmt_pct(meta.get("target_vol_annual")), className="meta-value")], className="meta-chip"),
                            html.Div([html.Span("Primary Source", className="meta-label"), html.Span("Walk-forward OOS", className="meta-value")], className="meta-chip"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="controls",
                children=[
                    html.Div([html.Label("Strategy"), dcc.Dropdown(id="strategy-dropdown", options=[{"label": s.title(), "value": s} for s in strategy_order], value=default_strategy, clearable=False)], className="control"),
                    html.Div([html.Label("Metric Source"), dcc.Dropdown(id="metric-source-dropdown", options=[{"label": "Walk-forward OOS", "value": SOURCE_WFO}, {"label": "Static OOS", "value": SOURCE_STATIC_OOS}, {"label": "Full", "value": SOURCE_FULL}], value=SOURCE_WFO, clearable=False)], className="control"),
                    html.Div([html.Label("Status"), dcc.Dropdown(id="status-filter-dropdown", options=[{"label": "WFO success", "value": "success"}, {"label": "Skipped insufficient data", "value": "skipped"}, {"label": "All", "value": "all"}], value="success", clearable=False)], className="control"),
                    html.Div([html.Label("Minimum WFO Windows"), dcc.Slider(id="min-windows-slider", min=0, max=16, step=1, value=3, marks={0: "0", 3: "3", 8: "8", 16: "16"})], className="control"),
                    html.Div([html.Label("Search Ticker"), dcc.Input(id="ticker-search", type="text", placeholder="e.g. DOGE", debounce=True, className="text-input")], className="control"),
                    html.Div([html.Label("Sort"), dcc.Dropdown(id="sort-dropdown", options=SORT_OPTIONS, value="robust_score", clearable=False)], className="control"),
                    html.Div([html.Label("Benchmark"), dcc.Dropdown(id="benchmark-dropdown", options=benchmark_options, value="BTCUSDT", clearable=False)], className="control"),
                    html.Div([html.Label("Selected Ticker"), dcc.Dropdown(id="coin-dropdown", options=[{"label": default_coin, "value": default_coin}], value=default_coin, clearable=False)], className="control"),
                ],
            ),
            dcc.Tabs(
                id="main-tabs",
                value="overview",
                children=[
                    dcc.Tab(
                        label="Overview",
                        value="overview",
                        children=[
                            html.Div(id="overview-kpis", className="section"),
                            html.Div(
                                className="chart-grid",
                                children=[
                                    html.Div(dcc.Graph(id="wfo-sharpe-hist", config={"displayModeBar": False}), className="card"),
                                    html.Div(dcc.Graph(id="wfo-return-hist", config={"displayModeBar": False}), className="card"),
                                    html.Div(dcc.Graph(id="return-dd-scatter", config={"displayModeBar": False}), className="card"),
                                    html.Div(dcc.Graph(id="sharpe-comparison-scatter", config={"displayModeBar": False}), className="card"),
                                    html.Div(dcc.Graph(id="robust-score-top", config={"displayModeBar": False}), className="card"),
                                    html.Div(dcc.Graph(id="wfo-worst-chart", config={"displayModeBar": False}), className="card"),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Ticker Explorer",
                        value="explorer",
                        children=[
                            html.Div(
                                className="section",
                                children=[
                                    html.Div([html.H2("Ticker Explorer"), html.P("Sort, filter, and select a ticker for detailed WFO diagnostics.")], className="section-header"),
                                    _datatable("ticker-table", TABLE_COLUMNS, page_size=25, row_selectable="single"),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Ticker Detail",
                        value="detail",
                        children=[
                            html.Div(
                                className="section",
                                children=[
                                    html.Div([html.H2(id="detail-title"), html.P("Static fitted behavior, walk-forward OOS behavior, sizing, costs, and parameter stability.")], className="section-header"),
                                    html.Div(className="chart-grid", children=[
                                        html.Div(dcc.Graph(id="equity-comparison", config={"displayModeBar": False}), className="card wide-card"),
                                        html.Div(dcc.Graph(id="drawdown-figure", config={"displayModeBar": False}), className="card"),
                                        html.Div(dcc.Graph(id="return-distribution", config={"displayModeBar": False}), className="card"),
                                        html.Div(dcc.Graph(id="detail-figure", config={"displayModeBar": False}), className="card wide-card"),
                                        html.Div(dcc.Graph(id="rolling-performance", config={"displayModeBar": False}), className="card"),
                                        html.Div(dcc.Graph(id="cost-turnover", config={"displayModeBar": False}), className="card"),
                                        html.Div(dcc.Graph(id="weight-stability", config={"displayModeBar": False}), className="card wide-card"),
                                    ]),
                                    html.Div(className="stats-panel", children=[html.H3("Performance Stats"), _datatable("stats-table", [{"name": c, "id": c} for c in ["Segment", "Sharpe", "Deflated Sharpe", "CAGR", "Max Drawdown", "Total Return", "Skew", "Lower Tail (5%)", "Upper Tail (95%)", "T-Stat", "Information Ratio"]], page_size=8)]),
                                    html.Div(className="stats-panel", children=[html.H3("WFO Window Diagnostics"), _datatable("wfo-window-table", [{"name": c, "id": c} for c in ["Refit date", "Test start", "Test end", "Return", "Sharpe", "Max DD", "Dominant horizon", "Turnover"]], page_size=20)]),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Basket Builder",
                        value="basket",
                        children=[
                            html.Div(
                                className="section",
                                children=[
                                    html.Div([html.H2("Basket Builder"), html.P("Research-only equal-weight WFO basket from selected tickers.")], className="section-header"),
                                    html.Div([html.Label("Basket Tickers"), dcc.Dropdown(id="basket-dropdown", multi=True, options=[{"label": t, "value": t} for t in default_basket], value=default_basket)], className="control full-width"),
                                    html.Div(id="basket-kpis", className="kpi-grid"),
                                    html.Div(className="chart-grid", children=[
                                        html.Div(dcc.Graph(id="basket-equity", config={"displayModeBar": False}), className="card"),
                                        html.Div(dcc.Graph(id="basket-drawdown", config={"displayModeBar": False}), className="card"),
                                        html.Div(dcc.Graph(id="basket-rolling", config={"displayModeBar": False}), className="card"),
                                        html.Div(dcc.Graph(id="basket-cost", config={"displayModeBar": False}), className="card"),
                                        html.Div(dcc.Graph(id="basket-contrib", config={"displayModeBar": False}), className="card wide-card"),
                                    ]),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Run Metadata",
                        value="metadata",
                        children=[
                            html.Div(
                                className="section",
                                children=[
                                    html.Div([html.H2("Run Metadata"), html.P("Bundle configuration and data validation context.")], className="section-header"),
                                    _datatable("metadata-table", [{"name": "Field", "id": "Field"}, {"name": "Value", "id": "Value"}], page_size=50),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

    def _strategy_items(strategy: str) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], str]:
        item = strategies[strategy]
        meta_local = item["meta"]
        return item["summary"], item["results"], meta_local, meta_local.get("data_frequency", "1d")

    @app.callback(
        [
            Output("coin-dropdown", "options"),
            Output("coin-dropdown", "value"),
            Output("basket-dropdown", "options"),
            Output("basket-dropdown", "value"),
        ],
        [
            Input("strategy-dropdown", "value"),
            Input("status-filter-dropdown", "value"),
            Input("min-windows-slider", "value"),
            Input("ticker-search", "value"),
            Input("sort-dropdown", "value"),
        ],
    )
    def _update_options(strategy: str, status: str, min_windows: int, search: str | None, sort_by: str):
        summary, _, _, _ = _strategy_items(strategy)
        filtered = _filter_summary(summary, status, min_windows or 0, search, sort_by)
        if filtered.empty:
            filtered = summary.copy()
        options = [{"label": ticker, "value": ticker} for ticker in filtered["ticker"].tolist()]
        value = options[0]["value"] if options else None
        basket_df = filtered[(filtered["wfo_status"] == "success") & (filtered["wfo_windows"] >= max(3, min_windows or 0))]
        basket = basket_df.dropna(subset=["robust_score"]).sort_values("robust_score", ascending=False)["ticker"].head(12).tolist()
        basket_options = [{"label": ticker, "value": ticker} for ticker in filtered["ticker"].tolist()]
        return options, value, basket_options, basket

    @app.callback(
        [
            Output("overview-kpis", "children"),
            Output("wfo-sharpe-hist", "figure"),
            Output("wfo-return-hist", "figure"),
            Output("return-dd-scatter", "figure"),
            Output("sharpe-comparison-scatter", "figure"),
            Output("robust-score-top", "figure"),
            Output("wfo-worst-chart", "figure"),
            Output("ticker-table", "data"),
            Output("metadata-table", "data"),
        ],
        [
            Input("strategy-dropdown", "value"),
            Input("status-filter-dropdown", "value"),
            Input("min-windows-slider", "value"),
            Input("ticker-search", "value"),
            Input("sort-dropdown", "value"),
        ],
    )
    def _update_overview(strategy: str, status: str, min_windows: int, search: str | None, sort_by: str):
        summary, results, meta_local, freq = _strategy_items(strategy)
        filtered = _filter_summary(summary, status, min_windows or 0, search, sort_by)
        portfolio = build_equal_weight_portfolio(
            filtered[filtered["wfo_status"] == "success"]["ticker"].tolist(),
            SOURCE_WFO,
            results,
            first_oos_ts=meta_local.get("first_oos_ts"),
            portfolio_value=_safe_float(meta.get("portfolio_value")) or 1.0,
            freq=freq,
        )
        return (
            _build_kpis(summary, portfolio),
            _histogram(filtered, "wfo_sharpe", "WFO Sharpe Distribution"),
            _histogram(filtered, "wfo_return", "WFO Total Return Distribution", pct=True),
            _scatter_return_drawdown(filtered),
            _scatter_static_vs_wfo(filtered),
            _bar_figure(filtered, "robust_score", "Top 20 Robust WFO Scores"),
            _bar_figure(filtered, "wfo_sharpe", "Worst 20 WFO Sharpe", ascending=True),
            _table_records(filtered),
            _metadata_rows(bundle, source_path, strategies),
        )

    @app.callback(
        [
            Output("detail-title", "children"),
            Output("detail-figure", "figure"),
            Output("stats-table", "data"),
            Output("equity-comparison", "figure"),
            Output("drawdown-figure", "figure"),
            Output("return-distribution", "figure"),
            Output("rolling-performance", "figure"),
            Output("cost-turnover", "figure"),
            Output("weight-stability", "figure"),
            Output("wfo-window-table", "data"),
        ],
        [
            Input("strategy-dropdown", "value"),
            Input("coin-dropdown", "value"),
            Input("ticker-table", "data"),
            Input("ticker-table", "selected_rows"),
            Input("benchmark-dropdown", "value"),
        ],
    )
    def _update_detail(strategy: str, dropdown_ticker: str | None, table_data: List[Dict[str, Any]], selected_rows: List[int], benchmark: str):
        ticker = _selected_ticker(dropdown_ticker, table_data, selected_rows)
        if not ticker:
            empty = _empty_figure("Select a ticker.")
            return "Ticker Detail", empty, [], empty, empty, empty, empty, empty, empty, []
        _, results, meta_local, freq = _strategy_items(strategy)
        payload = results.get(ticker)
        if not payload:
            empty = _empty_figure("No data available for this ticker.")
            return f"{ticker} Detail", empty, [], empty, empty, empty, empty, empty, empty, []

        first_oos_ts = meta_local.get("first_oos_ts")
        bench_df = strategies[strategy].get("benchmark")
        benchmark_returns = bench_df["returns"] if isinstance(bench_df, pd.DataFrame) and "returns" in bench_df and benchmark else None
        wfo = get_wfo_payload(payload)
        wf_metrics = wfo.get("metrics_oos")
        rows = [
            _format_metrics_row("Full", payload.get("metrics")),
            _format_metrics_row("Static OOS", payload.get("metrics_oos")),
        ]
        if wf_metrics:
            returns = _returns_for_source(payload, SOURCE_WFO, first_oos_ts)
            ir = float("nan")
            if benchmark_returns is not None and not returns.empty:
                aligned = returns.align(benchmark_returns, join="inner")
                ir = _annualized_sharpe(aligned[0] - aligned[1], freq)
            rows.append(_format_metrics_row("Walk-forward OOS", wf_metrics, information_ratio=ir))

        series_for_detail = _series_for_source(payload, SOURCE_WFO, first_oos_ts)
        if series_for_detail.empty:
            series_for_detail = _series_for_source(payload, SOURCE_STATIC_OOS, first_oos_ts)
        weights = wfo.get("weights") if wfo else None
        portfolio_value = _safe_float(meta.get("portfolio_value")) or 1.0
        return (
            f"{ticker} Detail",
            _build_detail_figure(series_for_detail, ticker, strategy, freq),
            rows,
            _build_equity_comparison(payload, ticker, benchmark_returns, first_oos_ts=first_oos_ts, freq=freq),
            _build_drawdown_figure(payload, ticker, first_oos_ts=first_oos_ts, freq=freq),
            _build_return_distribution(payload, ticker, first_oos_ts=first_oos_ts),
            _build_rolling_performance(payload, ticker, freq, first_oos_ts=first_oos_ts),
            _build_cost_turnover(payload, ticker, portfolio_value, first_oos_ts=first_oos_ts, freq=freq),
            _build_weight_stability(weights),
            _wfo_window_rows(payload, freq),
        )

    @app.callback(
        [
            Output("basket-kpis", "children"),
            Output("basket-equity", "figure"),
            Output("basket-drawdown", "figure"),
            Output("basket-rolling", "figure"),
            Output("basket-cost", "figure"),
            Output("basket-contrib", "figure"),
        ],
        [
            Input("strategy-dropdown", "value"),
            Input("metric-source-dropdown", "value"),
            Input("basket-dropdown", "value"),
        ],
    )
    def _update_basket(strategy: str, source: str, selected_tickers: List[str] | None):
        _, results, meta_local, freq = _strategy_items(strategy)
        selected = selected_tickers or []
        basket = build_equal_weight_portfolio(
            selected,
            source,
            results,
            first_oos_ts=meta_local.get("first_oos_ts"),
            portfolio_value=_safe_float(meta.get("portfolio_value")) or 1.0,
            freq=freq,
        )
        metrics = basket["metrics"]
        kpis = [
            ("Tickers", f"{len(selected):,}"),
            ("Total Return", _fmt_pct(metrics.get("total_return"))),
            ("CAGR", _fmt_pct(metrics.get("cagr"))),
            ("Sharpe", _fmt_num(metrics.get("sharpe"))),
            ("Max DD", _fmt_pct(metrics.get("max_drawdown"))),
            ("Volatility", _fmt_pct(metrics.get("volatility"))),
            ("Avg Corr", _fmt_num(basket.get("avg_pairwise_corr"))),
        ]
        kpi_nodes = [
            html.Div([html.Span(label, className="kpi-label"), html.Strong(value, className="kpi-value")], className="kpi-card")
            for label, value in kpis
        ]
        if basket["returns"].empty:
            empty = _empty_figure("Select basket tickers.")
            return kpi_nodes, empty, empty, empty, empty, empty

        eq_fig = _style_figure(go.Figure(go.Scatter(x=basket["equity"].index, y=basket["equity"], name="Basket equity", line=dict(color="#16a34a", width=2.2), connectgaps=False)), "Equal-weight Basket Equity", height=360)
        dd_fig = _style_figure(go.Figure(go.Scatter(x=basket["drawdown"].index, y=basket["drawdown"], name="Drawdown", line=dict(color="#b91c1c", width=2), connectgaps=False)), "Basket Drawdown", height=320).update_yaxes(tickformat=".0%")
        rolling = _rolling_sharpe(basket["returns"], freq, 180 if freq == "1d" else max(20, int(_periods_per_year(freq) * 0.5)))
        rolling = _calendarize_series(rolling, freq)
        roll_fig = _style_figure(go.Figure(go.Scatter(x=rolling.index, y=rolling, name="Rolling Sharpe", line=dict(color="#2563eb", width=2), connectgaps=False)), "Basket Rolling Sharpe", height=320)
        cost_fig = _style_figure(go.Figure(go.Scatter(x=basket["cost_drag"].index, y=basket["cost_drag"], name="Cost drag", line=dict(color="#f97316", width=2), connectgaps=False)), "Basket Cumulative Cost Drag", height=320).update_yaxes(tickformat=".1%")
        contrib = basket["contributions"].sort_values(ascending=False)
        contrib_fig = _style_figure(
            go.Figure(
                go.Bar(
                    x=contrib.values,
                    y=contrib.index,
                    orientation="h",
                    marker=dict(color=["#15803d" if v >= 0 else "#b91c1c" for v in contrib.values]),
                )
            ),
            "Ticker Contribution to Total Return",
            height=420,
        ).update_xaxes(tickformat=".0%").update_yaxes(autorange="reversed")
        return kpi_nodes, eq_fig, dd_fig, roll_fig, cost_fig, contrib_fig

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Plotly Dash dashboard for backtest results.")
    parser.add_argument("--results", type=str, default="")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="Do not open the dashboard in a browser.")
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else None
    if results_path is None or not results_path.exists():
        results_path = _find_latest_bundle(Path("data_store"))
    if results_path is None:
        raise FileNotFoundError("No backtest_results_bundle_*.pkl found in data_store or data_store/wfo_runs.")

    bundle = pd.read_pickle(results_path)
    app = _build_app(bundle, results_path)

    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        webbrowser.open_new_tab(url)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
