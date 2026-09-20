from __future__ import annotations

import hashlib
import json
import sys
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


RESULTS = ROOT / "data_store/crypto_momentum_research/production_like_walkforward_v1"
OUTPUT = ROOT / "reports/crypto_trend_production_like_walkforward_20260920.html"
COMPOSITION_OUTPUT = ROOT / "reports/crypto_trend_production_like_composition_20260920.html"


def chart_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def pct(value):
    return "—" if pd.isna(value) else f"{value:.1%}"


def num(value):
    return "—" if pd.isna(value) else f"{value:.2f}"


def format_table(frame: pd.DataFrame) -> str:
    output = frame.copy()
    for column in output:
        key = str(column).lower()
        if any(token in key for token in ("return", "volatility", "drawdown", "funding", "fee", "slippage", "cvar", "concentration")) or key.endswith("_rate"):
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


def equity_figure(daily: pd.DataFrame) -> go.Figure:
    names = ["time_series_ensemble", "breakout_ensemble", "combined_50_50"]
    subset = daily[daily.strategy.isin(names)].copy()
    pivot = subset.pivot(index="timestamp", columns=["variant", "strategy"], values="net_return").fillna(0)
    growth = (1 + pivot).cumprod().rename_axis("timestamp")
    long = growth.stack(["variant", "strategy"], future_stack=True).rename("growth").reset_index()
    long["series"] = long.variant + " · " + long.strategy
    fig = px.line(long, x="timestamp", y="growth", color="series")
    style_plotly(fig, "Stitched outer-fold growth of $1", 680)
    fig.update_yaxes(title="Growth of $1")
    return fig


def fold_figure(folds: pd.DataFrame) -> go.Figure:
    subset = folds[folds.strategy.isin(["time_series_ensemble", "breakout_ensemble", "combined_50_50"])].copy()
    subset["series"] = subset.variant + " · " + subset.strategy
    fig = px.bar(subset, x="fold", y="net_sharpe", color="series", barmode="group")
    style_plotly(fig, "Net Sharpe by untouched outer fold", 620)
    fig.update_yaxes(title="Net Sharpe")
    return fig


def rolling_figure(rolling: pd.DataFrame) -> go.Figure:
    subset = rolling[rolling.strategy.eq("combined_50_50")].dropna(subset=["sharpe"]).copy()
    subset["series"] = subset.variant + " · " + subset.window_days.astype(str) + "d"
    fig = px.line(subset, x="timestamp", y="sharpe", color="series")
    style_plotly(fig, "Combined portfolio rolling Sharpe", 620)
    fig.add_hline(y=0, line_width=1, line_color="#777")
    return fig


def selection_churn(selection: pd.DataFrame) -> pd.DataFrame:
    chosen = selection[selection.selected].sort_values(["variant", "family", "fold", "rank"])
    rows = []
    for (variant, family), group in chosen.groupby(["variant", "family"]):
        previous = set()
        for fold, block in group.groupby("fold", sort=False):
            current = set(block.config_id)
            rows.append({"variant": variant, "family": family, "fold": fold,
                         "retained_from_prior": len(current & previous) if previous else np.nan,
                         "new_models": len(current - previous) if previous else 3})
            previous = current
    return pd.DataFrame(rows)


def strategy_dashboard(state: pd.DataFrame, rolling: pd.DataFrame, variant: str, strategy: str) -> go.Figure:
    subset = state[(state.variant == variant) & (state.strategy == strategy)].sort_values("timestamp")
    roll = rolling[(rolling.variant == variant) & (rolling.strategy == strategy)]
    equity = (1 + subset.net_return.fillna(0)).cumprod()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08,
                        subplot_titles=("Stitched outer-fold equity", "Gross and net exposure", "Rolling Sharpe"))
    fig.add_trace(go.Scatter(x=subset.timestamp, y=equity, name="Growth of $1", line=dict(color="#7658c9")), row=1, col=1)
    fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.gross_exposure, name="Gross", line=dict(color="#4c94df")), row=2, col=1)
    fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.net_exposure, name="Net", line=dict(color="#268a87")), row=2, col=1)
    for window, color in ((90, "#ce9a48"), (365, "#66717e")):
        block = roll[roll.window_days.eq(window)]
        fig.add_trace(go.Scatter(x=block.timestamp, y=block.sharpe, name=f"{window}d Sharpe", line=dict(color=color)), row=3, col=1)
    style_plotly(fig, f"{variant} · {strategy}", 900)
    fig.update_yaxes(title="Growth", row=1, col=1)
    fig.update_yaxes(title="Weight", row=2, col=1)
    fig.update_yaxes(title="Sharpe", row=3, col=1)
    return fig


def weight_heatmap(positions: pd.DataFrame, variant: str, strategy: str) -> go.Figure:
    subset = positions[(positions.variant == variant) & (positions.strategy == strategy)].copy()
    pivot = subset.pivot(index="timestamp", columns="symbol", values="held_weight").fillna(0)
    top = pivot.abs().mean().nlargest(15).index.tolist()
    other = [column for column in pivot if column not in top]
    weekly = pivot[top].resample("W").last()
    if other:
        weekly["OTHER_NET"] = pivot[other].sum(axis=1).resample("W").last()
    fig = go.Figure(go.Heatmap(
        z=weekly.T.to_numpy(), x=weekly.index, y=weekly.columns,
        colorscale=[[0, "#bd4e4e"], [.5, "#f5f6f9"], [1, "#268a87"]], zmid=0,
        colorbar=dict(title="Held weight"),
    ))
    style_plotly(fig, f"Signed weekly holdings · {variant} · {strategy}", 650)
    fig.update_yaxes(title="")
    return fig


def main():
    manifest = json.loads((RESULTS / "study_manifest.json").read_text(encoding="utf-8"))
    selection = pd.read_parquet(RESULTS / "rolling_selection_ledger.parquet")
    selected = selection[selection.selected].copy()
    folds = pd.read_parquet(RESULTS / "outer_fold_metrics.parquet")
    stitched = pd.read_parquet(RESULTS / "stitched_oos_metrics.parquet")
    daily = pd.read_parquet(RESULTS / "daily_portfolio_state.parquet")
    rolling = pd.read_parquet(RESULTS / "rolling_metrics.parquet")
    positions = pd.read_parquet(RESULTS / "daily_ticker_positions.parquet")
    for frame in (selection, folds, daily, rolling, positions):
        if "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    combined = stitched[stitched.strategy.eq("combined_50_50")].set_index("variant")
    best_variant = combined.net_sharpe.idxmax()
    variant_labels = {"nested_expanding": "Expanding history", "nested_rolling_730d": "730-day rolling history"}
    best_variant_label = variant_labels.get(best_variant, best_variant)
    worst_fold = folds[folds.strategy.eq("combined_50_50")].sort_values("net_sharpe").iloc[0]
    churn = selection_churn(selection)
    selected_table = selected[["variant", "fold", "family", "rank", "model", "selection_net_sharpe",
                               "target_vol", "gross_cap", "ticker_risk_cap", "rebalance"]]
    stitched_table = stitched[["variant", "strategy", "annual_return", "annual_volatility", "net_sharpe", "sortino",
                               "calmar", "max_drawdown", "annual_turnover", "fees", "slippage", "funding", "btc_beta"]]

    body = hero(
        "ACAUSAL CAPITAL · PRODUCTION-LIKE WALK-FORWARD",
        "Two causal histories, six model slots,",
        "and the portfolios actually held through time.",
        "Annual no-leakage roster decisions with expanding and 730-day rolling fits, next-open execution, event funding and complete position-state reporting.",
        ["Outer folds 2022–2026 YTD", "90/365-day diagnostics", "3 momentum + 3 breakout", "97-asset portfolio universe", "Shadow only"],
    )
    body += '<nav><a href="#conclusion">Conclusion</a><a href="#performance">Performance</a><a href="#selection">Selection</a><a href="#composition">Composition</a><a href="#method">Method</a><a href="#downloads">Downloads</a></nav>'
    body += '<section class="metrics">' + metric("Expanding combined Sharpe", num(combined.loc["nested_expanding", "net_sharpe"]), "Stitched outer folds") + metric("Rolling-730 combined Sharpe", num(combined.loc["nested_rolling_730d", "net_sharpe"]), "Stitched outer folds") + metric("Better history treatment", best_variant_label, "Selected by stitched OOS only for comparison") + metric("Worst combined fold", num(worst_fold.net_sharpe), f"{variant_labels.get(worst_fold.variant, worst_fold.variant)} · {worst_fold.fold}") + '</section>'
    prospective_observations = int(manifest.get("prospective_observations", 0))
    funding_coverage = manifest.get("funding_coverage", {})
    body += f'<aside class="callout warning"><strong>Historical reconstruction, not prospective evidence.</strong> Every outer fold is causal, but the universe remains a current-universe historical fallback. The prospective ledger contains {prospective_observations} timestamped seed snapshot; it has no elapsed forward return period yet.</aside>'
    body += '<section id="conclusion"><h2>Investment conclusion</h2><div class="findings">' + finding(1, "Two history assumptions are visible", f"{best_variant_label} produced the stronger combined stitched Sharpe, while the fold table shows whether that advantage was stable.") + finding(2, "The roster is part of the model", "Three distinct signal-model names are selected per family at every annual cutoff; risk settings are also selected using prior data only.") + finding(3, "Holdings—not averages—drive the portfolio", "Every reported ensemble is reconstructed from timestamped held weights, with offsetting trades, fees, slippage and funding recalculated at portfolio level.") + finding(4, "Forward proof has only started", f"The shadow database contains the locked six-model seed roster and {prospective_observations} timestamped seed snapshot, but no realized forward return period yet.") + '</div></section>'
    body += '<section id="performance"><h2>1. Stitched outer-fold performance</h2>' + figure_html(chart_html(equity_figure(daily)), "Causal stitched equity", "Only next-year outer-fold returns enter these paths.", "../data_store/crypto_momentum_research/production_like_walkforward_v1/stitched_oos_returns.parquet") + figure_html(chart_html(fold_figure(folds)), "Fold-by-fold Sharpe", "A production-like result should survive several folds rather than one 2026 period.", "../data_store/crypto_momentum_research/production_like_walkforward_v1/outer_fold_metrics.parquet") + figure_html(chart_html(rolling_figure(rolling)), "Rolling diagnostics", "The user-selected 90-day and 365-day windows are shown.", "../data_store/crypto_momentum_research/production_like_walkforward_v1/rolling_metrics.parquet") + '<h3>Complete stitched metrics</h3>' + format_table(stitched_table) + '</section>'
    body += '<section id="selection"><h2>2. What would have been selected at each date?</h2><p>Each annual roster uses only returns available through the preceding 31 December. Candidate fitting uses either all prior history or the trailing 730 days. The three model slots then remain locked through the next outer fold while their declared internal refits continue.</p>' + format_table(selected_table) + '<h3>Roster churn</h3>' + format_table(churn) + '</section>'
    body += '<section id="composition"><h2>3. Portfolio composition through time</h2><p>The companion report contains an equity/exposure/rolling-Sharpe dashboard and signed weekly holdings heatmap for every standalone slot, family ensemble and combined portfolio under both history treatments.</p><p><a href="crypto_trend_production_like_composition_20260920.html"><strong>Open all strategy composition analytics →</strong></a></p></section>'
    body += '<section id="method"><h2>4. Methodology and accounting</h2><ol><li>At every year-end cutoff, fit and score the full eligible time-series momentum and corrected Carver5 breakout grids using prior information only.</li><li>Select the highest prior-only net-Sharpe configuration for each distinct model name, then take the top three model names per family with deterministic turnover and risk tie-breaks.</li><li>Activate the roster for the next calendar fold. Internal quarterly, semiannual, annual and frozen schedules retain their original behavior and activate on the next candle.</li><li>Execute at next open with 100% taker commission, five-basis-point one-way slippage and archived event-level funding.</li><li>Build equal-risk family ensembles from unit-risk constituent positions and a 50/50 family combination. Recalculate portfolio-level trades and costs after netting.</li></ol>'
    body += f'<h3>Funding coverage and futures eligibility</h3><p>The funding ledger contains {funding_coverage.get("events", 0):,} events, of which {funding_coverage.get("finite_mark_prices", 0):,} have a usable mark price. Missing historical marks were backfilled from Binance USD-M eight-hour mark-price candles at the funding boundary. The remaining {funding_coverage.get("unresolved_pre_eligibility", 0):,} events all precede the first available mark-price history for their contracts; no unresolved event remains after eligibility. A ticker can warm its forecast on earlier spot history, but cannot hold a simulated futures position until the day after its first fully priced funding event.</p>'
    body += '<h3>Limitations</h3><ul><li>Current-universe history remains survivor-biased.</li><li>The broad candidate matrix carries multiple-testing and winner’s-curse risk.</li><li>The 730-day window is a locked implementation default rather than an independently validated optimum.</li><li>Historical fills use fixed slippage rather than order-book replay.</li><li>The prospective shadow ledger has no elapsed forward sample.</li></ul></section>'
    body += '<section id="downloads"><h2>Downloads and reproduction</h2><p><a href="../data_store/crypto_momentum_research/production_like_walkforward_v1/all_tested_configurations.json">Every tested configuration (JSON)</a> · <a href="../data_store/crypto_momentum_research/production_like_walkforward_v1/selected_deployable_configurations.json">Selected exact configurations (JSON)</a> · <a href="../data_store/crypto_momentum_research/production_like_walkforward_v1/rolling_selection_ledger.parquet">Selection ledger</a> · <a href="../data_store/crypto_momentum_research/production_like_walkforward_v1/outer_fold_metrics.parquet">Fold metrics</a> · <a href="../data_store/crypto_momentum_research/production_like_walkforward_v1/daily_portfolio_state.parquet">Daily state</a> · <a href="../data_store/crypto_momentum_research/production_like_walkforward_v1/daily_ticker_positions.parquet">Ticker positions</a> · <a href="../data_store/crypto_momentum_research/production_like_walkforward_v1/study_manifest.json">Manifest</a></p><pre>python scripts/execute_production_like_walkforward.py&#10;python scripts/build_production_like_walkforward_report.py</pre></section>'
    OUTPUT.write_text(render_page("Crypto Trend Production-Like Walk-Forward", body, plotly=True, accent="purple"), encoding="utf-8")

    composition_body = hero(
        "ACAUSAL CAPITAL · PORTFOLIO COMPOSITION",
        "Every selected strategy,",
        "position by position through time.",
        "Signed holdings, changing exposure, concentration and rolling performance for all six model slots, both family ensembles and the combined portfolio.",
        ["Expanding + rolling-730", "90/365-day Sharpe", "Top-15 ticker heatmaps", "Net costs and funding", "Outer folds only"],
    )
    composition_body += '<nav><a href="#directory">Strategy directory</a><a href="#interpretation">Interpretation</a></nav>'
    strategy_order = ["time_series_slot_1", "time_series_slot_2", "time_series_slot_3", "breakout_slot_1", "breakout_slot_2", "breakout_slot_3", "time_series_ensemble", "breakout_ensemble", "combined_50_50"]
    composition_body += '<section id="directory"><h2>Strategy directory</h2><p>Slots can change model identity at annual boundaries. Use the roster table in the main report to identify the active configuration in each fold.</p>'
    for variant in [item["id"] for item in manifest["variants"]]:
        composition_body += f'<h2>{variant}</h2>'
        for strategy in strategy_order:
            section_id = f'{variant}-{strategy}'.replace('_', '-')
            summary = stitched[(stitched.variant == variant) & (stitched.strategy == strategy)].iloc[0]
            composition_body += f'<section id="{section_id}"><h3>{strategy}</h3><div class="metrics">' + metric("Net Sharpe", num(summary.net_sharpe), "Stitched OOS") + metric("Annual return", pct(summary.annual_return), "Arithmetic annualized") + metric("Maximum drawdown", pct(summary.max_drawdown), "Stitched OOS") + metric("Funding", pct(summary.funding), "Cumulative contribution") + '</div>'
            composition_body += figure_html(chart_html(strategy_dashboard(daily, rolling, variant, strategy)), "Performance and exposure", "Roster changes occur at annual boundaries.")
            composition_body += figure_html(chart_html(weight_heatmap(positions, variant, strategy)), "Signed holdings heatmap", "Weekly last observation; top 15 tickers by mean absolute held weight plus net Other.", "../data_store/crypto_momentum_research/production_like_walkforward_v1/daily_ticker_positions.parquet") + '</section>'
    composition_body += '</section><section id="interpretation"><h2>Interpretation</h2><ul><li>Green cells are long exposure and red cells are short exposure.</li><li>A blank or near-white row means no meaningful held weight, not missing P&amp;L.</li><li>Slot heatmaps can change abruptly when an annual roster selects a different model or risk configuration.</li><li>Family and combined heatmaps reflect positions after cross-model netting.</li></ul></section>'
    COMPOSITION_OUTPUT.write_text(render_page("Crypto Trend Portfolio Composition", composition_body, plotly=True, accent="blue"), encoding="utf-8")

    source_names = ["study_manifest.json", "rolling_selection_ledger.parquet", "outer_fold_metrics.parquet",
                    "stitched_oos_metrics.parquet", "daily_portfolio_state.parquet", "rolling_metrics.parquet",
                    "daily_ticker_positions.parquet", "all_tested_configurations.json",
                    "selected_deployable_configurations.json"]
    report_manifest = {
        "reports": [str(OUTPUT), str(COMPOSITION_OUTPUT)],
        "generator": "scripts/build_production_like_walkforward_report.py",
        "rebuild": "python scripts/build_production_like_walkforward_report.py",
        "inputs": {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in source_names},
    }
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(COMPOSITION_OUTPUT)


if __name__ == "__main__":
    main()
