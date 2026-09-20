from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.report_theme.report_theme import figure_html, finding, hero, metric, render_page, style_plotly

RESULTS = ROOT / "data_store/crypto_momentum_research/complete_results"
OUTPUT = ROOT / "reports/crypto_momentum_complete_study_20260920.html"


def chart_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def pct(value):
    return "—" if pd.isna(value) else f"{value:.1%}"


def num(value):
    return "—" if pd.isna(value) else f"{value:.2f}"


def format_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted:
        if any(token in column for token in ("return", "volatility", "drawdown", "funding", "ticker_risk", "target_vol")):
            formatted[column] = formatted[column].map(pct)
        elif "sharpe" in column:
            formatted[column] = formatted[column].map(num)
        elif column == "gross_cap":
            formatted[column] = formatted[column].map(lambda value: "—" if pd.isna(value) else f"{value:.1f}×")
    return '<div class="table-wrap" tabindex="0">' + formatted.to_html(index=False, border=0, escape=True) + "</div>"


def main():
    configs = pd.read_parquet(RESULTS / "tested_configurations.parquet")
    selected = json.loads((RESULTS / "selected_configurations.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "study_manifest.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(RESULTS / "default_daily_returns.parquet")
    daily_funding = pd.read_parquet(RESULTS / "default_daily_funding.parquet")
    funding_summary = pd.read_csv(RESULTS / "modeled_funding_summary.csv")
    signals = pd.read_csv(RESULTS / "latest_signal_records.csv")
    groups = pd.read_parquet(RESULTS / "group_attribution.parquet")
    capacity = pd.read_csv(RESULTS / "capacity_analysis.csv")
    actual = pd.read_parquet(RESULTS / "actual_funding_reconciliation.parquet")
    headline = configs[(configs.taker_share.fillna(1).eq(1)) & (configs.slippage_bps.fillna(5).eq(5)) &
                       configs.activation.fillna("next_open").eq("next_open")].copy()
    selected_rows = []
    for row in selected:
        selected_rows.append({**row["config"],
                              **{f"validation_{key}": value for key, value in row["metrics"]["validation"].items()},
                              **{f"holdout_{key}": value for key, value in row["metrics"]["holdout_2026"].items()}})
    selected_frame = pd.DataFrame(selected_rows)
    best = {row.family: row for row in selected_frame.itertuples()}

    reference_models = ["ts_equal_vol90", "ts_shrink_80_primary_quarterly_vol90", "ts_legacy_optuna",
                        "xs_ic5_20_primary_dollar_neutral", "breakout_horizon_32", "breakout_equal_vol90",
                        "breakout_shrink_80_primary_quarterly_vol90", "breakout_legacy_optuna",
                        "diagnostic_ts_xs_equal_risk", "baseline_btc_buy_hold", "baseline_equal_basket",
                        "baseline_fdv_weighted", "baseline_no_skill"]
    reference = headline[(headline.model.isin(reference_models)) & headline.target_vol.fillna(.15).eq(.15) &
                         headline.gross_cap.fillna(1).eq(1) & headline.ticker_risk_cap.fillna(.10).eq(.10) &
                         headline.rebalance.fillna("daily").eq("daily")].drop_duplicates("model")

    family_chart = selected_frame[selected_frame.family.ne("baseline")].melt(
        id_vars=["family", "model"], value_vars=["validation_net_sharpe", "holdout_net_sharpe"],
        var_name="period", value_name="net_sharpe")
    family_chart["period"] = family_chart.period.map({"validation_net_sharpe": "Validation 2024–25", "holdout_net_sharpe": "Holdout 2026"})
    fig_family = px.bar(family_chart, x="family", y="net_sharpe", color="period", barmode="group", hover_data=["model"])
    style_plotly(fig_family, "Validation selection versus disclosed 2026 holdout", 560)
    fig_family.update_xaxes(title=""); fig_family.update_yaxes(title="Net Sharpe")

    equity_names = [name for name in reference_models if name in daily.columns and name != "baseline_no_skill"]
    equity = (1 + daily.loc[daily.index >= "2024-01-01", equity_names].fillna(0)).cumprod()
    equity_long = equity.rename_axis("date").reset_index().melt("date", var_name="model", value_name="growth")
    fig_equity = px.line(equity_long, x="date", y="growth", color="model")
    style_plotly(fig_equity, "Growth of $1 at the common 15% / 1× / 10% reference setting", 650)
    fig_equity.update_yaxes(title="Growth multiple")

    model_best = (headline.sort_values("validation_net_sharpe", ascending=False).drop_duplicates("model")
                  .sort_values(["family", "validation_net_sharpe"], ascending=[True, False]))
    scatter = model_best[model_best.family.isin(["time_series", "cross_sectional", "breakout"])]
    fig_models = px.scatter(scatter, x="validation_net_sharpe", y="holdout_2026_net_sharpe", color="family",
                            hover_name="model", size="validation_annual_volatility")
    style_plotly(fig_models, "Every model: validation Sharpe versus holdout Sharpe", 620)
    fig_models.add_hline(y=0, line_dash="dot", line_color="#888"); fig_models.add_vline(x=0, line_dash="dot", line_color="#888")
    fig_models.update_xaxes(title="Validation net Sharpe"); fig_models.update_yaxes(title="2026 holdout net Sharpe")

    cost_models = ["ts_shrink_80_primary_quarterly_vol90", "xs_ic5_20_primary_dollar_neutral",
                   "breakout_shrink_80_primary_quarterly_vol90"]
    cost = configs[(configs.model.isin(cost_models)) & configs.target_vol.eq(.15) & configs.gross_cap.eq(1) &
                   configs.ticker_risk_cap.eq(.10) & configs.activation.eq("next_open")].drop_duplicates(
                       ["model", "taker_share", "slippage_bps"])
    fig_cost = px.line(cost.sort_values("slippage_bps"), x="slippage_bps", y="validation_net_sharpe", color="model",
                       line_dash="taker_share", markers=True)
    style_plotly(fig_cost, "Execution-cost sensitivity at the common reference risk", 600)
    fig_cost.update_xaxes(title="One-way slippage (bps)"); fig_cost.update_yaxes(title="Validation net Sharpe")

    funding_models = [name for name in cost_models + ["breakout_equal_vol90", "breakout_legacy_optuna"] if name in daily_funding]
    funding_totals = funding_summary[(funding_summary.period.eq("ALL")) & funding_summary.model.isin(funding_models)].groupby("model")[["paid", "received", "net"]].sum().reset_index()
    funding_totals = funding_totals.melt("model", var_name="measure", value_name="funding_return")
    fig_funding = px.bar(funding_totals, x="model", y="funding_return", color="measure", barmode="group")
    style_plotly(fig_funding, "Modeled funding contribution at the common reference setting", 540)
    fig_funding.update_yaxes(tickformat=".1%", title="Cumulative return"); fig_funding.update_xaxes(title="")

    primary_signals = signals[signals.model.eq("ts_shrink_80_primary_quarterly_vol90")].dropna(subset=["forecast"])
    extremes = pd.concat([primary_signals.nlargest(12, "forecast"), primary_signals.nsmallest(12, "forecast")]).sort_values("forecast")
    fig_strength = px.bar(extremes, x="forecast", y="symbol", orientation="h", color="forecast", color_continuous_scale="RdYlGn")
    style_plotly(fig_strength, "Latest individual-ticker time-series momentum strength", 690)
    fig_strength.update_layout(coloraxis_showscale=False); fig_strength.update_xaxes(title="Final forecast (±20 cap)")

    sector = groups[(groups.model.eq("xs_ic5_20_primary_dollar_neutral")) & groups.grouping.eq("sector")].sort_values("net_return_contribution")
    fig_sector = px.bar(sector, x="net_return_contribution", y="group", orientation="h", color="net_return_contribution", color_continuous_scale="RdYlGn")
    style_plotly(fig_sector, "Cross-sectional default contribution by sector", 660)
    fig_sector.update_layout(coloraxis_showscale=False); fig_sector.update_xaxes(tickformat=".1%", title="Net return contribution")

    selected_table = selected_frame[["family", "model", "target_vol", "gross_cap", "ticker_risk_cap", "rebalance",
                                     "validation_net_sharpe", "holdout_net_sharpe", "validation_annual_return",
                                     "holdout_annual_return", "validation_max_drawdown", "holdout_max_drawdown"]]
    reference_table = reference[["family", "model", "validation_net_sharpe", "holdout_2026_net_sharpe",
                                 "validation_annual_return", "holdout_2026_annual_return", "validation_max_drawdown",
                                 "holdout_2026_max_drawdown", "validation_funding_return", "validation_annual_turnover"]]
    appendix_table = model_best[["family", "model", "target_vol", "gross_cap", "ticker_risk_cap", "rebalance",
                                 "validation_net_sharpe", "holdout_2026_net_sharpe", "validation_annual_return",
                                 "holdout_2026_annual_return"]]
    ts, xs, bo = best["time_series"], best["cross_sectional"], best["breakout"]
    account_net = float(actual.income_usd.sum())
    matched_count = int(actual.match_status.str.startswith("direction_agrees").sum())
    primary_capacity = capacity[(capacity.model.eq("xs_ic5_20_primary_dollar_neutral")) & capacity.participation_of_median_daily_quote_volume.eq(.01)]
    capacity_value = float(primary_capacity.fifth_percentile_capacity_usd.iloc[0]) if len(primary_capacity) else np.nan
    ts_funding = reference.loc[reference.model.eq("ts_shrink_80_primary_quarterly_vol90"), "validation_funding_return"].iloc[0]

    body = hero("ACAUSAL CAPITAL · CRYPTO MOMENTUM", "Cross-sectional momentum survived execution repair,",
                "while standalone time-series momentum remained weak.",
                "A causal portfolio study of time-series EWMAC, cross-sectional momentum and breakout with tradable next-open returns, event funding, Binance commissions, slippage and full risk grids.",
                ["Data cutoff 18 Sep 2026", "581-contract fitting panel", "97 eligible portfolio assets",
                 f"{manifest['configuration_count']:,} saved configurations", "Shadow research only"])
    body += '<nav><a href="#verdict">Verdict</a><a href="#methods">Methods</a><a href="#portfolio">Portfolio</a><a href="#costs">Costs</a><a href="#funding">Funding</a><a href="#tickers">Tickers</a><a href="#appendix">Appendix</a></nav>'
    body += '<section class="metrics">' + metric("Cross-sectional holdout Sharpe", num(xs.holdout_net_sharpe), f"Selected: {xs.model}") + metric("Breakout holdout Sharpe", num(bo.holdout_net_sharpe), f"Selected: {bo.model}") + metric("Time-series holdout Sharpe", num(ts.holdout_net_sharpe), f"Selected: {ts.model}") + metric("Authenticated account funding", f"${account_net:,.2f}", f"{len(actual):,} records") + '</section>'
    body += '<aside class="callout warning"><strong>Not prospective OOS evidence.</strong> Configuration selection used 2024–25 only and 2026 is disclosed as a holdout, but historical point-in-time FDV membership was unavailable and the present-day top-100 fallback creates survivor bias. Treat this as a strict historical check, not promotion evidence.</aside>'
    body += '<aside class="callout"><strong>Legacy Optuna is a reference, not an OOS candidate.</strong> The saved momentum and breakout files contain valid optimized weights for 64 of 97 portfolio tickers; 33 use explicit equal-weight fallbacks. Those static weights may include later observations, so they are reported but excluded from validation-selected headline models.</aside>'
    body += '<section id="verdict"><h2>Research verdict</h2><div class="findings">'
    body += finding(1, "Time-series is not production-ready", f"The selected time-series model delivered {ts.validation_net_sharpe:.2f} validation Sharpe and {ts.holdout_net_sharpe:.2f} in 2026. Equal weights beat both the shrunk pooled reference and legacy per-ticker Optuna at common risk.")
    body += finding(2, "Cross-sectional is the only strong candidate", f"The selected {xs.model} produced {xs.validation_net_sharpe:.2f} validation Sharpe and {xs.holdout_net_sharpe:.2f} holdout Sharpe after 100% taker fees, 5 bps slippage and event funding. Survivor bias is still the central threat.")
    body += finding(3, "Breakout matters", f"The selected {bo.model} reached {bo.validation_net_sharpe:.2f} validation and {bo.holdout_net_sharpe:.2f} holdout Sharpe—better than EWMAC, materially below cross-sectional momentum.")
    body += finding(4, "The combination diluted the winner", f"The equal-risk diagnostic combination fell to {best['diagnostic_combination'].holdout_net_sharpe:.2f} holdout Sharpe; it is not a recommended blend.")
    body += finding(5, "Funding drags absolute trend", f"At common risk, pooled time-series momentum paid {ts_funding:.1%} during validation while dollar-neutral cross-sectional momentum received a small positive contribution.")
    body += finding(6, "Multiple testing is substantial", f"All requested scenarios are retained and ranked by raw net Sharpe, but {manifest['configuration_count']:,} configurations are not independent evidence. Freeze a parsimonious model before prospective logging.")
    body += '</div>' + figure_html(chart_html(fig_family), "Selected families across validation and holdout", "Selection used validation only; holdout was disclosed afterward.", "../data_store/crypto_momentum_research/complete_results/selected_configurations.json") + '<h3>Validation-selected headline configurations</h3>' + format_table(selected_table) + '</section>'
    body += '<section id="methods"><h2>What was tested</h2><p><strong>Time-series:</strong> five independently constructed EWMAC rules for every ticker; equal, 75/80/90%-shrunk and legacy per-ticker Optuna weights; 25–30% rule caps; quarterly/semiannual/annual/frozen refits; and 60/90/180/360-day volatility. Pooled calibration uses 581 usable historical contracts including delisted names.</p><p><strong>Cross-sectional:</strong> 1-, 5-, 20-, and equal 5/20-day rank-IC fits; signed, continuous dollar-neutral, 10/20/30% baskets, long-only, BTC hedge, beta constraint, BTC residualization and weekly rank buffers.</p><p><strong>Breakout:</strong> five independently constructed 16/32/64/128/256-day channel rules for every ticker; equal and 75/80/90%-shrunk pooled horizon weights; quarterly/semiannual/annual/frozen refits; 60/90/180/360-day volatility; five single-horizon controls; and a legacy per-ticker Optuna reference reconstructed on its exact 10/20/40/80/160-day horizons. Breakout contributes 56 models and 42,456 configurations.</p><p><strong>Portfolio:</strong> 15–40% volatility targets, 1–3× gross, 5–40% ticker caps, daily/weekly/monthly rebalancing, 0/50/100% taker, 0/2/5/10 bps slippage and next-close sensitivity. Headline results use next-open open-to-open P&amp;L, 100% taker and 5 bps. The time-series/cross-sectional combination is diagnostic only.</p>' + figure_html(chart_html(fig_models), "All standalone model variants", "Each point is that model’s validation-best headline risk setting.", "../data_store/crypto_momentum_research/complete_results/tested_configurations.parquet") + '</section>'
    body += '<section id="portfolio"><h2>Portfolio-level evidence</h2>' + figure_html(chart_html(fig_equity), "Common-risk comparison", "15% target, 1× gross and 10% ticker cap; these are references, not selected maxima.", "../data_store/crypto_momentum_research/complete_results/default_daily_returns.parquet") + '<h3>Common-risk methodology comparison</h3>' + format_table(reference_table) + '</section>'
    body += '<section id="costs"><h2>Costs, turnover and capacity</h2>' + figure_html(chart_html(fig_cost), "Fee and slippage sensitivity", "Line style distinguishes maker/taker mix; headline uses 100% taker and 5 bps.", "../data_store/crypto_momentum_research/complete_results/tested_configurations.json") + f'<p>At one-percent participation in trailing median Binance quote volume, the primary dollar-neutral sleeve’s fifth-percentile daily capacity estimate is <strong>${capacity_value:,.0f}</strong>. This is a screen, not a market-impact model.</p></section>'
    body += '<section id="funding"><h2>Funding accounting</h2>' + figure_html(chart_html(fig_funding), "Historical modeled funding paid, received and net", "Midnight settlements use the pre-rebalance position; ticker/side/year detail is included in the download.", "../data_store/crypto_momentum_research/complete_results/modeled_funding_summary.csv") + f'<p>The public archive contains <strong>433,572</strong> settlement events across 100 requested symbols and no post-listing gap above 24 hours. Authenticated account history contains <strong>{len(actual):,}</strong> funding records totaling <strong>${account_net:,.2f}</strong>; {matched_count:,} have direction agreeing with the default model. Where a public rate is available, actual account exposure notional is reconstructed from cashflow ÷ rate and kept separate from model weights.</p></section>'
    body += '<section id="tickers"><h2>Ticker strength and cohorts</h2>' + figure_html(chart_html(fig_strength), "Current absolute momentum extremes", "Warm-up values remain missing; final forecasts are capped at ±20.", "../data_store/crypto_momentum_research/complete_results/latest_signal_records.csv") + figure_html(chart_html(fig_sector), "Cross-sectional contribution by sector", "Current-universe fallback; archived provider classifications.", "../data_store/crypto_momentum_research/complete_results/group_attribution.parquet") + '</section>'
    body += '<section id="appendix"><h2>Complete model appendix</h2><p>The validation-best headline risk setting for every model appears below. The JSON registry retains every risk, cost and delay scenario.</p>' + format_table(appendix_table) + '<h3>Coverage and reproducibility</h3><ul><li>Current top-100 eligible Binance USDT perpetuals by FDV, $1m median quote-volume floor; 97 had 90 valid closes.</li><li>581-contract historical fitting panel; pre-listing data never earns P&amp;L.</li><li>Forward Binance open-to-open headline returns; the earlier close-to-close approximation is superseded.</li><li>Causal expanding refits, validation-only selection, historical 2026 holdout; no prospective OOS period yet.</li></ul><h3>Downloads</h3><p><a href="../data_store/crypto_momentum_research/complete_results/tested_configurations.json">All configurations JSON</a> · <a href="../data_store/crypto_momentum_research/complete_results/deployable_configurations.json">Deployable parameter JSON</a> · <a href="../data_store/crypto_momentum_research/complete_results/selected_configurations.json">Selected results JSON</a> · <a href="../data_store/crypto_momentum_research/complete_results/study_manifest.json">Run manifest</a> · <a href="../data_store/crypto_momentum_research/complete_results/latest_signal_records.csv">Ticker signals</a> · <a href="../data_store/crypto_momentum_research/complete_results/actual_funding_summary.csv">Actual funding</a></p><h3>Rebuild</h3><pre>python scripts/archive_portfolio_open_prices.py\npython scripts/execute_complete_crypto_study.py\npython scripts/build_full_crypto_study_report.py</pre></section>'

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_page("Complete Crypto Momentum Study", body, plotly=True, accent="purple"), encoding="utf-8")
    sources = ["tested_configurations.parquet", "selected_configurations.json", "default_daily_returns.parquet", "default_daily_funding.parquet", "latest_signal_records.csv", "study_manifest.json"]
    report_manifest = {"report": str(OUTPUT), "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "study_generated_at": manifest["generated_at"], "generator": "scripts/build_full_crypto_study_report.py", "rebuild": "python scripts/build_full_crypto_study_report.py", "inputs": {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in sources}}
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
