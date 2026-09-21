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

RESULTS = ROOT / "data_store/crypto_momentum_research/complete_results_carver5_v2"
V1 = ROOT / "data_store/crypto_momentum_research/complete_results"
OUTPUT = ROOT / "reports/crypto_momentum_complete_study_carver5_v2_20260920.html"
TICKER_OUTPUT = ROOT / "reports/crypto_breakout_ticker_analytics_carver5_v2_20260920.html"


def chart_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def pct(value):
    return "—" if pd.isna(value) else f"{value:.1%}"


def num(value):
    return "—" if pd.isna(value) else f"{value:.2f}"


def format_table(frame: pd.DataFrame) -> str:
    output = frame.copy()
    for column in output:
        if any(token in column for token in ("return", "volatility", "drawdown", "funding", "target_vol", "ticker_risk")):
            output[column] = output[column].map(pct)
        elif "sharpe" in column:
            output[column] = output[column].map(num)
        elif "turnover" in column:
            output[column] = output[column].map(lambda value: "—" if pd.isna(value) else f"{value:,.1f}×")
        elif column == "gross_cap":
            output[column] = output[column].map(lambda value: "—" if pd.isna(value) else f"{value:.1f}×")
        elif pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda value: "—" if pd.isna(value) else f"{value:,.2f}")
    return '<div class="table-wrap" tabindex="0">' + output.to_html(index=False, border=0, escape=True) + "</div>"


def unpack_selected(path: Path) -> pd.DataFrame:
    rows = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        rows.append({**item["config"],
                     **{f"validation_{key}": value for key, value in item["metrics"]["validation"].items()},
                     **{f"holdout_{key}": value for key, value in item["metrics"]["holdout_2026"].items()}})
    return pd.DataFrame(rows)


def period_metrics(series: pd.Series) -> dict:
    values = series.dropna()
    equity = (1 + values).cumprod()
    volatility = values.std() * np.sqrt(365)
    annual_return = values.mean() * 365
    return {"annual_return": annual_return, "annual_volatility": volatility,
            "net_sharpe": annual_return / volatility if volatility > 0 else np.nan,
            "max_drawdown": (equity / equity.cummax() - 1).min() if len(equity) else np.nan}


def main():
    configs = pd.read_parquet(RESULTS / "tested_configurations.parquet")
    selected = unpack_selected(RESULTS / "selected_configurations.json")
    manifest = json.loads((RESULTS / "study_manifest.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(RESULTS / "default_daily_returns.parquet")
    daily_funding = pd.read_parquet(RESULTS / "default_daily_funding.parquet")
    eligibility = pd.read_parquet(RESULTS / "breakout_eligibility.parquet")
    fdm = pd.read_parquet(RESULTS / "primary_fdm_history.parquet")
    attribution = pd.read_parquet(RESULTS / "ticker_attribution.parquet")
    signals = pd.read_parquet(RESULTS / "daily_signal_records.parquet")
    components = pd.read_parquet(RESULTS / "carver_scaled_component_forecasts.parquet")
    capacity = pd.read_csv(RESULTS / "capacity_analysis.csv")
    groups = pd.read_parquet(RESULTS / "group_attribution.parquet")
    v1_selected = unpack_selected(V1 / "selected_configurations.json")
    v1_daily = pd.read_parquet(V1 / "default_daily_returns.parquet")
    headline = configs[configs.activation.eq("next_open") & configs.taker_share.eq(1) & configs.slippage_bps.eq(5)]
    common = headline[headline.target_vol.eq(.15) & headline.gross_cap.eq(1) & headline.ticker_risk_cap.eq(.10) & headline.rebalance.eq("daily")].drop_duplicates("model")
    best = headline.sort_values(["validation_net_sharpe", "validation_annual_turnover"], ascending=[False, True]).drop_duplicates("model")
    bo = selected[selected.family.eq("breakout")].iloc[0]
    combo = selected[selected.family.eq("divergent_combination")].iloc[0]
    old = v1_selected[v1_selected.family.eq("breakout")].iloc[0]

    selected_long = selected.melt(id_vars=["family", "model"], value_vars=["validation_net_sharpe", "holdout_net_sharpe"], var_name="period", value_name="net_sharpe")
    selected_long["period"] = selected_long.period.map({"validation_net_sharpe": "Validation 2024–25", "holdout_net_sharpe": "Historical holdout 2026"})
    fig_selected = px.bar(selected_long, x="family", y="net_sharpe", color="period", barmode="group", hover_data=["model"])
    style_plotly(fig_selected, "Validation-selected corrected strategies", 540)
    fig_selected.update_xaxes(title=""); fig_selected.update_yaxes(title="Net Sharpe")

    refs = ["breakout_carver5_equal_causal_fdm_vol90", "breakout_carver5_equal_fixed_fdm_vol90",
            "breakout_crypto_equal_quarterly_vol90", "breakout_crypto_shrink_80_primary_quarterly_vol90",
            "divergent_forecast_blend", "divergent_sleeve_blend", "divergent_meta_blend"]
    reference = common[common.model.isin(refs)].copy()
    reference["label"] = reference.model.str.replace("breakout_", "", regex=False).str.replace("divergent_", "", regex=False)
    fig_reference = px.bar(reference.sort_values("validation_net_sharpe"), x="validation_net_sharpe", y="label", orientation="h", color="holdout_2026_net_sharpe", color_continuous_scale="Viridis")
    style_plotly(fig_reference, "Corrected combined models at common risk", 620)
    fig_reference.update_xaxes(title="Validation net Sharpe"); fig_reference.update_yaxes(title="")

    controls = common[common.family.eq("breakout_control")].copy()
    controls["horizon"] = controls.model.str.extract(r"h(\d+)").astype(float)
    control_long = controls.melt(id_vars="horizon", value_vars=["validation_net_sharpe", "holdout_2026_net_sharpe"], var_name="period", value_name="net_sharpe")
    control_long["period"] = control_long.period.map({"validation_net_sharpe": "Validation", "holdout_2026_net_sharpe": "Holdout"})
    fig_controls = px.line(control_long, x="horizon", y="net_sharpe", color="period", markers=True, log_x=True)
    style_plotly(fig_controls, "Smoothed single-horizon controls", 540)
    fig_controls.update_xaxes(tickvals=[10, 20, 40, 80, 160], title="Horizon (days)")

    paths = daily.loc[daily.index >= "2024-01-01", [name for name in refs if name in daily]].copy()
    paths["v1_ts_equal_vol60"] = v1_daily["ts_equal_vol60"].reindex(paths.index)
    paths["v1_xs_primary"] = v1_daily["xs_ic5_20_primary_dollar_neutral"].reindex(paths.index)
    growth = (1 + paths.fillna(0)).cumprod().rename_axis("date").reset_index().melt("date", var_name="model", value_name="growth")
    fig_growth = px.line(growth, x="date", y="growth", color="model")
    style_plotly(fig_growth, "Common-risk growth of $1", 680)

    costs = configs[configs.model.eq(bo.model) & configs.target_vol.eq(.15) & configs.gross_cap.eq(1) & configs.ticker_risk_cap.eq(.10) & configs.rebalance.eq("daily") & configs.activation.eq("next_open")].drop_duplicates(["taker_share", "slippage_bps"])
    fig_cost = px.line(costs.sort_values("slippage_bps"), x="slippage_bps", y="validation_net_sharpe", color="taker_share", markers=True)
    style_plotly(fig_cost, "Cost and speed-filter sensitivity", 560)

    eligible = eligibility[eligibility.taker_share.eq(1) & eligibility.slippage_bps.eq(5) & eligibility.risk_window.eq(90) & eligibility.refit_schedule.eq("quarterly")]
    coverage = eligible.groupby("horizon")["eligible"].mean().reset_index(name="eligible_share")
    fig_eligibility = px.bar(coverage, x="horizon", y="eligible_share")
    style_plotly(fig_eligibility, "Cost-eligible ticker/refit observations", 520)
    fig_eligibility.update_yaxes(tickformat=".0%")

    fdm_summary = pd.DataFrame({"date": fdm.index, "median": fdm.median(axis=1), "p10": fdm.quantile(.10, axis=1), "p90": fdm.quantile(.90, axis=1)})
    fig_fdm = px.line(fdm_summary.melt("date", var_name="statistic", value_name="fdm"), x="date", y="fdm", color="statistic")
    style_plotly(fig_fdm, "Causal forecast diversification multiplier", 550)

    risk = headline[headline.model.eq(bo.model) & headline.target_vol.eq(.20) & headline.rebalance.eq("daily")].sort_values("validation_net_sharpe", ascending=False).drop_duplicates(["gross_cap", "ticker_risk_cap"])
    pivot = risk.pivot(index="ticker_risk_cap", columns="gross_cap", values="validation_net_sharpe").sort_index()
    pivot.index = [f"{value:.0%}" for value in pivot.index]; pivot.columns = [f"{value:.1f}×" for value in pivot.columns]
    fig_risk = px.imshow(pivot, text_auto=".2f", color_continuous_scale="Viridis", aspect="auto")
    style_plotly(fig_risk, "Selected breakout risk-grid sensitivity", 600)

    common_table = common[common.model.isin(refs + controls.model.tolist())][["family", "model", "validation_net_sharpe", "holdout_2026_net_sharpe", "validation_annual_return", "holdout_2026_annual_return", "validation_max_drawdown", "holdout_2026_max_drawdown", "validation_annual_turnover", "validation_funding_return"]].sort_values("validation_net_sharpe", ascending=False)
    appendix = best[["family", "model", "method", "refit", "volatility_window", "validation_net_sharpe", "holdout_2026_net_sharpe", "validation_annual_turnover"]].sort_values("validation_net_sharpe", ascending=False)
    comparison = pd.DataFrame([
        {"version": "Invalid v1 unsmoothed winner", "model": old.model, "validation_net_sharpe": old.validation_net_sharpe, "holdout_net_sharpe": old.holdout_net_sharpe, "validation_turnover": old.validation_annual_turnover},
        {"version": "Corrected v2 combined breakout", "model": bo.model, "validation_net_sharpe": bo.validation_net_sharpe, "holdout_net_sharpe": bo.holdout_net_sharpe, "validation_turnover": bo.validation_annual_turnover},
    ])
    annual_rows = []
    for year, block in paths.groupby(paths.index.year):
        for model in paths:
            annual_rows.append({"year": int(year), "model": model, **period_metrics(block[model])})
    annual_table = pd.DataFrame(annual_rows)
    capacity_table = capacity[capacity.model.eq(bo.model)].copy()
    capacity_table["participation"] = capacity_table.participation.map(pct)
    sector_table = groups[(groups.model.eq(bo.model)) & groups.grouping.eq("sector")].sort_values("net_return_contribution", ascending=False)
    funding_table = pd.DataFrame([
        {"model": model, "validation_funding_return": daily_funding.loc[(daily_funding.index >= "2024-01-01") & (daily_funding.index < "2026-01-01"), model].sum(),
         "holdout_funding_return": daily_funding.loc[daily_funding.index >= "2026-01-01", model].sum()}
        for model in refs if model in daily_funding
    ])

    body = hero("ACAUSAL CAPITAL · BREAKOUT RESEARCH V2", "Carver-style breakout, rebuilt causally", "and tested as a complete portfolio.", "Five smoothed channel horizons with Carver scalars, causal cost-speed eligibility, expanding diversification, crypto-adapted alternatives and breakout–EWMAC blends.", ["Data cutoff 18 Sep 2026", "10/20/40/80/160-day rules", "320-day rule excluded", f"{manifest['configuration_count']:,} configurations", "Shadow research only"])
    body += '<nav><a href="#conclusion">Conclusion</a><a href="#method">Method</a><a href="#results">Results</a><a href="#robustness">Robustness</a><a href="#appendix">Appendix</a></nav>'
    body += '<section class="metrics">' + metric("Corrected breakout validation Sharpe", num(bo.validation_net_sharpe), bo.model) + metric("Corrected breakout holdout Sharpe", num(bo.holdout_net_sharpe), "Historical 2026") + metric("Divergent blend holdout Sharpe", num(combo.holdout_net_sharpe), combo.model) + metric("Eligible observations", pct(eligible.eligible.mean()), "Ticker × horizon × refit") + '</section>'
    body += '<aside class="callout warning"><strong>Historical holdout, not prospective OOS.</strong> The current-universe fallback remains survivor-biased, and the broad matrix creates multiple-testing risk.</aside>'
    body += '<section id="conclusion"><h2>Investment conclusion</h2><div class="findings">' + finding(1, "The old result is retired", "The v1 winner was an unsmoothed 32-day control and cannot enter v2 selection.") + finding(2, "The headline is now a true blend", f"{bo.model} is selected only from combined five-horizon strategies.") + finding(3, "Cost eligibility is signal state", "Every ticker’s horizon set is determined causally from volatility and scenario execution cost.") + finding(4, "Combinations remain separate", f"{combo.model} wins only inside the divergent-combination family.") + '</div><h3>Before and after</h3>' + format_table(comparison) + figure_html(chart_html(fig_selected), "Corrected selected families", "Holdout results were not used for selection.", "../data_store/crypto_momentum_research/complete_results_carver5_v2/selected_configurations.json") + '</section>'
    body += '<section id="method"><h2>1. Signal and portfolio methodology</h2><div class="formula">raw(h) = 40 × [price − midpoint(h)] / range(h)<br>component(h) = clip(scalar(h) × EWMA(raw(h), span=h/4), −20, +20)<br>combined = clip(FDM × Σ eligible_weight(h) × component(h), −20, +20)</div><p>The headline speed filter applies Carver’s 0.15-SR annual cost budget using Binance commission, scenario slippage and trailing annualized volatility. Eligibility and fitted parameters activate on the next candle.</p>' + figure_html(chart_html(fig_eligibility), "Speed-filter coverage", "Faster rules are expected to fail more often because their assumed turnover is higher.", "../data_store/crypto_momentum_research/complete_results_carver5_v2/breakout_eligibility.parquet") + figure_html(chart_html(fig_fdm), "FDM path", "Expanding pooled forecast correlations only.", "../data_store/crypto_momentum_research/complete_results_carver5_v2/primary_fdm_history.parquet") + '</section>'
    body += '<section id="results"><h2>2. Portfolio results</h2>' + figure_html(chart_html(fig_reference), "Combined-model comparison", "Common 15% / 1× / 10% risk.", "../data_store/crypto_momentum_research/complete_results_carver5_v2/tested_configurations.parquet") + figure_html(chart_html(fig_growth), "Growth paths", "Net of commissions, slippage and funding.", "../data_store/crypto_momentum_research/complete_results_carver5_v2/default_daily_returns.parquet") + '<h3>Common-risk statistics</h3>' + format_table(common_table) + '<h3>Calendar-year decomposition</h3>' + format_table(annual_table) + '</section>'
    body += '<section id="robustness"><h2>3. Controls and robustness</h2>' + figure_html(chart_html(fig_controls), "Single-horizon controls", "Correctly smoothed controls cannot win selection.") + figure_html(chart_html(fig_risk), "Risk grid", "Broad plateaus are preferable to isolated maxima.") + figure_html(chart_html(fig_cost), "Execution sensitivity", "Eligibility is recomputed in every scenario.") + '</section>'
    body += '<section><h2>4. Funding, capacity and concentration</h2><p>Returns retain event-level Binance funding and pre-rebalance midnight settlement. Capacity uses target trades divided by the archived trailing median Binance quote volume; it is a screen rather than an order-book impact model.</p><h3>Funding contribution</h3>' + format_table(funding_table) + '<h3>Selected breakout capacity</h3>' + format_table(capacity_table) + '<h3>Sector attribution</h3>' + format_table(sector_table) + '<p>This pipeline is shadow-only and does not modify production signals.</p><p><a href="crypto_breakout_ticker_analytics_carver5_v2_20260920.html"><strong>Open corrected breakout ticker analytics →</strong></a></p></section>'
    body += '<section><h2>5. Limitations</h2><ol><li>Current-universe historical fallback creates survivor bias.</li><li>The holdout is historical rather than prospective.</li><li>The large dependent matrix creates selection risk.</li><li>Fixed slippage is not a nonlinear order-book impact model.</li><li>The existing five-rule EWMAC sleeve is held fixed and still omits 64/256.</li></ol></section>'
    body += '<section id="appendix"><h2>Appendix — Complete model registry</h2>' + format_table(appendix) + '<h3>Downloads and rebuild</h3><p><a href="../data_store/crypto_momentum_research/complete_results_carver5_v2/tested_configurations.json">All configurations</a> · <a href="../data_store/crypto_momentum_research/complete_results_carver5_v2/deployable_configurations.json">Selected JSON</a> · <a href="../data_store/crypto_momentum_research/complete_results_carver5_v2/study_manifest.json">Manifest</a> · <a href="../data_store/crypto_momentum_research/complete_results_carver5_v2/daily_signal_records.parquet">Signals</a></p><pre>python scripts/execute_carver_breakout_v2.py&#10;python scripts/build_carver_breakout_v2_report.py</pre></section>'
    OUTPUT.write_text(render_page("Carver-Style Crypto Breakout Study V2", body, plotly=True, accent="purple"), encoding="utf-8")

    latest = signals[signals.model.eq(bo.model)].sort_values("timestamp").groupby("symbol").tail(1).set_index("symbol")
    component_latest = components.iloc[-1].unstack(level=0)
    ticker = attribution[attribution.model.eq(bo.model)].set_index("symbol").join(latest[["forecast", "target_position", "fdm", "eligible_horizons", "quality_flag"]], how="outer").join(component_latest, how="left").reset_index()
    fig_ticker = px.bar(ticker.sort_values("net_return_contribution").tail(25), x="net_return_contribution", y="symbol", orientation="h", color="funding_return", color_continuous_scale="RdYlGn")
    style_plotly(fig_ticker, "Largest selected-breakout ticker contributions", 720)
    ticker_body = hero("ACAUSAL CAPITAL · TICKER ANALYTICS", "Corrected breakout,", "ticker by ticker.", "Current five-horizon forecasts, cost eligibility, FDM, portfolio contribution, funding and turnover.", [bo.model, f"{len(ticker)} tickers", "Carver5 components", "Shadow research"])
    ticker_body += '<section class="metrics">' + metric("Tickers", len(ticker), "Portfolio universe") + metric("Median FDM", num(ticker.fdm.median()), "Latest") + metric("Positive forecasts", int((ticker.forecast > 0).sum()), "Latest close") + metric("No eligible rules", int(ticker.eligible_horizons.eq("[]").sum()), "Latest filter") + '</section>'
    ticker_body += figure_html(chart_html(fig_ticker), "Ticker contribution", "Portfolio attribution after costs and funding.", "../data_store/crypto_momentum_research/complete_results_carver5_v2/ticker_attribution.parquet") + '<h2>Complete ticker directory</h2>' + format_table(ticker) + '<h2>Interpretation</h2><ul><li>Component columns are smoothed, scaled and capped forecasts.</li><li>Eligible horizons use the headline taker-plus-5-bps scenario.</li><li>Contribution is portfolio attribution, not standalone ticker Sharpe.</li></ul>'
    TICKER_OUTPUT.write_text(render_page("Carver5 Breakout Ticker Analytics", ticker_body, plotly=True, accent="blue"), encoding="utf-8")

    sources = ["tested_configurations.parquet", "selected_configurations.json", "default_daily_returns.parquet", "default_daily_funding.parquet", "breakout_eligibility.parquet", "primary_fdm_history.parquet", "ticker_attribution.parquet", "capacity_analysis.csv", "group_attribution.parquet", "study_manifest.json"]
    report_manifest = {"report": str(OUTPUT), "ticker_report": str(TICKER_OUTPUT), "generator": "scripts/build_carver_breakout_v2_report.py", "rebuild": "python scripts/build_carver_breakout_v2_report.py", "inputs": {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in sources}}
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(TICKER_OUTPUT)


if __name__ == "__main__":
    main()
