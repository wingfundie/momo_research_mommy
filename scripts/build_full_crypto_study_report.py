from __future__ import annotations

import hashlib
import json
import sqlite3
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
        elif "turnover" in column:
            formatted[column] = formatted[column].map(lambda value: "—" if pd.isna(value) else f"{value:,.1f}×")
        elif "capacity" in column or column.endswith("_usd"):
            formatted[column] = formatted[column].map(lambda value: "—" if pd.isna(value) else f"${value:,.0f}")
        elif column == "gross_cap":
            formatted[column] = formatted[column].map(lambda value: "—" if pd.isna(value) else f"{value:.1f}×")
        elif pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "—" if pd.isna(value) else f"{value:,.2f}")
    return '<div class="table-wrap" tabindex="0">' + formatted.to_html(index=False, border=0, escape=True) + "</div>"


def period_metrics(series: pd.Series) -> dict:
    values = series.dropna()
    if values.empty:
        return {"annual_return": np.nan, "annual_volatility": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    equity = (1 + values).cumprod()
    volatility = values.std() * np.sqrt(365)
    annual_return = values.mean() * 365
    return {
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": annual_return / volatility if volatility > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }


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
    with sqlite3.connect(RESULTS / "crypto_momentum_research.sqlite") as connection:
        ledger_config_count = int(connection.execute("SELECT COUNT(*) FROM tested_configurations").fetchone()[0])
        ledger_signal_count = int(connection.execute("SELECT COUNT(*) FROM signal_records").fetchone()[0])
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

    display_names = {
        "xs_ic20_dollar_neutral": "Cross-sectional IC20",
        "xs_ic5_20_primary_dollar_neutral": "Cross-sectional IC5/20",
        "breakout_horizon_32": "Breakout 32d",
        "breakout_equal_vol90": "Breakout equal",
        "breakout_shrink_80_primary_quarterly_vol90": "Breakout pooled",
        "ts_equal_vol90": "EWMAC equal",
        "ts_shrink_80_primary_quarterly_vol90": "EWMAC pooled",
        "diagnostic_ts_xs_equal_risk": "EWMAC + cross-sectional",
        "baseline_btc_buy_hold": "BTC buy-and-hold",
        "baseline_equal_basket": "Equal-weight basket",
        "baseline_fdv_weighted": "FDV-weighted basket",
    }
    path_models = [name for name in display_names if name in daily]
    path_returns = daily.loc[daily.index >= "2024-01-01", path_models].rename(columns=display_names).fillna(0)
    rolling_sharpe = path_returns.rolling(180, min_periods=120).mean().div(
        path_returns.rolling(180, min_periods=120).std().replace(0, np.nan)
    ) * np.sqrt(365)
    rolling_long = rolling_sharpe.rename_axis("date").reset_index().melt("date", var_name="strategy", value_name="rolling_sharpe")
    fig_rolling = px.line(rolling_long, x="date", y="rolling_sharpe", color="strategy")
    style_plotly(fig_rolling, "Rolling 180-day Sharpe at the common reference risk", 680)
    fig_rolling.add_hline(y=0, line_dash="dot", line_color="#777")
    fig_rolling.update_yaxes(title="Annualized rolling Sharpe")

    path_equity = (1 + path_returns).cumprod()
    drawdown = path_equity.div(path_equity.cummax()).sub(1)
    drawdown_long = drawdown.rename_axis("date").reset_index().melt("date", var_name="strategy", value_name="drawdown")
    fig_drawdown = px.line(drawdown_long, x="date", y="drawdown", color="strategy")
    style_plotly(fig_drawdown, "Drawdown paths at the common reference risk", 620)
    fig_drawdown.update_yaxes(title="Drawdown", tickformat=".0%")

    correlation = path_returns.corr()
    fig_correlation = px.imshow(correlation, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    style_plotly(fig_correlation, "Daily net-return correlation, 2024 onward", 690)
    fig_correlation.update_layout(coloraxis_colorbar_title="Correlation")

    xs_grid = headline[(headline.model.eq("xs_ic20_dollar_neutral")) & headline.target_vol.eq(.20) &
                       headline.rebalance.eq("daily")].copy()
    xs_grid = xs_grid.sort_values("validation_net_sharpe", ascending=False).drop_duplicates(["gross_cap", "ticker_risk_cap"])
    xs_pivot = xs_grid.pivot(index="ticker_risk_cap", columns="gross_cap", values="validation_net_sharpe").sort_index()
    xs_pivot.index = [f"{value:.0%}" for value in xs_pivot.index]
    xs_pivot.columns = [f"{value:.1f}×" for value in xs_pivot.columns]
    fig_risk_grid = px.imshow(xs_pivot, text_auto=".2f", color_continuous_scale="Viridis", aspect="auto")
    style_plotly(fig_risk_grid, "Cross-sectional validation Sharpe across construction limits", 620)
    fig_risk_grid.update_xaxes(title="Gross cap"); fig_risk_grid.update_yaxes(title="Single-ticker risk cap")

    horizon_names = [f"breakout_horizon_{h}" for h in (16, 32, 64, 128, 256)]
    breakout_horizons = headline[(headline.model.isin(horizon_names)) & headline.target_vol.eq(.15) &
                                 headline.gross_cap.eq(1) & headline.ticker_risk_cap.eq(.10) &
                                 headline.rebalance.eq("daily")].drop_duplicates("model").copy()
    breakout_horizons["horizon"] = breakout_horizons.model.str.extract(r"(\d+)$").astype(float)
    breakout_plot = breakout_horizons.melt(id_vars=["horizon"], value_vars=["validation_net_sharpe", "holdout_2026_net_sharpe"],
                                           var_name="period", value_name="net_sharpe")
    breakout_plot["period"] = breakout_plot.period.map({"validation_net_sharpe": "Validation 2024–25", "holdout_2026_net_sharpe": "Holdout 2026"})
    fig_horizons = px.line(breakout_plot, x="horizon", y="net_sharpe", color="period", markers=True, log_x=True)
    style_plotly(fig_horizons, "Breakout horizon sensitivity at common risk", 560)
    fig_horizons.update_xaxes(title="Channel horizon (days)", tickvals=[16, 32, 64, 128, 256])
    fig_horizons.update_yaxes(title="Net Sharpe")

    rebalance_models = ["xs_ic20_dollar_neutral", "breakout_horizon_32", "ts_equal_vol90"]
    rebalance = headline[(headline.model.isin(rebalance_models)) & headline.target_vol.eq(.15) &
                         headline.gross_cap.eq(1) & headline.ticker_risk_cap.eq(.10)].copy()
    rebalance = rebalance.sort_values("validation_net_sharpe", ascending=False).drop_duplicates(["model", "rebalance"])
    rebalance["strategy"] = rebalance.model.map(display_names)
    fig_rebalance = px.bar(rebalance, x="strategy", y="validation_net_sharpe", color="rebalance", barmode="group")
    style_plotly(fig_rebalance, "Rebalancing-frequency sensitivity at common risk", 560)
    fig_rebalance.update_xaxes(title=""); fig_rebalance.update_yaxes(title="Validation net Sharpe")

    activation = configs[(configs.model.isin(rebalance_models)) & configs.target_vol.eq(.15) & configs.gross_cap.eq(1) &
                         configs.ticker_risk_cap.eq(.10) & configs.rebalance.eq("daily") &
                         configs.taker_share.eq(1) & configs.slippage_bps.eq(5)].copy()
    activation["strategy"] = activation.model.map(display_names)
    activation_table = activation[["strategy", "activation", "validation_net_sharpe", "holdout_2026_net_sharpe",
                                   "validation_annual_return", "holdout_2026_annual_return"]].sort_values(["strategy", "activation"])

    annual_rows = []
    for year, block in path_returns.groupby(path_returns.index.year):
        for strategy in path_returns:
            annual_rows.append({"year": int(year), "strategy": strategy, **period_metrics(block[strategy])})
    annual_table = pd.DataFrame(annual_rows)
    annual_sharpe = annual_table.pivot(index="strategy", columns="year", values="sharpe").reset_index()
    annual_return_table = annual_table.pivot(index="strategy", columns="year", values="annual_return").reset_index()
    annual_return_table.columns = ["strategy"] + [f"return_{column}" for column in annual_return_table.columns[1:]]
    annual_sharpe.columns = ["strategy"] + [f"sharpe_{column}" for column in annual_sharpe.columns[1:]]
    annual_summary = annual_return_table.merge(annual_sharpe, on="strategy")

    baseline_table = reference[reference.family.eq("baseline")][["model", "validation_net_sharpe", "holdout_2026_net_sharpe",
                                                                  "validation_annual_return", "holdout_2026_annual_return",
                                                                  "validation_max_drawdown", "holdout_2026_max_drawdown",
                                                                  "validation_funding_return"]].copy()
    baseline_table["model"] = baseline_table.model.map(display_names).fillna(baseline_table.model)

    cohort = groups[(groups.model.eq("xs_ic5_20_primary_dollar_neutral")) &
                    groups.grouping.isin(["liquidity_group", "listing_age_days_group", "btc_beta_group", "volatility_group"])][
        ["grouping", "group", "asset_count", "net_return_contribution", "funding_return", "turnover"]
    ].sort_values(["grouping", "group"])

    capacity_table = capacity[(capacity.model.isin(["xs_ic5_20_primary_dollar_neutral", "breakout_shrink_80_primary_quarterly_vol90",
                                                     "ts_shrink_80_primary_quarterly_vol90"]))].copy()
    capacity_table["model"] = capacity_table.model.map(display_names)
    capacity_table["participation"] = capacity_table.participation_of_median_daily_quote_volume.map(pct)
    capacity_table = capacity_table[["model", "participation", "median_capacity_usd", "fifth_percentile_capacity_usd"]]

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

    body = hero("ACAUSAL CAPITAL · CRYPTO MOMENTUM RESEARCH", "Cross-sectional momentum survived realistic execution,",
                "but the evidence is not yet production proof.",
                "A full causal portfolio investigation of absolute EWMAC, relative cross-sectional momentum and channel breakout across Binance USDT perpetuals, including model selection, risk construction, fees, slippage, event funding, capacity and historical holdout evidence.",
                ["Data cutoff 18 Sep 2026", "581-contract fitting panel", "97 eligible portfolio assets",
                 "154 signal models", f"{manifest['configuration_count']:,} saved configurations", "Research / shadow mode"])
    body += """<style>
    .paper-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.paper-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:22px}.paper-card h3{margin-top:0}.formula{background:#ececf3;border-left:4px solid var(--accent);padding:16px 18px;margin:14px 0;border-radius:0 8px 8px 0;font:14px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-x:auto}.decision{border-collapse:separate;border-spacing:0 9px}.decision td{background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.decision td:first-child{border-left:1px solid var(--line);border-radius:8px 0 0 8px}.decision td:last-child{border-right:1px solid var(--line);border-radius:0 8px 8px 0}.status-go{color:#126b58}.status-watch{color:#9a681a}.status-stop{color:#a13e48}.section-dek{color:var(--muted);max-width:1000px;font-size:17px}.chart-scroll{contain:inline-size}.table-wrap{contain:inline-size}.footnote{font-size:12px;color:var(--muted)}
    @media(max-width:900px){.paper-grid{grid-template-columns:1fr}main{overflow-x:clip}figure{contain:inline-size}}
    </style>"""
    body += '<nav><a href="#abstract">Abstract</a><a href="#design">Research design</a><a href="#signals">Signals</a><a href="#construction">Portfolio construction</a><a href="#results">Results</a><a href="#robustness">Robustness</a><a href="#implementation">Costs &amp; funding</a><a href="#cross-section">Cross-section</a><a href="#decision">Decision</a><a href="#appendix">Appendix</a></nav>'
    body += '<section class="metrics">' + metric("Cross-sectional holdout Sharpe", num(xs.holdout_net_sharpe), f"Selected: {xs.model}") + metric("Breakout holdout Sharpe", num(bo.holdout_net_sharpe), f"Selected: {bo.model}") + metric("Time-series holdout Sharpe", num(ts.holdout_net_sharpe), f"Selected: {ts.model}") + metric("Authenticated account funding", f"${account_net:,.2f}", f"{len(actual):,} archived payments") + '</section>'
    body += f'<aside class="callout warning"><strong>Evidence classification: historical holdout, not prospective OOS.</strong> Signals and expanding refits are causal and configuration selection used 2024–25 only; 2026 was disclosed afterward. However, the present-day top-100 fallback creates survivor bias, the research process preceded the holdout, and {manifest["configuration_count"]:,} configurations create material selection risk.</aside>'

    body += '<section id="abstract"><h2>Abstract and investment conclusion</h2><p class="section-dek">The study asks whether crypto trend signals retain economically useful performance after causal activation, portfolio-level risk allocation, Binance fees, conservative slippage and event-level funding. Three signal families are compared on a common infrastructure rather than by averaging ticker Sharpes.</p><div class="findings">'
    body += finding(1, "Relative momentum dominates", f"Dollar-neutral 20-day rank-IC momentum produced {xs.validation_net_sharpe:.2f} validation Sharpe and {xs.holdout_net_sharpe:.2f} in the historical 2026 holdout. At common risk it remained above 1.2 holdout Sharpe.")
    body += finding(2, "Breakout is simpler than expected", f"The 32-day single-horizon channel beat the pooled breakout ensembles: {bo.validation_net_sharpe:.2f} validation and {bo.holdout_net_sharpe:.2f} holdout Sharpe. More parameterization did not improve the evidence.")
    body += finding(3, "Absolute EWMAC is weak", f"The selected EWMAC model fell from {ts.validation_net_sharpe:.2f} validation Sharpe to {ts.holdout_net_sharpe:.2f}. Equal weights beat pooled shrinkage and the legacy Optuna reference at common risk.")
    body += finding(4, "Implementation remains decisive", "The cross-sectional sleeve survives the harshest fee/slippage sensitivity tested, but annual turnover near 90× means live market impact and fill quality can still erase a meaningful portion of the edge.")
    body += finding(5, "Funding separates constructions", f"Absolute trend paid funding—pooled EWMAC lost {abs(ts_funding):.1%} during validation at common risk—while the dollar-neutral sleeve received a small positive contribution.")
    body += finding(6, "Promotion should remain blocked", "The correct next step is a frozen shadow portfolio with actual fills and prospective logs, not additional in-sample optimization or automatic production deployment.")
    body += '</div><h3>Decision summary</h3><div class="table-wrap"><table class="decision"><thead><tr><th>Method</th><th>Evidence</th><th>Decision</th><th>Reason</th></tr></thead><tbody><tr><td>Cross-sectional IC20 dollar-neutral</td><td>Strong validation and holdout</td><td class="status-go"><strong>Advance to frozen shadow</strong></td><td>Robust at common risk and under cost stress; survivor bias and turnover still unresolved.</td></tr><tr><td>32-day breakout</td><td>Moderate, persistent</td><td class="status-watch"><strong>Retain as secondary sleeve</strong></td><td>Simple rule beats ensembles; weaker risk-adjusted performance than cross-sectional momentum.</td></tr><tr><td>Absolute EWMAC</td><td>Weak</td><td class="status-stop"><strong>Do not promote</strong></td><td>Near-zero common-risk holdout performance and negative funding drag.</td></tr><tr><td>Legacy per-ticker Optuna</td><td>Non-causal reference</td><td class="status-stop"><strong>Exclude from selection</strong></td><td>Only 64/97 optimized tickers and static full-sample weights.</td></tr></tbody></table></div></section>'

    body += '<section id="design"><h2>1. Research design and data architecture</h2><p class="section-dek">The analysis is constructed as a portfolio research exercise. Forecasts are calculated per instrument, but performance is evaluated after simultaneous cross-asset sizing, concentration limits, leverage limits, trading costs and funding.</p><div class="paper-grid"><article class="paper-card"><h3>Universe</h3><ul><li>Binance USDT perpetual contracts only; stablecoins and non-standard index contracts excluded.</li><li>Current snapshot ranked by fully diluted valuation, top-100 hysteresis rules and a trailing 30-day median quote-volume floor above $1 million.</li><li>97 contracts passed the minimum 90-close requirement for portfolio simulation.</li><li>581 active and delisted contracts contribute to pooled parameter fitting; delisted history stops at the final tradable candle.</li></ul></article><article class="paper-card"><h3>Timing and prices</h3><ul><li>Daily signals use information available through the completed close.</li><li>Headline P&amp;L uses the following Binance open-to-open interval after a one-candle activation delay.</li><li>Next-close execution is retained as a delay sensitivity.</li><li>Pre-perpetual warm-up can initialize forecasts but never creates futures positions or P&amp;L.</li></ul></article><article class="paper-card"><h3>Estimation protocol</h3><ul><li>At least one year of pooled history before fitted parameters activate; equal weights beforehand.</li><li>Expanding quarterly refits are primary; semiannual, annual and frozen schedules are comparisons.</li><li>Refitted parameters activate on the next daily candle and transition through 90- or 125-day exponential smoothing.</li><li>2024–25 is the selection window; 2026 is the disclosed historical holdout.</li></ul></article><article class="paper-card"><h3>Execution and accounting</h3><ul><li>Headline: 100% taker commission plus 5 bps one-way slippage.</li><li>Sensitivities: 0/50/100% taker and 0/2/5/10 bps slippage.</li><li>Funding settled at every archived timestamp; midnight events use the pre-rebalance position.</li><li>Long and short signs, paid/received cashflow and ticker-to-portfolio reconciliation are tested.</li></ul></article></div><aside class="callout"><strong>Critical universe limitation.</strong> A complete point-in-time FDV archive was unavailable. Portfolio membership therefore applies the current eligible universe backward and is labelled <code>current_universe_historical_fallback</code>. This can remove failed assets and materially overstate historical results.</aside></section>'

    body += '<section id="signals"><h2>2. Signal definitions and causal model fitting</h2><div class="paper-grid"><article class="paper-card"><h3>Absolute EWMAC</h3><p>Five fast/slow exponential moving-average differences—2/8, 4/16, 8/32, 16/64 and 32/128—are normalized by a causal price-volatility estimate. Warm-up observations remain missing and each component is scaled toward a median absolute forecast of 10.</p><div class="formula">EWMAC<sub>i,t</sub> = [EMA<sub>fast</sub>(P<sub>i,t</sub>) − EMA<sub>slow</sub>(P<sub>i,t</sub>)] / σ<sub>P,i,t</sub><br>F<sub>i,t</sub> = clip(DFM<sub>t</sub> · Σ<sub>r</sub> w<sub>r,t</sub> · scaled(EWMAC<sub>i,r,t</sub>), −20, +20)</div><p>Equal weights, 75/80/90%-shrunk pooled weights and the legacy per-ticker Optuna weights are compared across 60/90/180/360-day volatility windows.</p></article><article class="paper-card"><h3>Channel breakout</h3><p>Each component locates the close within its trailing high/low channel. Horizons are 16, 32, 64, 128 and 256 days. Equal and shrunk pooled horizon combinations receive the same refit and risk grid as EWMAC.</p><div class="formula">B<sub>i,h,t</sub> = clip(40 · [P<sub>i,t</sub> − (High<sub>h</sub> + Low<sub>h</sub>)/2] / [High<sub>h</sub> − Low<sub>h</sub>], −20, +20)</div><p>Five single-horizon controls isolate whether diversification across horizons adds value. In this sample, the 32-day rule dominates the more complicated ensembles.</p></article><article class="paper-card"><h3>Cross-sectional momentum</h3><p>The signed pooled momentum forecast is ranked across eligible contracts each day. Rule weights are estimated from expanding 1-, 5- and 20-day rank information coefficients; the primary comparison averages 5- and 20-day IC.</p><div class="formula">R<sub>i,t</sub> = percentile_rank(F<sub>i,t</sub>)<br>w<sub>i,t</sub><sup>DN</sup> ∝ R<sub>i,t</sub> − 0.5, with Σw<sub>i,t</sub> = 0 and Σ|w<sub>i,t</sub>| = 1</div><p>Continuous dollar-neutral, top/bottom baskets, long-only, BTC-hedged, beta-constrained and BTC-residualized variants are tested.</p></article><article class="paper-card"><h3>Shrinkage and diversification</h3><p>Unconstrained historical scores are deliberately pulled toward equal weights. The primary model uses 80% shrinkage, a 25% maximum rule weight and 125-day smoothing. Forecast correlations produce a capped diversification multiplier.</p><div class="formula">w<sub>t</sub> = Project<sub>cap</sub>[λ · w<sub>equal</sub> + (1−λ) · w<sub>fitted,t</sub>]<br>DFM<sub>t</sub> = min([w′Corr<sub>t</sub>w]<sup>−1/2</sup>, 2.5)</div><p>The purpose is stability, not maximizing retrospective Sharpe. Legacy Optuna is shown only as a non-causal reference.</p></article></div>' + figure_html(chart_html(fig_models), "Model-level validation versus holdout", "Each point is a model’s validation-best headline risk setting. Legacy static Optuna remains visible but is excluded from causal family selection.", "../data_store/crypto_momentum_research/complete_results/tested_configurations.parquet") + '</section>'

    body += '<section id="construction"><h2>3. Portfolio construction</h2><p class="section-dek">Forecasts are converted into risk units using causal trailing volatility, capped at the ticker level, scaled to the portfolio target and finally constrained by gross exposure. Positions rebalance only at the specified schedule and become effective on the next candle.</p><div class="formula">u<sub>i,t</sub> ∝ signal_budget<sub>i,t</sub> / σ<sub>i,t</sub><br>w*<sub>i,t</sub> = u<sub>i,t</sub> · target_vol / predicted_portfolio_vol<sub>t</sub><br>w<sub>t</sub> = GrossCap(w*<sub>t</sub>, 1×…3×), &nbsp; |risk contribution<sub>i,t</sub>| ≤ 5%…40%</div><div class="paper-grid"><article class="paper-card"><h3>Risk grid</h3><p>Annual volatility targets: 15%, 20%, 25%, 30%, 35% and 40%. Gross caps: 1.0× through 3.0× in 0.5× steps. Single-ticker risk caps: 5% through 40% in five-point steps.</p></article><article class="paper-card"><h3>Trading schedule</h3><p>Daily, Monday-weekly and first-UTC-day monthly schedules are compared. Cross-sectional basket variants also test 5% and 10% rank buffers to reduce unnecessary turnover.</p></article><article class="paper-card"><h3>Neutrality variants</h3><p>Cross-sectional portfolios include unhedged signed exposure, dollar-neutral weights, BTC hedging after construction, direct beta constraints, BTC-residualized forecasts and positive-forecast long-only exposure with unused risk held in cash.</p></article><article class="paper-card"><h3>Common-risk reference</h3><p>To separate signal quality from risk maximization, the paper repeatedly compares models at a common 15% target, 1× gross cap, 10% ticker cap, daily rebalance, 100% taker and 5 bps slippage.</p></article></div>' + figure_html(chart_html(fig_risk_grid), "Cross-sectional risk-grid sensitivity", "Validation Sharpe for the IC20 dollar-neutral model at a 20% target. Broad plateaus are preferable to isolated maxima.", "../data_store/crypto_momentum_research/complete_results/tested_configurations.parquet") + figure_html(chart_html(fig_rebalance), "Rebalancing-frequency comparison", "Common-risk validation results for the leading model from each standalone family.", "../data_store/crypto_momentum_research/complete_results/tested_configurations.parquet") + '<h3>Next-open versus next-close activation</h3>' + format_table(activation_table) + '</section>'

    body += '<section id="results"><h2>4. Portfolio results</h2><p class="section-dek">Headline configurations are chosen only from 2024–25 validation results under the stated cost assumptions. The 2026 column is disclosed after selection and is not used to choose the winner.</p>' + figure_html(chart_html(fig_family), "Validation-selected family results", "The fall from validation to holdout is the relevant model-stability test.", "../data_store/crypto_momentum_research/complete_results/selected_configurations.json") + '<h3>Validation-selected configurations</h3>' + format_table(selected_table) + '<h3>Interpretation</h3><p>The cross-sectional result is materially stronger than the absolute-directional families: 2.49 validation Sharpe and 1.32 holdout Sharpe at the selected risk setting, versus 0.72/0.40 for breakout and 0.42/0.18 for EWMAC. The deterioration is meaningful but not a complete collapse. The selected cross-sectional construction also retains performance at common risk, which argues against leverage alone explaining the result.</p>' + figure_html(chart_html(fig_equity), "Common-risk growth of $1", "All paths use 15% target, 1× gross, 10% ticker cap, daily rebalance, taker fees and 5 bps slippage. This chart does not use each family’s selected maximum-risk configuration.", "../data_store/crypto_momentum_research/complete_results/default_daily_returns.parquet") + '<h3>Common-risk methodology comparison</h3>' + format_table(reference_table) + '<h3>Calendar-year decomposition at common risk</h3>' + format_table(annual_summary) + '</section>'

    body += '<section id="robustness"><h2>5. Stability, drawdowns and parameter robustness</h2><p class="section-dek">A strategy should not be judged only by an aggregate Sharpe. This section shows time variation, drawdown persistence, sleeve correlation and whether the central conclusions survive nearby parameter choices.</p>' + figure_html(chart_html(fig_rolling), "Rolling risk-adjusted performance", "180-day annualized Sharpe from daily net returns; short windows are noisy and are presented as regime diagnostics.", "../data_store/crypto_momentum_research/complete_results/default_daily_returns.parquet") + figure_html(chart_html(fig_drawdown), "Drawdown experience", "Common-risk net return paths from 2024 onward.", "../data_store/crypto_momentum_research/complete_results/default_daily_returns.parquet") + figure_html(chart_html(fig_correlation), "Diversification structure", "Daily net-return correlations; low correlation alone does not justify adding a low-quality sleeve.", "../data_store/crypto_momentum_research/complete_results/default_daily_returns.parquet") + figure_html(chart_html(fig_horizons), "Breakout horizon study", "Single-horizon results at common risk. The 32-day channel is the strongest causal breakout reference.", "../data_store/crypto_momentum_research/complete_results/tested_configurations.parquet") + '<p>The equal-risk EWMAC/cross-sectional diagnostic reached only ' + num(best['diagnostic_combination'].holdout_net_sharpe) + ' holdout Sharpe. Diversification did not offset the low expected return and funding drag of absolute EWMAC. Likewise, pooled breakout weighting did not improve on the 32-day rule; complexity is not supported by the current sample.</p></section>'

    body += '<section id="implementation"><h2>6. Implementation: fees, slippage, turnover, funding and capacity</h2><h3>Transaction-cost stress</h3><p>The headline assumption is intentionally conservative for daily research: current Binance symbol commissions, 100% taker execution and 5 bps one-way slippage. The sensitivity below varies both taker share and slippage while holding the portfolio risk setting constant.</p>' + figure_html(chart_html(fig_cost), "Fee and slippage sensitivity", "The cross-sectional edge declines monotonically with cost but remains positive at the harshest tested setting; this is not a full nonlinear market-impact model.", "../data_store/crypto_momentum_research/complete_results/tested_configurations.json") + '<h3>Turnover and capacity screen</h3><p>The selected cross-sectional portfolio turns over roughly 90× annually. At one-percent participation in trailing median Binance quote volume, the primary dollar-neutral sleeve’s fifth-percentile daily capacity screen is <strong>$' + f'{capacity_value:,.0f}' + '</strong>. Capacity is estimated from median quote volume and target trades; it does not model book depth, volatility-dependent impact or simultaneous liquidation.</p>' + format_table(capacity_table) + '<h3>Funding methodology and evidence</h3><div class="formula">funding_cost_usd = quantity × mark_price × funding_rate<br>funding_cashflow_usd = −funding_cost_usd</div><p>Positive cost means the position paid; positive cashflow means it received. Coincident midnight events settle against the position held immediately before rebalancing. Missing required public funding fails the headline run rather than silently assuming zero.</p>' + figure_html(chart_html(fig_funding), "Modeled funding paid, received and net", "Common-risk models; ticker, side and year detail are retained in the downloadable summary.", "../data_store/crypto_momentum_research/complete_results/modeled_funding_summary.csv") + f'<p>The public archive contains <strong>433,572</strong> funding events across 100 requested symbols. Authenticated Binance account history contains <strong>{len(actual):,}</strong> payments from February 2021 through September 2026, totaling <strong>${account_net:,.2f}</strong>. Only {matched_count:,} records have direction agreeing with the default model; the remainder stay explicitly unmatched rather than being attributed to the strategy.</p></section>'

    body += '<section id="baselines"><h2>7. Baselines and alternative explanations</h2><p class="section-dek">Momentum evidence is credible only if it improves on simple exposures. BTC buy-and-hold, equal-weight eligible coins, FDV-weighted eligible coins and a deterministic no-skill portfolio are included.</p>' + format_table(baseline_table) + '<p>The FDV-weighted basket scored 0.86 validation Sharpe but only 0.13 in the holdout. Conversely, the equal-weight basket was weak in validation and unusually strong during 2026. This confirms that the historical holdout contains a favourable broad-alt regime. Dollar-neutral cross-sectional momentum is more informative than the long-only baselines because it retains strong results without relying on positive market beta, but survivor bias remains capable of inflating both sides of the comparison.</p></section>'

    body += '<section id="cross-section"><h2>8. Ticker, sector and cohort diagnostics</h2><p class="section-dek">The research system retains individual component forecasts, percentile ranks, target positions and ticker-level return/funding/turnover attribution. These diagnostics identify whether results depend on a few contracts or a particular market segment.</p>' + figure_html(chart_html(fig_strength), "Current absolute momentum strength", "The latest pooled EWMAC forecast is shown for the strongest and weakest tickers; every final forecast is capped at ±20.", "../data_store/crypto_momentum_research/complete_results/latest_signal_records.csv") + figure_html(chart_html(fig_sector), "Cross-sectional contribution by sector", "Sector labels use the archived provider metadata and the current-universe historical fallback.", "../data_store/crypto_momentum_research/complete_results/group_attribution.parquet") + '<h3>Cross-sectional cohort attribution</h3>' + format_table(cohort) + '<p><a href="crypto_momentum_ticker_analytics_20260920.html"><strong>Open the complete searchable 97-ticker companion report →</strong></a></p><p class="footnote">Ticker contribution is measured inside the portfolio and must not be interpreted as standalone ticker Sharpe. Account funding includes all archived Binance account activity and remains separate from model-attributed funding.</p></section>'

    body += '<section id="decision"><h2>9. Research decision and next experiment</h2><div class="paper-grid"><article class="paper-card"><h3>Freeze</h3><p>Freeze one parsimonious cross-sectional configuration—IC20, continuous dollar-neutral construction—and the simple 32-day breakout control. Record their exact JSON parameters and hashes without further tuning.</p></article><article class="paper-card"><h3>Shadow</h3><p>Emit next-open target positions daily, preserve model and data hashes, and compare reconstructed targets with actually emitted signals. Record quoted spread, fill price, fee tier and funding cashflow.</p></article><article class="paper-card"><h3>Evaluate prospectively</h3><p>Report live turnover, implementation shortfall, concentration, beta, funding and capacity beside net returns. Do not use the forward period for additional parameter search.</p></article><article class="paper-card"><h3>Promotion gate</h3><p>Require an explicit later decision. The current report deliberately imposes no automatic Sharpe threshold; the purpose of the forward period is to resolve survivor bias, cost realism and model-decay uncertainty.</p></article></div><aside class="callout"><strong>Recommended research posture.</strong> Advance cross-sectional momentum to frozen shadow monitoring; retain 32-day breakout as a secondary benchmark; stop allocating research complexity to EWMAC or static Optuna until prospective evidence justifies reopening them.</aside></section>'

    body += f'<section id="limitations"><h2>10. Limitations</h2><ol><li><strong>Survivor bias:</strong> historical portfolio membership uses the current eligible universe because complete point-in-time FDV snapshots were unavailable.</li><li><strong>Multiple testing:</strong> {manifest["configuration_count"]:,} configurations are highly dependent but still create selection risk; raw Sharpe ranks are not corrected for data mining.</li><li><strong>Historical holdout:</strong> 2026 is causally untouched by the selection rule, but it is not a prospective period conceived before the broader research process.</li><li><strong>Optuna contamination:</strong> legacy static weights cover only 64 of 97 tickers and may include later data. They are excluded from causal selection.</li><li><strong>Market impact:</strong> the simulator includes fees and fixed slippage but not a nonlinear order-book impact model or exchange outages.</li><li><strong>Funding reconstruction:</strong> public rates are complete for the requested study symbols, while many account records outside that set cannot be matched to a public rate or model position.</li><li><strong>Borrow and operational constraints:</strong> the study assumes perpetual availability and does not model exchange-specific position limits, liquidation mechanics or collateral haircuts.</li><li><strong>Short holdout:</strong> the 2026 sample contains {int(xs.holdout_observations):,} daily observations and may represent only a small number of crypto regimes.</li></ol></section>'

    body += '<section id="appendix"><h2>Appendix A — Complete model results</h2><p>The table reports the validation-best headline risk setting for every signal model. It includes causal models and explicitly labelled legacy references; the JSON registry retains every risk, execution and delay scenario.</p>' + format_table(appendix_table) + f'<h2>Appendix B — Reproducibility and data products</h2><ul><li>Input hashes are recorded in the study manifest; every research scenario has a deterministic configuration ID.</li><li>The canonical SQLite ledger contains {ledger_config_count:,} tested configurations and {ledger_signal_count:,} daily ticker/model signal records.</li><li>Headline returns use actual archived Binance daily opens and the next open-to-open interval.</li><li>Warm-up missing values are preserved, final forecasts are capped at ±20 and no future observation may alter an earlier signal, rank, refit or position.</li></ul><h3>Downloads</h3><p><a href="../data_store/crypto_momentum_research/complete_results/tested_configurations.json">All configurations JSON</a> · <a href="../data_store/crypto_momentum_research/complete_results/deployable_configurations.json">Deployable parameter JSON</a> · <a href="../data_store/crypto_momentum_research/complete_results/selected_configurations.json">Selected results JSON</a> · <a href="../data_store/crypto_momentum_research/complete_results/study_manifest.json">Run manifest</a> · <a href="../data_store/crypto_momentum_research/complete_results/default_daily_returns.parquet">Common-risk daily returns</a> · <a href="../data_store/crypto_momentum_research/complete_results/modeled_funding_summary.csv">Modeled funding</a> · <a href="../data_store/crypto_momentum_research/complete_results/actual_funding_summary.csv">Actual account funding</a> · <a href="../data_store/crypto_momentum_research/complete_results/crypto_momentum_research.sqlite">SQLite ledger</a></p><h3>Rebuild</h3><pre>python scripts/archive_portfolio_open_prices.py\npython scripts/execute_complete_crypto_study.py\npython scripts/reconcile_actual_funding_archive.py\npython scripts/build_funding_analysis.py\npython scripts/build_full_crypto_study_report.py\npython scripts/build_crypto_ticker_analytics_report.py</pre></section>'

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_page("Complete Crypto Momentum Study", body, plotly=True, accent="purple"), encoding="utf-8")
    sources = ["tested_configurations.parquet", "selected_configurations.json", "default_daily_returns.parquet",
               "default_daily_funding.parquet", "modeled_funding_summary.csv", "latest_signal_records.csv",
               "group_attribution.parquet", "capacity_analysis.csv", "actual_funding_reconciliation.parquet",
               "study_manifest.json"]
    report_manifest = {"report": str(OUTPUT), "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "study_generated_at": manifest["generated_at"], "generator": "scripts/build_full_crypto_study_report.py", "rebuild": "python scripts/build_full_crypto_study_report.py", "inputs": {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in sources}}
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
