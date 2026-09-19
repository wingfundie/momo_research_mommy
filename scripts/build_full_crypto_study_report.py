from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_theme.report_theme import figure_html, finding, hero, metric, render_page, style_plotly

RESULTS = ROOT / "data_store/crypto_momentum_research/results"
OUTPUT = ROOT / "reports/crypto_momentum_full_study_20260920.html"


def chart_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def main():
    scenarios = pd.read_parquet(RESULTS / "scenario_results.parquet")
    daily = pd.read_parquet(RESULTS / "headline_daily_returns.parquet")
    strength = pd.read_csv(RESULTS / "ticker_strength_and_funding.csv", index_col=0)
    summary = json.loads((RESULTS / "study_summary.json").read_text(encoding="utf-8"))
    refs = scenarios[~scenarios.model.isin(["shrink_80_primary", "cost_sensitivity"])].copy()
    grid = scenarios[scenarios.model.eq("shrink_80_primary")].copy()
    best = scenarios.iloc[0]
    best_ts = grid[grid.sleeve.eq("time_series_pooled")].sort_values("net_sharpe", ascending=False).iloc[0]
    best_xs = grid[grid.sleeve.eq("cross_sectional_continuous")].sort_values("net_sharpe", ascending=False).iloc[0]

    equity = (1 + daily.fillna(0)).cumprod()
    eq_long = equity.reset_index(names="date").melt("date", var_name="sleeve", value_name="growth")
    fig_equity = style_plotly(px.line(eq_long, x="date", y="growth", color="sleeve"),
                              "Growth of $1 — standalone reference sleeves", 620)
    fig_equity.update_yaxes(title="Growth multiple")

    fig_grid = px.scatter(grid, x="annual_volatility", y="annual_return", color="sleeve",
                          symbol="rebalance", hover_data=["target_vol", "gross_cap", "ticker_cap", "net_sharpe"])
    style_plotly(fig_grid, "Risk grid — net return versus realized volatility", 600)
    fig_grid.update_xaxes(tickformat=".0%", title="Realized annual volatility")
    fig_grid.update_yaxes(tickformat=".0%", title="Net annual return")

    funding = refs[["sleeve", "funding_paid_return", "funding_received_return", "funding_return"]].copy()
    funding_long = funding.melt("sleeve", var_name="measure", value_name="return")
    fig_funding = px.bar(funding_long, x="sleeve", y="return", color="measure", barmode="group")
    style_plotly(fig_funding, "Modeled funding paid, received and net", 570)
    fig_funding.update_yaxes(tickformat=".1%", title="Cumulative return contribution")
    fig_funding.update_xaxes(title="")

    usable_strength = strength.dropna(subset=["forecast"])
    extremes = pd.concat([usable_strength.nlargest(10, "forecast"), usable_strength.nsmallest(10, "forecast")])
    fig_strength = px.bar(extremes.sort_values("forecast"), x="forecast", y=extremes.sort_values("forecast").index,
                          orientation="h", color="forecast", color_continuous_scale="RdYlGn")
    style_plotly(fig_strength, "Latest individual-ticker momentum strength", 650)
    fig_strength.update_layout(coloraxis_showscale=False)

    top_table = scenarios[["sleeve", "target_vol", "gross_cap", "ticker_cap", "rebalance", "net_sharpe",
                           "annual_return", "annual_volatility", "max_drawdown", "funding_return"]].head(15).copy()
    for col in ["target_vol", "ticker_cap", "annual_return", "annual_volatility", "max_drawdown", "funding_return"]:
        top_table[col] = top_table[col].map(lambda x: f"{x:.1%}")
    top_table["net_sharpe"] = top_table["net_sharpe"].map(lambda x: f"{x:.2f}")
    top_table["gross_cap"] = top_table["gross_cap"].map(lambda x: f"{x:.1f}x")

    body = hero("ACAUSAL CAPITAL · CRYPTO MOMENTUM", "Cross-sectional momentum led,",
                "but survivor bias still dominates the verdict.",
                "A causal, portfolio-level shadow study with event funding, execution costs and 1,460 risk scenarios.",
                ["Data cutoff 18 Sep 2026", "100-name current FDV universe", "97 names with ≥90 closes",
                 "Current-universe historical fallback", "Shadow research only"])
    body += '<nav><a href="#findings">Findings</a><a href="#portfolio">Portfolio</a><a href="#funding">Funding</a><a href="#tickers">Ticker strength</a><a href="#methods">Methods</a></nav>'
    body += '<section class="metrics">'
    body += metric("Best raw net Sharpe", f"{best.net_sharpe:.2f}", str(best.sleeve))
    body += metric("Best time-series Sharpe", f"{best_ts.net_sharpe:.2f}", f"{best_ts.annual_return:.1%} net return")
    body += metric("Best cross-sectional Sharpe", f"{best_xs.net_sharpe:.2f}", f"{best_xs.annual_return:.1%} net return")
    body += metric("Account funding, recent", f"${summary['account_net_funding_usd']:,.2f}", f"{summary['account_funding_events']:,} authenticated records")
    body += '</section>'
    body += '<aside class="callout warning"><strong>Scope limitation.</strong> Historical point-in-time FDV snapshots were not configured. The present-day top-100 universe is applied backward, creating material survivor bias. Rankings are exploratory and not promotion evidence.</aside>'
    body += '<section id="findings"><h2>What the run says</h2><div class="findings">'
    body += finding(1, "Cross-sectional structure was stronger", f"The best continuous cross-sectional grid result reached {best_xs.net_sharpe:.2f} net Sharpe versus {best_ts.net_sharpe:.2f} for pooled time-series momentum.")
    body += finding(2, "Long-only was the raw winner", f"Positive-forecast top-20% long-only produced {best.net_sharpe:.2f} Sharpe, {best.annual_return:.1%} annual return and {best.max_drawdown:.1%} maximum drawdown at {best.annual_volatility:.1%} realized volatility.")
    body += finding(3, "Funding separated the sleeves", f"Time-series funding cost {best_ts.funding_return:.1%} cumulatively in its best grid result, while the best continuous cross-sectional construction gained {best_xs.funding_return:.1%}.")
    body += finding(4, "Pooled weights added no visible edge", "At the primary settings, pooled and equal-rule time-series results were identical. The fitted rule optimizer effectively collapsed toward its equal anchor, so complexity was not rewarded in this sample.")
    body += finding(5, "Costs matter, but did not erase the signal", "For primary time-series momentum, moving from all-maker/no-slippage execution to the 100%-taker/5-bps headline reduced net Sharpe from 0.74 to 0.70.")
    body += finding(6, "The best grid is not independent evidence", "The reported maximum was selected from 1,460 scenarios. No multiple-testing correction or untouched prospective period has yet been applied.")
    body += '</div></section>'
    body += '<section id="portfolio"><h2>Portfolio comparisons</h2>'
    body += figure_html(chart_html(fig_equity), "Standalone sleeve growth", "Net of configured fees, slippage and historical event funding.", "../data_store/crypto_momentum_research/results/headline_daily_returns.parquet")
    body += figure_html(chart_html(fig_grid), "Full primary risk grid", "Daily, weekly and monthly rebalancing across the requested target, leverage and ticker-cap ranges.", "../data_store/crypto_momentum_research/results/scenario_results.csv")
    body += '<h3>Leading raw configurations</h3><div class="table-scroll" tabindex="0">' + top_table.to_html(index=False, border=0) + '</div></section>'
    body += '<section id="funding"><h2>Funding economics</h2>'
    body += figure_html(chart_html(fig_funding), "Funding decomposition by reference sleeve", "Positive net values are received; negative values are paid.", "../data_store/crypto_momentum_research/results/headline_daily_funding_returns.parquet")
    body += f'<p>The archive contains <strong>{summary["funding_events"]:,}</strong> public settlement events. Authenticated recent account history contains <strong>{summary["account_funding_events"]:,}</strong> records and nets <strong>${summary["account_net_funding_usd"]:,.2f}</strong>. Account cashflows are reported separately because the account can contain exposure unrelated to this model.</p></section>'
    body += '<section id="tickers"><h2>Individual ticker strength</h2>'
    body += figure_html(chart_html(fig_strength), "Strongest and weakest current forecasts", "Signed final pooled EWMAC forecast; final forecast cap is ±20.", "../data_store/crypto_momentum_research/results/ticker_strength_and_funding.csv")
    body += '</section>'
    body += '<section id="methods"><h2>Methodology and coverage</h2><p>Signals use five causal EWMAC rules, a 90-day volatility estimate, quarterly expanding refits, an 80% equal-weight anchor, a 25% rule cap, 125-day transitions and next-candle activation. Portfolio results include Binance event funding, 100% taker fees and five basis points of one-way slippage in the headline. The risk grid spans 15–40% targets, 1–3x gross caps, 5–40% ticker risk caps and three rebalance frequencies.</p><p>Three selected universe members had fewer than 90 valid closes and were excluded from modeled positions, leaving 97 tradable research series. Pre-perpetual prices were not needed for these local series and no pre-listing P&amp;L was generated.</p><h3>Rebuild</h3><pre>python scripts/execute_full_crypto_study.py\npython scripts/build_full_crypto_study_report.py</pre></section>'
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_page("Full Crypto Momentum Study", body, plotly=True, accent="purple"), encoding="utf-8")
    manifest = {"report": str(OUTPUT), "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "inputs": {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in
                           ["scenario_results.parquet", "headline_daily_returns.parquet", "ticker_strength_and_funding.csv"]},
                "generator": "scripts/build_full_crypto_study_report.py", "theme": "editorial-html-report/1.0"}
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__": main()
