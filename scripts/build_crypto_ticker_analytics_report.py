from __future__ import annotations

import hashlib
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_theme.report_theme import figure_html, hero, metric, render_page, style_plotly


RESULTS = ROOT / "data_store/crypto_momentum_research/complete_results"
INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"
OUTPUT = ROOT / "reports/crypto_momentum_ticker_analytics_20260920.html"
SNAPSHOT = RESULTS / "ticker_analytics_snapshot.csv"

TS_MODEL = "ts_shrink_80_primary_quarterly_vol90"
XS_MODEL = "xs_ic5_20_primary_dollar_neutral"
BO_MODEL = "breakout_shrink_80_primary_quarterly_vol90"
DISPLAY_MODELS = (TS_MODEL, XS_MODEL, BO_MODEL)


def fmt(value, digits=2, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}{suffix}"


def pct(value, digits=1):
    return "—" if value is None or pd.isna(value) else f"{float(value):.{digits}%}"


def money(value):
    if value is None or pd.isna(value):
        return "—"
    sign = "−" if float(value) < 0 else ""
    return f"{sign}${abs(float(value)):,.2f}"


def table_html(frame: pd.DataFrame, classes="") -> str:
    return f'<div class="table-scroll" tabindex="0">{frame.to_html(index=False, border=0, classes=classes, escape=True)}</div>'


def sparkline(series: pd.Series, width=470, height=92) -> str:
    values = series.dropna().tail(365).to_numpy(float)
    if len(values) < 2:
        return '<div class="spark-empty">Insufficient price history</div>'
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    scaled = np.full(len(values), height / 2) if hi <= lo else height - 5 - (values - lo) / (hi - lo) * (height - 10)
    x = np.linspace(5, width - 5, len(values))
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, scaled))
    colour = "#268a87" if values[-1] >= values[0] else "#b55d64"
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" aria-label="Trailing 365-day price path">'
            f'<polyline fill="none" stroke="{colour}" stroke-width="2.2" points="{points}"/></svg>')


def current_components(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).ffill()
    latest = frame.iloc[-1]
    output = latest.unstack(level=0)
    output.index.name = "symbol"
    return output


def main():
    manifest = json.loads((RESULTS / "study_manifest.json").read_text(encoding="utf-8"))
    latest = pd.read_csv(RESULTS / "latest_signal_records.csv")
    signals = pd.read_parquet(
        RESULTS / "daily_signal_records.parquet",
        columns=["timestamp", "model", "symbol", "forecast", "cross_sectional_percentile", "target_position", "quality_flag"],
    )
    signals["timestamp"] = pd.to_datetime(signals["timestamp"])
    attribution = pd.read_parquet(RESULTS / "ticker_attribution.parquet")
    actual = pd.read_parquet(RESULTS / "actual_funding_reconciliation.parquet")
    expected = pd.read_csv(RESULTS / "expected_funding_snapshot.csv")
    opens = pd.read_parquet(INPUTS / "portfolio_open_prices.parquet")
    ts_components = current_components(RESULTS / "primary_ts_component_forecasts.parquet")
    bo_components = current_components(RESULTS / "primary_breakout_component_forecasts.parquet")

    symbols = sorted(latest.symbol.unique())
    latest_wide = latest.pivot(index="symbol", columns="model", values=["forecast", "cross_sectional_percentile", "target_position"])
    market_returns = opens.pct_change(fill_method=None)
    btc = market_returns.get("BTCUSDT")
    snapshot = pd.DataFrame(index=symbols)
    snapshot.index.name = "symbol"
    snapshot["last_open"] = opens.ffill().iloc[-1].reindex(symbols)
    snapshot["return_30d"] = opens.ffill().iloc[-1].reindex(symbols) / opens.ffill().iloc[-31].reindex(symbols) - 1
    snapshot["return_90d"] = opens.ffill().iloc[-1].reindex(symbols) / opens.ffill().iloc[-91].reindex(symbols) - 1
    snapshot["volatility_90d"] = market_returns.rolling(90, min_periods=60).std().iloc[-1].reindex(symbols) * np.sqrt(365)
    if btc is not None:
        snapshot["btc_beta_90d"] = market_returns.rolling(90, min_periods=60).cov(btc).iloc[-1].reindex(symbols) / btc.rolling(90, min_periods=60).var().iloc[-1]

    for prefix, model in (("ts", TS_MODEL), ("xs", XS_MODEL), ("breakout", BO_MODEL)):
        for field in ("forecast", "cross_sectional_percentile", "target_position"):
            snapshot[f"{prefix}_{field}"] = latest_wide.get((field, model), pd.Series(index=symbols, dtype=float)).reindex(symbols)

    meta = attribution[attribution.model.eq(TS_MODEL)].drop_duplicates("symbol").set_index("symbol")
    for column in ("liquidity", "sector", "listing_age_days"):
        snapshot[column] = meta.get(column, pd.Series(index=symbols, dtype=float)).reindex(symbols)

    actual_summary = actual.groupby("symbol").income_usd.agg(
        actual_funding_net="sum", actual_funding_events="count"
    )
    actual_summary["actual_funding_paid"] = actual[actual.income_usd.lt(0)].groupby("symbol").income_usd.sum()
    actual_summary["actual_funding_received"] = actual[actual.income_usd.gt(0)].groupby("symbol").income_usd.sum()
    snapshot = snapshot.join(actual_summary)
    snapshot["actual_funding_events"] = snapshot.actual_funding_events.fillna(0).astype(int)

    expected_primary = expected[expected.model.isin(DISPLAY_MODELS)].pivot(
        index="symbol", columns="model", values="next_24h_expected_return"
    )
    snapshot["expected_funding_24h_ts"] = expected_primary.get(TS_MODEL, pd.Series(dtype=float)).reindex(symbols)
    snapshot["expected_funding_24h_xs"] = expected_primary.get(XS_MODEL, pd.Series(dtype=float)).reindex(symbols)
    snapshot["expected_funding_24h_breakout"] = expected_primary.get(BO_MODEL, pd.Series(dtype=float)).reindex(symbols)

    signal_subset = signals[signals.model.isin(DISPLAY_MODELS) & signals.quality_flag.eq("valid")].copy()
    signal_stats = signal_subset.groupby(["symbol", "model"]).agg(
        forecast_mean=("forecast", "mean"), forecast_std=("forecast", "std"),
        mean_abs_forecast=("forecast", lambda x: x.abs().mean()),
        positive_share=("forecast", lambda x: x.gt(0).mean()), valid_days=("forecast", "count"),
    ).reset_index()
    signal_subset["direction"] = np.sign(signal_subset.forecast)
    flips = signal_subset.sort_values("timestamp").groupby(["symbol", "model"]).direction.apply(
        lambda x: int(x.ne(x.shift()).sum() - 1)
    ).rename("direction_flips").reset_index()
    signal_stats = signal_stats.merge(flips, on=["symbol", "model"], how="left")

    attribution_primary = attribution[attribution.model.isin(DISPLAY_MODELS)].copy()
    for prefix, model in (("ts", TS_MODEL), ("xs", XS_MODEL), ("breakout", BO_MODEL)):
        part = attribution_primary[attribution_primary.model.eq(model)].set_index("symbol")
        snapshot[f"{prefix}_net_contribution"] = part.net_return_contribution.reindex(symbols)
        snapshot[f"{prefix}_funding_return"] = part.funding_return.reindex(symbols)
        snapshot[f"{prefix}_turnover"] = part.turnover.reindex(symbols)

    snapshot.reset_index().to_csv(SNAPSHOT, index=False)

    chart_data = snapshot.reset_index().copy()
    chart_data["liquidity_plot"] = chart_data.liquidity.fillna(chart_data.liquidity.median()).clip(lower=1)
    fig_strength = px.scatter(
        chart_data, x="ts_forecast", y="breakout_forecast", color="xs_cross_sectional_percentile",
        size="liquidity_plot", hover_name="symbol", hover_data=["sector", "return_30d", "volatility_90d"],
        color_continuous_scale="RdBu", color_continuous_midpoint=.5,
    )
    style_plotly(fig_strength, "Current time-series versus breakout strength", 680)
    fig_strength.update_xaxes(title="Pooled EWMAC forecast (−20 to +20)")
    fig_strength.update_yaxes(title="Pooled breakout forecast (−20 to +20)")
    fig_strength.update_layout(coloraxis_colorbar_title="XS rank")

    contribution = chart_data.dropna(subset=["ts_net_contribution", "breakout_net_contribution"])
    fig_contribution = px.scatter(
        contribution, x="ts_net_contribution", y="breakout_net_contribution", color="sector",
        size="liquidity_plot", hover_name="symbol", hover_data=["xs_net_contribution", "actual_funding_net"],
    )
    style_plotly(fig_contribution, "Post-training ticker contribution: EWMAC versus breakout", 700)
    fig_contribution.update_xaxes(title="EWMAC cumulative net-return contribution", tickformat=".1%")
    fig_contribution.update_yaxes(title="Breakout cumulative net-return contribution", tickformat=".1%")

    funding_rank = chart_data.nlargest(25, "actual_funding_events").sort_values("actual_funding_net")
    fig_funding = px.bar(funding_rank, x="symbol", y=["actual_funding_received", "actual_funding_paid"], barmode="relative")
    style_plotly(fig_funding, "Authenticated account funding for the 25 most-active symbols", 610)
    fig_funding.update_xaxes(title=""); fig_funding.update_yaxes(title="USD cashflow")

    directory = chart_data[["symbol", "sector", "ts_forecast", "breakout_forecast", "xs_cross_sectional_percentile",
                            "return_30d", "volatility_90d", "btc_beta_90d", "liquidity", "actual_funding_net"]].copy()
    directory.columns = ["Ticker", "Sector", "TS forecast", "Breakout forecast", "XS percentile",
                         "30d return", "90d vol", "BTC beta", "Median quote volume", "Account funding"]
    for column in ("30d return", "90d vol", "XS percentile"):
        directory[column] = directory[column].map(lambda x: pct(x))
    directory["TS forecast"] = directory["TS forecast"].map(fmt)
    directory["Breakout forecast"] = directory["Breakout forecast"].map(fmt)
    directory["BTC beta"] = directory["BTC beta"].map(fmt)
    directory["Median quote volume"] = directory["Median quote volume"].map(lambda x: money(x))
    directory["Account funding"] = directory["Account funding"].map(money)

    cards = []
    for symbol in symbols:
        row = snapshot.loc[symbol]
        component_rows = []
        for name, value in ts_components.loc[symbol].items():
            component_rows.append({"Family": "EWMAC", "Component": name, "Forecast": fmt(value)})
        for name, value in bo_components.loc[symbol].items():
            component_rows.append({"Family": "Breakout", "Component": name, "Forecast": fmt(value)})
        component_table = table_html(pd.DataFrame(component_rows), "compact")

        attr = attribution_primary[attribution_primary.symbol.eq(symbol)][
            ["model", "net_return_contribution", "funding_return", "turnover"]
        ].copy()
        attr["model"] = attr.model.map({TS_MODEL: "Pooled EWMAC", XS_MODEL: "Cross-sectional", BO_MODEL: "Pooled breakout"})
        attr.columns = ["Model", "Net contribution", "Funding return", "Turnover"]
        attr["Net contribution"] = attr["Net contribution"].map(pct)
        attr["Funding return"] = attr["Funding return"].map(pct)
        attr["Turnover"] = attr["Turnover"].map(lambda x: f"{x:,.1f}×")

        stats = signal_stats[signal_stats.symbol.eq(symbol)][
            ["model", "forecast_mean", "forecast_std", "mean_abs_forecast", "positive_share", "direction_flips", "valid_days"]
        ].copy()
        stats["model"] = stats.model.map({TS_MODEL: "Pooled EWMAC", XS_MODEL: "Cross-sectional", BO_MODEL: "Pooled breakout"})
        stats.columns = ["Model", "Mean", "Std", "Mean |forecast|", "Positive days", "Direction flips", "Valid days"]
        for column in ("Mean", "Std", "Mean |forecast|"):
            stats[column] = stats[column].map(fmt)
        stats["Positive days"] = stats["Positive days"].map(pct)

        ticker_actual = actual[actual.symbol.eq(symbol)]
        account_note = (f"{len(ticker_actual):,} account events · paid {money(ticker_actual.loc[ticker_actual.income_usd.lt(0), 'income_usd'].sum())} · "
                        f"received {money(ticker_actual.loc[ticker_actual.income_usd.gt(0), 'income_usd'].sum())} · net {money(ticker_actual.income_usd.sum())}")
        cards.append(
            f'<details class="ticker-card" data-symbol="{escape(symbol.lower(), quote=True)}" data-sector="{escape(str(row.sector).lower(), quote=True)}">'
            f'<summary><span><strong>{escape(symbol)}</strong><small>{escape(str(row.sector))}</small></span>'
            f'<span class="summary-metrics">TS {fmt(row.ts_forecast)} · BO {fmt(row.breakout_forecast)} · XS {pct(row.xs_cross_sectional_percentile)}</span></summary>'
            f'<div class="ticker-body"><div class="ticker-kpis">'
            f'<span><small>30d return</small><strong>{pct(row.return_30d)}</strong></span>'
            f'<span><small>90d volatility</small><strong>{pct(row.volatility_90d)}</strong></span>'
            f'<span><small>BTC beta</small><strong>{fmt(row.btc_beta_90d)}</strong></span>'
            f'<span><small>Median quote volume</small><strong>{money(row.liquidity)}</strong></span>'
            f'<span><small>Listing age</small><strong>{fmt(row.listing_age_days, 0)}d</strong></span>'
            f'<span><small>Last open</small><strong>{money(row.last_open)}</strong></span></div>'
            f'<div class="spark-wrap"><h4>Trailing price path</h4>{sparkline(opens[symbol])}</div>'
            f'<div class="ticker-grid"><section><h4>Current component forecasts</h4>{component_table}</section>'
            f'<section><h4>Post-training attribution</h4>{table_html(attr, "compact")}<p class="mini-note">{escape(account_note)}</p>'
            f'<p class="mini-note">Expected next-24h funding: EWMAC {pct(row.expected_funding_24h_ts, 3)} · XS {pct(row.expected_funding_24h_xs, 3)} · breakout {pct(row.expected_funding_24h_breakout, 3)}</p></section></div>'
            f'<section><h4>Historical signal behaviour</h4>{table_html(stats, "compact")}</section></div></details>'
        )

    strongest = chart_data.nlargest(1, "xs_cross_sectional_percentile").iloc[0]
    weakest = chart_data.nsmallest(1, "xs_cross_sectional_percentile").iloc[0]
    largest_funding = chart_data.loc[chart_data.actual_funding_net.fillna(0).abs().idxmax()]
    css = """
    <style>
    html,body,main,section,figure{max-width:100%;min-width:0}main{overflow-x:clip}figure{contain:inline-size}.chart-scroll{display:block;width:100%;max-width:100%;overflow-x:auto;contain:inline-size}.table-scroll{width:100%;max-width:100%;overflow-x:auto;overscroll-behavior-x:contain;contain:inline-size;border:1px solid var(--border);border-radius:8px}.table-scroll table{margin:0}
    .controls{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.controls input,.controls select{font:inherit;padding:10px 12px;border:1px solid var(--border);border-radius:8px;background:#fff;min-width:240px}
    .ticker-list{display:grid;gap:10px}.ticker-card{background:#fff;border:1px solid var(--border);border-radius:10px;overflow:hidden}.ticker-card summary{cursor:pointer;display:flex;justify-content:space-between;gap:18px;padding:16px 18px;align-items:center}.ticker-card summary strong{font-size:17px}.ticker-card summary small{display:block;color:var(--muted);margin-top:2px}.summary-metrics{font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums}.ticker-body{border-top:1px solid var(--border);padding:18px}.ticker-kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:9px;margin-bottom:15px}.ticker-kpis span{background:#f5f6f9;border-radius:8px;padding:10px}.ticker-kpis small{display:block;color:var(--muted);font-size:11px}.ticker-kpis strong{font-size:15px}.ticker-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.ticker-grid section,.ticker-body>section,.spark-wrap{margin:12px 0}.ticker-body h4{margin:0 0 8px}.spark{width:100%;height:92px;background:#ececf3;border-radius:8px}.mini-note{font-size:12px;color:var(--muted)}.compact{font-size:12px}.compact th,.compact td{padding:7px 9px}.hidden{display:none!important}
    @media(max-width:900px){.ticker-kpis{grid-template-columns:repeat(3,1fr)}.ticker-grid{grid-template-columns:1fr}.ticker-card summary{align-items:flex-start;flex-direction:column}.summary-metrics{white-space:normal}}
    @media(max-width:520px){.ticker-kpis{grid-template-columns:repeat(2,1fr)}.controls input,.controls select{min-width:100%;width:100%}}
    </style>
    """
    script = """
    <script>
    (()=>{const q=document.getElementById('ticker-search'),s=document.getElementById('sector-filter'),cards=[...document.querySelectorAll('.ticker-card')],count=document.getElementById('visible-count');
    function apply(){const text=q.value.trim().toLowerCase(),sector=s.value.toLowerCase();let n=0;cards.forEach(c=>{const show=(!text||c.dataset.symbol.includes(text))&&(!sector||c.dataset.sector===sector);c.classList.toggle('hidden',!show);if(show)n++;});count.textContent=n;}q.addEventListener('input',apply);s.addEventListener('change',apply);apply();})();
    </script>
    """
    sectors = sorted(str(x) for x in snapshot.sector.dropna().unique())
    body = hero(
        "ACAUSAL CAPITAL · TICKER ANALYTICS", "Every eligible contract,", "from component signal to funding cashflow.",
        "A searchable companion to the complete crypto momentum study, reconciling current signal strength with ticker-level history, risk, contribution and funding.",
        [f"Data cutoff {manifest['holdout_period'][1][:10]}", f"{len(symbols)} eligible tickers", "10 component signals per ticker", "Shadow research only"],
    )
    body += css
    body += '<nav><a href="#overview">Overview</a><a href="#directory">Directory</a><a href="#profiles">Ticker profiles</a><a href="#methods">Methods</a></nav>'
    body += '<section class="metrics">' + metric("Highest XS rank", strongest.symbol, pct(strongest.xs_cross_sectional_percentile)) + metric("Lowest XS rank", weakest.symbol, pct(weakest.xs_cross_sectional_percentile)) + metric("Largest account funding magnitude", largest_funding.symbol, money(largest_funding.actual_funding_net)) + metric("Ticker profiles", f"{len(symbols)}", "All eligible portfolio assets") + '</section>'
    body += '<aside class="callout warning"><strong>Attribution is not standalone ticker performance.</strong> Net contribution, funding and turnover are each ticker’s contribution inside the specified portfolio model at the common reference setting. Account funding includes all archived Binance account exposure and is not assumed to belong to this strategy.</aside>'
    body += '<section id="overview"><h2>Cross-ticker overview</h2>' + figure_html(fig_strength.to_html(full_html=False, include_plotlyjs=False), "Current signal agreement", "Bubble size is trailing Binance quote volume; colour is the cross-sectional percentile.", "../data_store/crypto_momentum_research/complete_results/ticker_analytics_snapshot.csv") + figure_html(fig_contribution.to_html(full_html=False, include_plotlyjs=False), "Historical model contribution", "Post-training cumulative net-return contribution at the 15% / 1× / 10% common reference setting.", "../data_store/crypto_momentum_research/complete_results/ticker_attribution.parquet") + figure_html(fig_funding.to_html(full_html=False, include_plotlyjs=False), "Actual account funding", "Paid is negative and received is positive; this is account history, not automatically model-attributed.", "../data_store/crypto_momentum_research/complete_results/actual_funding_reconciliation.parquet") + '</section>'
    body += '<section id="directory"><h2>All-ticker directory</h2><p>Current signals, market context and archived account funding. Use the profiles below for components and history.</p>' + table_html(directory) + '</section>'
    options = ''.join(f'<option value="{escape(x, quote=True)}">{escape(x)}</option>' for x in sectors)
    body += f'<section id="profiles"><h2>Individual ticker profiles</h2><div class="controls"><input id="ticker-search" type="search" placeholder="Search ticker…" aria-label="Search ticker"><select id="sector-filter" aria-label="Filter sector"><option value="">All sectors</option>{options}</select><span><strong id="visible-count">{len(symbols)}</strong> visible</span></div><div class="ticker-list">{"".join(cards)}</div></section>'
    body += '<section id="methods"><h2>Coverage and interpretation</h2><ul><li>Current forecasts, ranks and target positions come from the frozen daily signal ledger.</li><li>EWMAC components are 2/8, 4/16, 8/32, 16/64 and 32/128. Breakout components are 16, 32, 64, 128 and 256 days.</li><li>Signal histories preserve warm-up missing values and final forecasts are capped at ±20.</li><li>Contribution includes the model’s fees, 5 bps one-way slippage and event-level funding allocation at the common reference setting.</li><li>Expected funding applies the latest observed funding rate and is an estimate, not a realized cashflow.</li><li>The current-universe historical fallback creates survivor bias; this remains research evidence, not a production recommendation.</li></ul><h3>Downloads</h3><p><a href="../data_store/crypto_momentum_research/complete_results/ticker_analytics_snapshot.csv">Ticker analytics snapshot CSV</a> · <a href="../data_store/crypto_momentum_research/complete_results/daily_signal_records.parquet">Daily signal ledger</a> · <a href="../data_store/crypto_momentum_research/complete_results/ticker_attribution.parquet">Ticker attribution</a> · <a href="../data_store/crypto_momentum_research/complete_results/primary_ts_component_forecasts.parquet">EWMAC components</a> · <a href="../data_store/crypto_momentum_research/complete_results/primary_breakout_component_forecasts.parquet">Breakout components</a></p><h3>Rebuild</h3><pre>python scripts/build_crypto_ticker_analytics_report.py</pre></section>'
    body += script
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_page("Crypto Momentum Ticker Analytics", body, plotly=True, accent="blue"), encoding="utf-8")
    sources = [RESULTS / "latest_signal_records.csv", RESULTS / "daily_signal_records.parquet", RESULTS / "ticker_attribution.parquet",
               RESULTS / "primary_ts_component_forecasts.parquet", RESULTS / "primary_breakout_component_forecasts.parquet",
               RESULTS / "actual_funding_reconciliation.parquet", INPUTS / "portfolio_open_prices.parquet"]
    report_manifest = {
        "report": str(OUTPUT), "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "study_generated_at": manifest["generated_at"], "generator": "scripts/build_crypto_ticker_analytics_report.py",
        "rebuild": "python scripts/build_crypto_ticker_analytics_report.py",
        "inputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    }
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "tickers": len(symbols), "snapshot": str(SNAPSHOT)}, indent=2))


if __name__ == "__main__":
    main()
