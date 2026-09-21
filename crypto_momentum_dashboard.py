from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html, dash_table

RESULTS = Path(os.getenv("MOMO_COMPLETE_RESULTS", "data_store/crypto_momentum_research/complete_results"))
BG, PANEL, BORDER, TEXT, MUTED, CYAN, GREEN, RED = (
    "#06080f", "#111827", "#1e2538", "#f1f5f9", "#94a3b8", "#22d3ee", "#22c55e", "#ef4444"
)


def load_data():
    if not RESULTS.exists():
        return {}
    return {
        "configs": pd.read_parquet(RESULTS / "tested_configurations.parquet"),
        "returns": pd.read_parquet(RESULTS / "default_daily_returns.parquet"),
        "funding": pd.read_parquet(RESULTS / "default_daily_funding.parquet"),
        "funding_summary": pd.read_csv(RESULTS / "modeled_funding_summary.csv"),
        "signals": pd.read_csv(RESULTS / "latest_signal_records.csv"),
        "groups": pd.read_parquet(RESULTS / "group_attribution.parquet"),
        "ticker": pd.read_parquet(RESULTS / "ticker_attribution.parquet"),
        "capacity": pd.read_csv(RESULTS / "capacity_analysis.csv"),
        "actual": pd.read_parquet(RESULTS / "actual_funding_reconciliation.parquet"),
        "expected": pd.read_csv(RESULTS / "expected_funding_snapshot.csv"),
        "manifest": json.loads((RESULTS / "study_manifest.json").read_text(encoding="utf-8")),
    }


def style_figure(fig, title):
    fig.update_layout(template="plotly_dark", title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Plus Jakarta Sans, system-ui", "color": MUTED, "size": 11},
                      colorway=[CYAN, "#f97316", "#a855f7", "#3b82f6", GREEN, RED, "#eab308"],
                      margin={"l": 48, "r": 16, "t": 50, "b": 40}, hoverlabel={"bgcolor": "#1e293b"})
    fig.update_xaxes(gridcolor="#1a2035", linecolor=BORDER)
    fig.update_yaxes(gridcolor="#1a2035", linecolor=BORDER)
    return fig


def card(title, child):
    return html.Div([html.Div(title, className="dash-card-title"), child], className="dash-card")


def stat(label, value, tone=TEXT):
    return html.Div([html.Div(label, className="stat-label"), html.Div(value, className="stat-value", style={"color": tone})], className="stat-card")


def table(frame, page_size=20):
    view = frame.copy().replace({pd.NA: None})
    return dash_table.DataTable(data=view.to_dict("records"), columns=[{"name": c.replace("_", " ").title(), "id": c} for c in view.columns],
                                sort_action="native", filter_action="native", page_size=page_size,
                                style_table={"overflowX": "auto"}, style_header={"backgroundColor": PANEL, "color": TEXT, "fontWeight": 700},
                                style_cell={"backgroundColor": BG, "color": TEXT, "border": f"1px solid {BORDER}", "padding": "9px 12px",
                                            "fontFamily": "JetBrains Mono, monospace", "fontSize": 12, "minWidth": 95, "maxWidth": 260, "overflow": "hidden"})


data = load_data()
app = Dash(__name__, title="Crypto Momentum Research")
if not data:
    app.layout = html.Div("Run scripts/execute_complete_crypto_study.py first.", style={"padding": 30})
else:
    configs = data["configs"]
    headline = configs[(configs.taker_share.fillna(1).eq(1)) & configs.slippage_bps.fillna(5).eq(5) & configs.activation.fillna("next_open").eq("next_open")]
    models = sorted(data["signals"].model.unique())
    selected = json.loads((RESULTS / "selected_configurations.json").read_text(encoding="utf-8"))
    selected_table = pd.DataFrame([{"family": row["config"]["family"], "model": row["config"]["model"],
                                    "validation_sharpe": row["metrics"]["validation"]["net_sharpe"],
                                    "holdout_sharpe": row["metrics"]["holdout_2026"]["net_sharpe"],
                                    "holdout_return": row["metrics"]["holdout_2026"]["annual_return"]} for row in selected])
    equity_models = [c for c in ["ts_equal_vol90", "ts_shrink_80_primary_quarterly_vol90", "xs_ic5_20_primary_dollar_neutral",
                                 "breakout_shrink_80_primary_quarterly_vol90", "breakout_equal_vol90",
                                 "breakout_legacy_optuna", "baseline_btc_buy_hold", "baseline_equal_basket"] if c in data["returns"]]
    equity = (1 + data["returns"].loc[data["returns"].index >= "2024-01-01", equity_models].fillna(0)).cumprod()
    equity_long = equity.rename_axis("date").reset_index().melt("date", var_name="model", value_name="growth")
    equity_fig = style_figure(px.line(equity_long, x="date", y="growth", color="model"), "Common-risk portfolio growth")
    best_xs = selected_table.loc[selected_table.family.eq("cross_sectional")].iloc[0]
    actual_net = data["actual"].income_usd.sum()
    funding_chart = data["funding_summary"][data["funding_summary"].period.eq("ALL")].groupby("model")[["paid", "received", "net"]].sum().reset_index().melt("model", var_name="measure", value_name="funding_return")

    app.layout = html.Div(className="app-shell", children=[
        html.Aside(className="sidebar", children=[html.Div([html.Div("CA", className="brand-mark"), html.Div([html.Strong("Acausal"), html.Small("Momentum Research")])], className="brand-block"),
                                                   html.Div("SHADOW MODE", className="live-badge"), html.P("Production signals unchanged.", className="muted")]),
        html.Main(className="content", children=[
            html.Div(className="page-header", children=[html.Div([html.H1("Crypto Momentum Research"), html.P("Time-series, cross-sectional, breakout and funding diagnostics")]),
                                                        html.Div(f"Cutoff {data['manifest']['holdout_period'][1]}", className="live-badge")]),
            html.Div(className="kpi-row", children=[stat("Saved configurations", f"{len(configs):,}"), stat("Portfolio assets", str(data["manifest"]["portfolio_assets"])),
                                                    stat("Selected XS holdout Sharpe", f"{best_xs.holdout_sharpe:.2f}", GREEN if best_xs.holdout_sharpe > 0 else RED),
                                                    stat("Actual net funding", f"${actual_net:,.2f}", GREEN if actual_net >= 0 else RED)]),
            dcc.Tabs(className="tab-bar", children=[
                dcc.Tab(label="Portfolio", children=[html.Div(className="page-section grid-2", children=[card("Portfolio growth", dcc.Graph(figure=equity_fig)), card("Validation-selected configurations", table(selected_table, 10))])]),
                dcc.Tab(label="Ticker Strength", children=[html.Div(className="controls-card", children=[html.Label("Model", className="field-label"), dcc.Dropdown(models, "ts_shrink_80_primary_quarterly_vol90", id="signal-model", clearable=False)]), html.Div(id="signal-view")]),
                dcc.Tab(label="Risk Grid", children=[card("All headline configurations", table(headline[["family", "model", "target_vol", "gross_cap", "ticker_risk_cap", "rebalance", "validation_net_sharpe", "holdout_2026_net_sharpe", "validation_max_drawdown"]], 25))]),
                dcc.Tab(label="Funding", children=[html.Div(className="grid-2", children=[card("Modeled funding by model", dcc.Graph(figure=style_figure(px.bar(funding_chart, x="model", y="funding_return", color="measure", barmode="group"), "Funding paid, received and net"))),
                                                                                              card("Actual account funding", table(data["actual"][["timestamp", "symbol", "income_usd", "match_status"]].sort_values("timestamp", ascending=False), 25))]),
                                                          card("Expected funding snapshot", table(data["expected"][["model", "symbol", "position_weight", "reference_funding_rate", "next_settlement_expected_return", "next_24h_expected_return"]], 25))]),
                dcc.Tab(label="Cohorts & Capacity", children=[html.Div(className="grid-2", children=[card("Group attribution", table(data["groups"], 25)), card("Capacity screens", table(data["capacity"], 25))])]),
            ])
        ])
    ])

    @app.callback(Output("signal-view", "children"), Input("signal-model", "value"))
    def update_signals(model):
        frame = data["signals"][data["signals"].model.eq(model)].sort_values("forecast", ascending=False)
        fig = style_figure(px.bar(pd.concat([frame.head(12), frame.tail(12)]).sort_values("forecast"), x="forecast", y="symbol", orientation="h", color="forecast", color_continuous_scale="RdYlGn"), "Strongest and weakest tickers")
        return html.Div(className="grid-2", children=[card("Signal strength", dcc.Graph(figure=fig)), card("Ticker detail", table(frame[["symbol", "forecast", "cross_sectional_percentile", "direction", "target_position", "quality_flag"]], 25))])


if __name__ == "__main__":
    app.run(debug=False, host=os.getenv("MOMO_DASH_HOST", "127.0.0.1"), port=int(os.getenv("MOMO_DASH_PORT", "8060")))
