from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html, dash_table

from momo_bot.research.ledger import ResearchLedger


LEDGER_PATH = Path(os.getenv("MOMO_RESEARCH_LEDGER", "data_store/crypto_momentum_research.sqlite"))
BG, PANEL, BORDER, TEXT, MUTED, CYAN, GREEN, RED = (
    "#060A12", "#0D1421", "#243047", "#E6EDF7", "#8FA2BC", "#39D0FF", "#4ADE80", "#FB7185"
)


def _load() -> dict[str, pd.DataFrame]:
    if not LEDGER_PATH.exists():
        return {name: pd.DataFrame() for name in ("signals", "runs", "daily", "funding", "actual")}
    with ResearchLedger(LEDGER_PATH) as ledger:
        return {"signals": ledger.read("signal_records"), "runs": ledger.read("portfolio_runs"),
                "daily": ledger.read("daily_portfolio"), "funding": ledger.read("funding_events"),
                "actual": ledger.read("actual_funding")}


def _figure(frame, x, y, title, color=None):
    if frame.empty:
        fig = px.line(title=f"{title} — no stored data")
    else:
        fig = px.line(frame, x=x, y=y, color=color, title=title)
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT, margin=dict(l=35, r=20, t=50, b=35),
                      xaxis_gridcolor=BORDER, yaxis_gridcolor=BORDER)
    return fig


def _card(label, value, color=TEXT):
    return html.Div([html.Div(label, style={"color": MUTED, "fontSize": "12px"}),
                     html.Div(value, style={"color": color, "fontSize": "24px", "fontWeight": 700})],
                    style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "8px", "padding": "14px"})


app = Dash(__name__, title="Crypto Momentum Research")
data = _load()
latest = data["signals"].sort_values("timestamp").groupby("symbol").tail(1) if not data["signals"].empty else pd.DataFrame()
funding_net = data["funding"]["funding_cashflow_usd"].sum() if not data["funding"].empty else 0.0
best_sharpe = float("nan")
if not data["runs"].empty:
    best_sharpe = max((json.loads(item).get("net_sharpe", float("nan")) for item in data["runs"]["metrics_json"]), default=float("nan"))

app.layout = html.Div([
    dcc.Interval(id="refresh", interval=60_000),
    html.Div([html.Div([html.H1("Crypto Momentum Research", style={"margin": 0, "fontSize": "22px"}),
                       html.Div("Causal shadow pipeline · production signals unchanged", style={"color": MUTED})]),
              html.Div("SHADOW", style={"color": CYAN, "border": f"1px solid {CYAN}", "padding": "6px 10px", "borderRadius": "20px"})],
             style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"}),
    html.Div([_card("Tracked tickers", str(len(latest))), _card("Best exploratory Sharpe", f"{best_sharpe:.2f}"),
              _card("Modeled net funding", f"${funding_net:,.2f}", GREEN if funding_net >= 0 else RED),
              _card("Completed runs", str(len(data["runs"])))],
             style={"display": "grid", "gridTemplateColumns": "repeat(4,minmax(0,1fr))", "gap": "12px"}),
    dcc.Tabs([
        dcc.Tab(label="Portfolio", children=[dcc.Graph(figure=_figure(data["daily"], "timestamp", "equity", "Portfolio equity", "run_id"))]),
        dcc.Tab(label="Ticker Strength", children=[dash_table.DataTable(
            data=latest[[c for c in ["symbol", "combined_forecast", "cross_sectional_percentile", "direction", "target_notional"] if c in latest]].to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in ["symbol", "combined_forecast", "cross_sectional_percentile", "direction", "target_notional"] if c in latest],
            sort_action="native", filter_action="native", style_header={"backgroundColor": PANEL, "color": TEXT},
            style_cell={"backgroundColor": BG, "color": TEXT, "border": f"1px solid {BORDER}", "padding": "8px"})]),
        dcc.Tab(label="Funding", children=[dcc.Graph(figure=_figure(data["funding"], "funding_time", "funding_cashflow_usd", "Funding paid, received and net", "symbol"))]),
        dcc.Tab(label="Models & Risk Grid", children=[dash_table.DataTable(
            data=data["runs"].to_dict("records"), columns=[{"name": c, "id": c} for c in data["runs"].columns],
            sort_action="native", filter_action="native", page_size=20,
            style_header={"backgroundColor": PANEL, "color": TEXT}, style_cell={"backgroundColor": BG, "color": TEXT, "border": f"1px solid {BORDER}", "maxWidth": 260, "overflow": "hidden"})]),
    ], colors={"border": BORDER, "primary": CYAN, "background": PANEL}),
], style={"background": BG, "color": TEXT, "minHeight": "100vh", "padding": "20px", "fontFamily": "Inter,Segoe UI,sans-serif"})


if __name__ == "__main__":
    app.run(debug=False, host=os.getenv("MOMO_DASH_HOST", "127.0.0.1"), port=int(os.getenv("MOMO_DASH_PORT", "8060")))
