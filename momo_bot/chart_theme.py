from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots


CANVAS = "#10131C"
CARD_BORDER = "#26313D"
TEXT = "#F3F5F7"
MUTED = "#7D899B"
GRID = "rgba(125,137,155,0.18)"
TEAL = "#57C9C9"
BLUE = "#5F86C8"
GOLD = "#D6A64F"
PURPLE = "#8C6BC6"
CORAL = "#DE7D83"
GREEN = "#6DBB92"
PINK = "#B96FA7"
SLATE = "#647584"
PALETTE = (TEAL, BLUE, GOLD, PURPLE, CORAL, GREEN, PINK, SLATE)
FONT = "Inter, Segoe UI, Arial, sans-serif"
WIDTH = 1600
HEIGHT = 900


def _plot_dates(index: pd.Index) -> list[str]:
    """Serialize dates explicitly for stable Plotly/Kaleido output across pandas versions."""
    return [pd.Timestamp(value).isoformat() for value in index]


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def apply_chart_theme(
    fig: go.Figure,
    *,
    eyebrow: str,
    title: str,
    unit: str | None = None,
    range_slider: bool = False,
    date_axis: bool = False,
    height: int = HEIGHT,
) -> go.Figure:
    fig.update_layout(
        width=WIDTH,
        height=height,
        autosize=False,
        paper_bgcolor=CANVAS,
        plot_bgcolor=CANVAS,
        font=dict(family=FONT, color=MUTED, size=16),
        colorway=list(PALETTE),
        margin=dict(l=112, r=54, t=230, b=105 if range_slider else 75),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.12,
            yanchor="bottom",
            font=dict(size=15, color="#9CA8B8"),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        annotations=list(fig.layout.annotations or ())
        + [
            dict(
                x=0,
                y=1.35,
                xref="paper",
                yref="paper",
                text=f"<b>{eyebrow.upper()}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(family=FONT, size=14, color="#62D2D0"),
            ),
            dict(
                x=0,
                y=1.275,
                xref="paper",
                yref="paper",
                text=f"<b>{title}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(family=FONT, size=28, color=TEXT),
            ),
        ],
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(color=MUTED, size=15),
        title_font=dict(color=MUTED, size=15),
        automargin=True,
        rangeslider=(
            dict(
                visible=True,
                thickness=0.075,
                bgcolor="#1A2632",
                bordercolor="#2A3542",
                borderwidth=1,
            )
            if range_slider
            else dict(visible=False)
        ),
        type="date" if date_axis else None,
        tickformat="%Y-%m-%d" if date_axis else None,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickfont=dict(color=MUTED, size=15),
        title_font=dict(color=MUTED, size=15),
        automargin=True,
    )
    if unit:
        left_active = unit.lower() == "usd"
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0.855,
            x1=0.995,
            y0=1.255,
            y1=1.39,
            line=dict(color="#2A3542", width=1),
            fillcolor="#17212D",
            layer="above",
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.92,
            x1=0.92,
            y0=1.255,
            y1=1.39,
            line=dict(color="#2A3542", width=1),
            layer="above",
        )
        fig.add_annotation(
            x=0.887,
            y=1.322,
            xref="paper",
            yref="paper",
            text="<b>USD</b>" if left_active else "USD",
            showarrow=False,
            font=dict(color=TEXT if left_active else MUTED, size=14),
        )
        fig.add_annotation(
            x=0.958,
            y=1.322,
            xref="paper",
            yref="paper",
            text="<b>Share</b>" if not left_active else "Share",
            showarrow=False,
            font=dict(color=TEXT if not left_active else MUTED, size=14),
        )
    return fig


def export_chart(fig: go.Figure, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(target), width=WIDTH, height=int(fig.layout.height or HEIGHT), scale=1)
    # Plotly cannot draw rounded paper corners. Add the thin rounded frame in a
    # post-process so static Telegram cards match the dashboard reference.
    with Image.open(target).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (8, 8, image.width - 9, image.height - 9),
            radius=24,
            outline=CARD_BORDER,
            width=2,
        )
        image.save(target, quality=95)
    return target


def portfolio_figure(ticker_state: pd.DataFrame, *, unit: str = "share") -> go.Figure:
    frame = ticker_state.copy()
    value_col = "target_notional" if unit.lower() == "usd" else "target_weight"
    frame = frame.loc[frame[value_col].abs().gt(1e-10)].copy()
    # Keep the Telegram card legible while the accompanying table retains the
    # complete portfolio. Select the largest exposures, then sort them for a
    # clean horizontal bar chart.
    frame = frame.reindex(frame[value_col].abs().nlargest(30).index)
    frame = frame.reindex(frame[value_col].sort_values().index)
    values = frame[value_col]
    colors = np.where(values.ge(0), TEAL, CORAL)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=frame["symbol"].str.removesuffix("USDT"),
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}<br>%{x}<extra></extra>",
            name="Target",
        )
    )
    apply_chart_theme(fig, eyebrow="Composition", title="Current target", unit=unit)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickformat="$,.0f" if unit.lower() == "usd" else ".1%")
    return fig


def risk_figure(portfolio: pd.DataFrame, *, unit: str = "share") -> go.Figure:
    scale = 100_000.0 if unit.lower() == "usd" else 1.0
    fig = go.Figure()
    for name, column, color in (
        ("Long", "long_exposure", TEAL),
        ("Short", "short_exposure", CORAL),
        ("Gross", "gross_exposure", BLUE),
        ("Net", "net_exposure", GOLD),
    ):
        fig.add_trace(
            go.Scatter(
                x=_plot_dates(portfolio.index),
                y=portfolio[column] * scale,
                name=name,
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy" if name in {"Long", "Short"} else None,
                fillcolor=_rgba(color, 0.16),
            )
        )
    apply_chart_theme(
        fig,
        eyebrow="Risk",
        title="Portfolio exposure",
        unit=unit,
        range_slider=True,
        date_axis=True,
    )
    fig.update_yaxes(tickformat="$,.0f" if unit.lower() == "usd" else ".0%")
    return fig


def performance_figures(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    benchmark_name: str | None = None,
) -> list[go.Figure]:
    series = {"Model portfolio": portfolio_returns.dropna()}
    if benchmark_returns is not None:
        aligned = pd.concat([portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1).dropna()
        series = {
            "Model portfolio": aligned["portfolio"],
            benchmark_name or "Benchmark": aligned["benchmark"],
        }
    growth = {name: 100_000.0 * (1.0 + values).cumprod() for name, values in series.items()}
    fig_growth = go.Figure()
    for (name, values), color in zip(growth.items(), (TEAL, BLUE)):
        fig_growth.add_trace(
            go.Scatter(
                x=_plot_dates(values.index),
                y=values,
                name=name,
                mode="lines",
                line=dict(color=color, width=2.2),
            )
        )
    apply_chart_theme(
        fig_growth,
        eyebrow="Performance",
        title=f"Portfolio vs {benchmark_name}" if benchmark_name else "Model portfolio",
        range_slider=True,
        date_axis=True,
    )
    fig_growth.update_yaxes(tickprefix="$", tickformat=",.0f")

    fig_dd = go.Figure()
    for (name, values), color in zip(growth.items(), (TEAL, CORAL)):
        drawdown = values.div(values.cummax()).sub(1.0)
        fig_dd.add_trace(
            go.Scatter(
                x=_plot_dates(drawdown.index),
                y=drawdown,
                name=name,
                mode="lines",
                line=dict(color=color, width=1.8),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.28),
            )
        )
    apply_chart_theme(
        fig_dd,
        eyebrow="Performance",
        title="Drawdown",
        range_slider=True,
        date_axis=True,
    )
    fig_dd.update_yaxes(tickformat=".0%")
    figures = [fig_growth, fig_dd]

    if len(growth) == 2:
        names = list(growth)
        relative = growth[names[0]].div(growth[names[1]]).dropna()
        fig_relative = go.Figure(
            go.Scatter(
                x=_plot_dates(relative.index),
                y=relative,
                name=f"{names[0]} / {names[1]}",
                mode="lines",
                line=dict(color=GOLD, width=2.2),
                fill="tozeroy",
                fillcolor=_rgba(GOLD, 0.12),
            )
        )
        apply_chart_theme(
            fig_relative,
            eyebrow="Performance",
            title="Relative wealth",
            range_slider=True,
            date_axis=True,
        )
        figures.append(fig_relative)
    return figures


def changes_figure(changes: pd.DataFrame) -> go.Figure:
    frame = changes.copy().sort_values("target_change", key=lambda s: s.abs())
    fig = go.Figure(
        go.Bar(
            x=frame["target_change"],
            y=frame["symbol"].str.removesuffix("USDT"),
            orientation="h",
            marker=dict(color=np.where(frame["target_change"].ge(0), TEAL, CORAL)),
            name="Target change",
        )
    )
    apply_chart_theme(fig, eyebrow="Signals", title="Largest changes")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickformat=".1%")
    return fig


def signal_distribution_figure(
    forecasts: pd.DataFrame,
    *,
    period_label: str,
) -> go.Figure:
    panel = forecasts.replace([np.inf, -np.inf], np.nan).sort_index()
    panel = panel.loc[panel.notna().sum(axis=1).gt(0)]
    if panel.empty:
        raise ValueError("No cross-sectional forecast history is available")

    q25 = panel.quantile(0.25, axis=1)
    q75 = panel.quantile(0.75, axis=1)
    average = panel.mean(axis=1)
    dates = _plot_dates(panel.index)
    fig = go.Figure()

    for symbol in panel.columns:
        values = panel[symbol]
        if not values.notna().any():
            continue
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines",
                line=dict(color="rgba(156,168,184,0.16)", width=0.8),
                showlegend=False,
                hoverinfo="skip",
                name=symbol.removesuffix("USDT"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=q75,
            name="75th percentile",
            mode="lines",
            line=dict(color="#7898E8", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=q25,
            name="25th percentile",
            mode="lines",
            line=dict(color=BLUE, width=2.4),
            fill="tonexty",
            fillcolor=_rgba(BLUE, 0.08),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=average,
            name="Average signal",
            mode="lines",
            line=dict(color=CORAL, width=3.0),
        )
    )
    apply_chart_theme(
        fig,
        eyebrow="Signals",
        title=f"Signal distribution · {period_label}",
        range_slider=True,
        date_axis=True,
    )
    fig.add_hline(
        y=0,
        line=dict(color="rgba(243,245,247,0.68)", width=1.4, dash="dash"),
    )
    fig.update_layout(legend_traceorder="normal")
    fig.update_yaxes(range=[-20.5, 20.5], dtick=5, title_text="Absolute forecast")
    return fig


def ticker_figures(history: pd.DataFrame, symbol: str) -> list[go.Figure]:
    label = symbol.removesuffix("USDT")
    first_price = history["price"].first_valid_index()
    if first_price is None:
        raise ValueError(f"No price history is available for {symbol}")
    history = history.loc[first_price:].copy()
    standalone = 100_000 * (1.0 + history["standalone_return"].fillna(0.0)).cumprod()
    buy_hold = 100_000 * history["price"].div(history["price"].dropna().iloc[0])
    returns_fig = go.Figure()
    returns_fig.add_trace(
        go.Scatter(
            x=_plot_dates(standalone.index),
            y=standalone,
            name="Standalone trend",
            line=dict(color=TEAL, width=2.2),
        )
    )
    returns_fig.add_trace(
        go.Scatter(
            x=_plot_dates(buy_hold.index),
            y=buy_hold,
            name=f"{label} buy-and-hold",
            line=dict(color=BLUE, width=2.0),
        )
    )
    apply_chart_theme(
        returns_fig,
        eyebrow="Ticker analytics",
        title=f"{label} returns",
        range_slider=True,
        date_axis=True,
    )
    returns_fig.update_yaxes(tickprefix="$", tickformat=",.0f")

    signal_fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_dates = _plot_dates(history.index)
    signal_fig.add_trace(go.Scatter(x=plot_dates, y=history["absolute_forecast"], name="Absolute forecast", line=dict(color=TEAL, width=2)), secondary_y=False)
    signal_fig.add_trace(go.Scatter(x=plot_dates, y=history["price"], name="Price", line=dict(color=BLUE, width=1.8)), secondary_y=True)
    apply_chart_theme(
        signal_fig,
        eyebrow="Ticker analytics",
        title=f"{label} price and forecast",
        range_slider=True,
        date_axis=True,
    )
    signal_fig.update_yaxes(title_text="Forecast", secondary_y=False)
    signal_fig.update_yaxes(title_text="Price", secondary_y=True, showgrid=False)

    rank_fig = go.Figure()
    rank_fig.add_trace(go.Scatter(x=plot_dates, y=history["rank"], name="XSec percentile", line=dict(color=PURPLE, width=2)))
    rank_fig.add_trace(go.Scatter(x=plot_dates, y=history["held_weight"], name="Held weight", line=dict(color=GOLD, width=2), yaxis="y2"))
    apply_chart_theme(
        rank_fig,
        eyebrow="Ticker analytics",
        title=f"{label} rank and portfolio weight",
        range_slider=True,
        date_axis=True,
    )
    rank_fig.update_yaxes(range=[0, 1], tickformat=".0%")
    rank_fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".1%", showgrid=False, color=MUTED))
    return [returns_fig, signal_fig, rank_fig]


def export_figures(figures: Iterable[go.Figure], directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, fig in enumerate(figures, 1):
        paths.append(export_chart(fig, directory / f"{stem}_{number}.png"))
    return paths
