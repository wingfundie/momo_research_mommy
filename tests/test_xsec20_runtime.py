import numpy as np
import pandas as pd

from momo_bot.chart_theme import (
    CANVAS,
    TEAL,
    performance_figures,
    portfolio_figure,
    risk_figure,
    signal_distribution_figure,
)
from momo_bot.xsec20 import (
    CONFIG_ID,
    GROSS_CAP,
    MODEL,
    PORTFOLIO_VALUE,
    SLIPPAGE_BPS,
    TARGET_VOL,
    TICKER_RISK_CAP,
    VOLATILITY_WINDOW,
    annual_metrics,
    compare_returns,
    normalize_symbol,
    round_quantity,
)
from momo_bot.xsec_telegram import (
    _performance_args,
    _portfolio_message,
    lookback_token,
    parse_lookback,
    result_count,
    wants_legacy,
)


def test_pinned_xsec20_contract() -> None:
    assert MODEL == "xsm_ic20_dollar_neutral_vol60"
    assert CONFIG_ID == "cc09fe90ed5150b8a67f"
    assert TARGET_VOL == 0.15
    assert GROSS_CAP == 2.0
    assert TICKER_RISK_CAP == 0.25
    assert VOLATILITY_WINDOW == 60
    assert PORTFOLIO_VALUE == 100_000.0
    assert SLIPPAGE_BPS == 5.0


def test_symbol_and_command_argument_compatibility() -> None:
    assert normalize_symbol("sol") == "SOLUSDT"
    assert normalize_symbol("btc/usdt") == "BTCUSDT"
    assert normalize_symbol("eth-usdt") == "ETHUSDT"
    assert wants_legacy(["10", "legacy"])
    assert not wants_legacy(["10"])
    assert result_count(["50"]) == 25
    assert result_count(["invalid"], default=7) == 7
    assert _performance_args(["90d", "btc"]) == ("90d", 90, "BTCUSDT")
    assert _performance_args(["ETH", "all"]) == ("all", None, "ETHUSDT")


def test_flexible_lookback_parser() -> None:
    assert lookback_token("45d") == ("45d", 45)
    assert lookback_token("12w") == ("12w", 84)
    assert lookback_token("6m") == ("6m", 180)
    assert lookback_token("2y") == ("2y", 730)
    assert lookback_token("all") == ("all", None)
    assert lookback_token("0d") is None
    assert lookback_token("11y") is None
    assert parse_lookback(["BTC", "6m"], default="1y") == ("6m", 180)
    assert parse_lookback(["BTC"], default="1y") == ("1y", 365)


def test_exchange_quantity_rounding_never_increases_risk() -> None:
    assert round_quantity(1.239, step_size=0.01, min_qty=0.01) == 1.23
    assert round_quantity(-1.239, step_size=0.01, min_qty=0.01) == -1.23
    assert round_quantity(0.005, step_size=0.01, min_qty=0.01) == 0.0
    assert round_quantity(np.nan, step_size=0.01) == 0.0


def test_portfolio_output_is_one_message_with_ticker_strength_and_sharpe() -> None:
    active = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "SOLUSDT", "PAXGUSDT"],
            "xsec_side": ["LONG", "LONG", "SHORT"],
            "absolute_forecast": [15.25, 17.80, 0.62],
            "rank": [0.80, 0.95, 0.05],
            "standalone_sr": [1.25, 1.80, 0.40],
            "target_weight": [0.02, 0.04, -0.03],
            "target_notional": [2_000.0, 4_000.0, -3_000.0],
            "quantity": [0.02, 30.0, -0.7],
        }
    )
    state = pd.Series(
        {
            "long_exposure": 0.06,
            "short_exposure": -0.03,
            "gross_exposure": 0.09,
            "net_exposure": 0.03,
        }
    )

    message = _portfolio_message(active, state)

    assert message.count("<pre>") == 1
    assert "LONGS 2 | $6.00k" in message
    assert "SHORTS 1 | $3.00k" in message
    assert "SOL     4.00     30 | +17.8  95 +1.8" in message
    assert "PAXG    3.00    0.7 |  +0.6   5 +0.4" in message
    assert "Str −20…20" in message
    assert "Rk percentile" in message
    assert len(message) <= 4096


def test_return_comparison_aligns_periods_and_reports_beta() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    portfolio = pd.Series([0.01, 0.02, -0.01, 0.00, 0.03], index=index)
    benchmark = pd.Series([0.005, 0.01, -0.005, 0.00], index=index[1:])

    aligned, comparison = compare_returns(portfolio, benchmark)

    assert list(aligned.index) == list(index[1:])
    assert comparison["observations"] == 4
    assert np.isfinite(comparison["beta"])
    assert comparison["portfolio"] == annual_metrics(aligned["portfolio"])


def test_reference_chart_theme_and_calendar_axes() -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    portfolio = pd.DataFrame(
        {
            "long_exposure": np.linspace(0.5, 0.7, 10),
            "short_exposure": np.linspace(-0.5, -0.4, 10),
            "gross_exposure": np.linspace(1.0, 1.1, 10),
            "net_exposure": np.linspace(0.0, 0.3, 10),
        },
        index=dates,
    )
    risk = risk_figure(portfolio)

    assert risk.layout.paper_bgcolor == CANVAS
    assert risk.layout.xaxis.type == "date"
    assert risk.layout.xaxis.rangeslider.visible is True
    assert {trace.line.color for trace in risk.data} >= {TEAL}
    assert any(annotation.text == "<b>RISK</b>" for annotation in risk.layout.annotations)
    assert len(risk.layout.shapes) == 2  # USD/Share segmented control

    figures = performance_figures(pd.Series(np.full(10, 0.001), index=dates))
    assert all(figure.layout.xaxis.type == "date" for figure in figures)
    assert all(str(figure.data[0].x[0]).startswith("2026-01-01") for figure in figures)


def test_portfolio_chart_limits_dense_universe_to_largest_exposures() -> None:
    frame = pd.DataFrame(
        {
            "symbol": [f"COIN{i}USDT" for i in range(40)],
            "target_weight": np.linspace(-0.2, 0.2, 40),
            "target_notional": np.linspace(-20_000, 20_000, 40),
        }
    )
    figure = portfolio_figure(frame)
    assert len(figure.data[0].x) == 30


def test_signal_distribution_chart_uses_forecast_paths_and_percentiles() -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    panel = pd.DataFrame(
        {
            "BTCUSDT": np.linspace(-10, 10, 20),
            "ETHUSDT": np.linspace(-5, 15, 20),
            "SOLUSDT": np.linspace(-15, 5, 20),
        },
        index=dates,
    )

    figure = signal_distribution_figure(panel, period_label="30d")

    assert len(figure.data) == 6  # three ticker paths plus P25, P75 and mean
    assert [trace.name for trace in figure.data[-3:]] == [
        "75th percentile",
        "25th percentile",
        "Average signal",
    ]
    assert figure.layout.xaxis.type == "date"
    assert figure.layout.xaxis.rangeslider.visible is True
    assert tuple(figure.layout.yaxis.range) == (-20.5, 20.5)
