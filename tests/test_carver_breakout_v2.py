import numpy as np
import pandas as pd

from momo_bot.research.config import BreakoutSignalConfig
from momo_bot.research.signals import breakout_forecast
from momo_bot.strategies.forecasts import calc_breakout_forecast
from scripts.execute_carver_breakout_v2 import (
    FIXED_SCALARS,
    HORIZONS,
    combine_components,
    deterministic_select,
    eligibility_path,
)


def test_carver5_configuration_excludes_320():
    config = BreakoutSignalConfig()
    assert config.horizons == (10, 20, 40, 80, 160)
    assert 320 not in config.horizons
    assert 320 not in dict(config.forecast_scalars)


def test_breakout_uses_complete_window_then_h_over_four_smoothing():
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    price = pd.Series([10, 11, 9, 12, 8, 13, 7, 14, 6, 15, 5, 16, 4, 17, 3, 18, 2, 19, 1, 20], index=index, dtype=float)
    horizon = 8
    high = price.rolling(horizon, min_periods=horizon).max()
    low = price.rolling(horizon, min_periods=horizon).min()
    expected_raw = 40 * (price - (high + low) / 2) / (high - low)
    expected = expected_raw.ewm(span=2, adjust=False).mean()
    actual = calc_breakout_forecast(price, horizon)
    pd.testing.assert_series_equal(actual, expected)
    assert actual.iloc[: horizon - 1].isna().all()


def test_research_and_shared_breakout_paths_are_identical():
    price = pd.Series(np.linspace(90, 120, 100), index=pd.date_range("2024-01-01", periods=100, freq="D"))
    pd.testing.assert_series_equal(breakout_forecast(price, 20), calc_breakout_forecast(price, 20).clip(-20, 20))


def test_future_price_cannot_change_earlier_breakout_forecast():
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    original = pd.Series(100 + np.cumsum(np.sin(np.arange(120) / 5)), index=index)
    changed = original.copy()
    changed.iloc[100:] = changed.iloc[100:] * 10
    left = calc_breakout_forecast(original, 20)
    right = calc_breakout_forecast(changed, 20)
    pd.testing.assert_series_equal(left.iloc[:100], right.iloc[:100])


def test_fixed_scalars_match_requested_carver5_values():
    assert FIXED_SCALARS == {10: 0.60, 20: 0.67, 40: 0.70, 80: 0.73, 160: 0.74}


def test_speed_filter_handles_zero_and_prohibitive_costs():
    index = pd.date_range("2022-01-01", periods=900, freq="D")
    returns = 0.001 + 0.02 * np.sin(np.arange(len(index)) / 8)
    prices = pd.DataFrame({"AAAUSDT": 100 * np.cumprod(1 + returns)}, index=index)
    zero = pd.Series({"AAAUSDT": 0.0})
    free, _ = eligibility_path(prices, zero, zero, taker_share=1, slippage_bps=0, risk_window=60, schedule="quarterly")
    assert free[-1, 0].all()
    expensive = pd.Series({"AAAUSDT": 0.50})
    blocked, records = eligibility_path(prices, expensive, expensive, taker_share=1, slippage_bps=0, risk_window=60, schedule="quarterly")
    assert not blocked[-1, 0].any()
    assert {row["reason"] for row in records} >= {"exceeds_cost_budget"}


def test_single_active_rule_has_fdm_one_and_final_cap():
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    raw = np.full((2, 1, 5), 100.0)
    scalars = np.ones((2, 5))
    weights = np.full((2, 5), 0.2)
    correlations = np.broadcast_to(np.eye(5), (2, 5, 5)).copy()
    eligible = np.zeros((2, 1, 5), dtype=bool)
    eligible[:, :, 2] = True
    forecast, fdm, effective = combine_components(raw, scalars, weights, correlations, eligible, index, pd.Index(["AAAUSDT"]))
    assert np.allclose(fdm, 1.0)
    assert np.allclose(effective[:, 0, 2], 1.0)
    assert np.allclose(forecast.iloc[:, 0], 1.0)


def test_fixed_fdm_is_applied_only_when_multiple_rules_are_active():
    index = pd.date_range("2024-01-01", periods=1, freq="D")
    raw = np.full((1, 1, 5), 10.0)
    scalars = np.ones((1, 5)); weights = np.full((1, 5), 0.2)
    correlations = np.broadcast_to(np.eye(5), (1, 5, 5)).copy()
    eligible = np.zeros((1, 1, 5), dtype=bool); eligible[:, :, :2] = True
    forecast, fdm, _ = combine_components(raw, scalars, weights, correlations, eligible, index, pd.Index(["AAAUSDT"]), fixed_fdm=1.20)
    assert np.allclose(fdm, 1.20)
    assert np.allclose(forecast.iloc[:, 0], 0.60)


def test_selector_cannot_promote_single_horizon_control():
    metric = {"validation": {"net_sharpe": 1.0, "annual_turnover": 10.0}}
    rows = [
        {"config_id": "control", "config": {"family": "breakout_control", "selection_eligible": False,
          "activation": "next_open", "taker_share": 1.0, "slippage_bps": 5.0}, "metrics": metric},
        {"config_id": "combined", "config": {"family": "breakout", "selection_eligible": True,
          "activation": "next_open", "taker_share": 1.0, "slippage_bps": 5.0,
          "gross_cap": 1.0, "target_vol": 0.15}, "metrics": metric},
    ]
    assert deterministic_select(rows, "breakout")["config_id"] == "combined"


def test_selector_breaks_sharpe_ties_with_lower_turnover_then_lower_risk():
    def row(identifier, turnover, gross, target):
        return {"config_id": identifier, "config": {"family": "breakout", "selection_eligible": True,
                "activation": "next_open", "taker_share": 1.0, "slippage_bps": 5.0,
                "gross_cap": gross, "target_vol": target},
                "metrics": {"validation": {"net_sharpe": 1.0, "annual_turnover": turnover}}}
    rows = [row("high-turnover", 20, 1, .15), row("high-risk", 10, 2, .20), row("winner", 10, 1, .15)]
    assert deterministic_select(rows, "breakout")["config_id"] == "winner"
