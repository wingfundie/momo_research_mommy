import numpy as np
import pandas as pd

from momo_bot.research.config import SignalConfig
from momo_bot.research.signals import (
    combine_forecasts, constrained_shrunk_weights, cross_sectional_percentiles, ewmac_forecast,
)


def test_ewmac_preserves_warmup_and_caps_final_forecast():
    price = pd.Series(np.exp(np.linspace(0, 2, 300)), index=pd.date_range("2024-01-01", periods=300))
    forecast = ewmac_forecast(price, 2, 8, volatility_lookback=90, scalar=1000)
    assert forecast.iloc[:90].isna().all()
    assert forecast.dropna().abs().max() <= 20


def test_combined_forecast_renormalizes_available_rules_and_caps():
    frame = pd.DataFrame({"a": [np.nan, 10, 100], "b": [np.nan, np.nan, 100]})
    result = combine_forecasts(frame, pd.Series({"a": 0.5, "b": 0.5}))
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 10
    assert result.iloc[2] == 20


def test_constrained_weights_sum_to_one_and_obey_cap():
    result = constrained_shrunk_weights(pd.Series({"a": 9, "b": 1, "c": 0, "d": 0, "e": 0}),
                                        shrinkage_to_equal=.75, max_rule_weight=.30)
    assert result.sum() == pytest.approx(1)
    assert result.max() <= .30 + 1e-12


def test_cross_sectional_rank_is_point_in_time():
    first = pd.DataFrame({"a": [1, 2], "b": [2, 1]}, index=pd.date_range("2024-01-01", periods=2))
    ranks = cross_sectional_percentiles(first)
    changed = first.copy(); changed.iloc[1] = [100, -100]
    assert ranks.iloc[0].equals(cross_sectional_percentiles(changed).iloc[0])


import pytest
