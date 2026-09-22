import numpy as np
import pandas as pd

from scripts.execute_cross_sectional_trend_study import (
    IC_SCHEMES,
    cross_sectional_constructions,
    ic_weighted_forecast,
    model_count,
)


def test_model_count_covers_four_ic_schemes_and_buffer_variants():
    assert len(IC_SCHEMES) == 4
    assert model_count() == 152


def test_ic_weighted_forecast_is_causal_before_future_change():
    index = pd.date_range("2020-01-01", periods=800, freq="D")
    columns = [f"S{i}" for i in range(6)]
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(index), len(columns))), axis=0)),
        index=index, columns=columns,
    )
    components = rng.normal(size=(len(index), len(columns), 5))
    first = ic_weighted_forecast(components, prices, prices, (5, 20), history_days=730)
    changed = prices.copy()
    changed.loc[index[700]:] *= np.arange(1, len(index) - 699)[:, None]
    second = ic_weighted_forecast(components, prices, changed, (5, 20), history_days=730)
    pd.testing.assert_frame_equal(first.loc[:index[678]], second.loc[:index[678]])


def test_cross_sectional_constructions_include_expected_controls():
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    columns = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT"]
    forecast = pd.DataFrame(np.arange(480).reshape(120, 4), index=index, columns=columns)
    returns = forecast.pct_change().fillna(0)
    result = cross_sectional_constructions(forecast, returns, include_buffers=True)
    assert set(result) == {
        "signed_unhedged", "dollar_neutral", "basket_10", "basket_20", "basket_30",
        "long_only_20", "btc_hedged", "beta_constrained", "btc_residualized",
        "basket20_buffer5", "basket20_buffer10",
    }
    assert result["dollar_neutral"][0].sum(axis=1).abs().max() < 1e-12
