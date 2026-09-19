import numpy as np
import pandas as pd

from momo_bot.strategies.forecasts import calc_breakout_forecast, calc_ewma_forecast, get_signal


def test_calc_ewma_forecast_produces_series():
    price = pd.Series(np.linspace(100, 200, 500), index=pd.date_range("2024-01-01", periods=500, freq="D"))
    fc = calc_ewma_forecast(price, Lfast=2, Lslow=8, vol_lookback=30)
    assert isinstance(fc, pd.Series)
    assert len(fc) == len(price)


def test_get_signal_caps_to_20():
    price = pd.Series(np.linspace(100, 200, 100), index=pd.date_range("2024-01-01", periods=100, freq="D"))
    # Create an extreme forecast and scale it up to guarantee clipping.
    forecast = pd.Series(np.ones(len(price)) * 1e9, index=price.index)
    sig = get_signal(price, forecast, None, instrument_multiplier=1.0)
    assert float(sig.max()) == 20.0
    assert float(sig.min()) == 20.0


def test_calc_breakout_forecast_length_and_nan_warmup():
    price = pd.Series(np.linspace(100, 200, 100), index=pd.date_range("2024-01-01", periods=100, freq="D"))
    horizon = 20
    fc = calc_breakout_forecast(price, horizon=horizon)
    assert isinstance(fc, pd.Series)
    assert len(fc) == len(price)
    # Warmup should include NaNs due to rolling window.
    assert fc.iloc[: horizon - 1].isna().all()

