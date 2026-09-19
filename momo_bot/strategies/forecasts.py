from __future__ import annotations

import numpy as np
import pandas as pd


def calc_ewma_forecast(price: pd.Series, Lfast: int, Lslow: int | None = None, *, vol_lookback: int = 365) -> pd.Series:
    if Lslow is None:
        Lslow = 4 * Lfast

    fast_ewma = price.ewm(span=Lfast).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma

    vol = price.diff().rolling(vol_lookback).std()
    return raw_ewmac / vol


def get_signal(
    price: pd.Series,
    forecast: pd.Series,
    get_daily_returns_volatility,
    instrument_multiplier: float,
    **kwargs,
) -> pd.Series:
    if forecast is None:
        raise Exception("If you don't provide a series of trades or positions, I need a forecast")

    forecast = forecast * instrument_multiplier
    return forecast.apply(lambda x: (x if ((x < 20) & (x > -20)) else (-20 if x < -20 else 20)))


def calc_breakout_forecast(price_series: pd.Series, horizon: int) -> pd.Series:
    rolling_max = price_series.rolling(window=horizon, min_periods=horizon).max()
    rolling_min = price_series.rolling(window=horizon, min_periods=horizon).min()

    rolling_mean = (rolling_max + rolling_min) / 2
    price_range = rolling_max - rolling_min
    price_range[price_range < 1e-8] = np.nan

    raw_forecast = 40.0 * (price_series - rolling_mean) / price_range
    smoothing_period = max(1, int(horizon / 4))
    return raw_forecast.ewm(span=smoothing_period, adjust=False).mean()

