from dataclasses import replace

import numpy as np
import pandas as pd

from momo_bot.research.config import RiskConfig
from momo_bot.research.portfolio import risk_target_weights, simulate_portfolio


def test_caps_and_next_candle_activation():
    index = pd.date_range("2024-01-01", periods=5)
    raw = pd.DataFrame({"a": [1] * 5, "b": [-1] * 5}, index=index)
    vol = pd.DataFrame(.2, index=index, columns=raw.columns)
    config = replace(RiskConfig(), gross_leverage_cap=.3, single_ticker_risk_cap=.2)
    weights = risk_target_weights(raw, vol, config)
    assert (weights.abs().sum(axis=1) <= .3 + 1e-12).all()
    result = simulate_portfolio(pd.DataFrame({"a": [100, 110, 120, 130, 140], "b": [100]*5}, index=index), weights, config)
    assert (result.weights.iloc[0] == 0).all()


def test_funding_reconciles_into_net_return():
    index = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"a": [100, 100, 100]}, index=index)
    weights = pd.DataFrame({"a": [0, 0, 0]}, index=index)
    config = replace(RiskConfig(), maker_fee=0, taker_fee=0, slippage_bps=0, portfolio_value=1000)
    result = simulate_portfolio(prices, weights, config, funding_cashflows_usd=pd.Series([0, 10, -5], index=index))
    assert result.net_returns.sum() == .005
    assert result.funding_cashflow.sum() == 5


def test_portfolio_volatility_scaler_accounts_for_correlation():
    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-01", periods=500)
    shared = pd.Series(rng.normal(0, .01, len(index)), index=index)
    returns = pd.DataFrame({"a": shared, "b": shared})
    raw = pd.DataFrame(1.0, index=index, columns=returns.columns)
    asset_vol = returns.rolling(90, min_periods=90).std() * np.sqrt(365)
    config = replace(RiskConfig(), annual_volatility_target=.10, gross_leverage_cap=10,
                     single_ticker_risk_cap=.5)
    weights = risk_target_weights(raw, asset_vol, config, asset_returns=returns)
    realized = (weights.shift(1) * returns).sum(axis=1).iloc[200:].std() * np.sqrt(365)
    assert realized == pytest.approx(.10, abs=.02)


import pytest
