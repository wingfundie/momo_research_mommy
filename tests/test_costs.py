import numpy as np
import pandas as pd

from momo_bot.costs import BacktestCostConfig, CommissionRate, calculate_costs


def test_fee_cost_equals_turnover_times_mixed_fee_rate():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    positions = pd.Series([1000.0, -500.0, 0.0], index=idx)
    commission = CommissionRate(symbol="BTCUSDT", maker=0.0002, taker=0.0004, source="test")
    config = BacktestCostConfig(enabled=True, taker_share=0.5, slippage_bps=0.0, include_funding=False)

    costs = calculate_costs(positions, commission=commission, config=config)

    expected_turnover = pd.Series([1000.0, 1500.0, 500.0], index=idx)
    expected_fee_rate = 0.0003
    assert np.allclose(costs.turnover_notional, expected_turnover)
    assert np.allclose(costs.fees, expected_turnover * expected_fee_rate)


def test_funding_cost_uses_signed_previous_position():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    positions = pd.Series([1000.0, -500.0, 0.0], index=idx)
    funding = pd.Series([0.01, 0.02, -0.03], index=idx)
    commission = CommissionRate(symbol="BTCUSDT", maker=0.0, taker=0.0, source="test")
    config = BacktestCostConfig(enabled=True, include_funding=True)

    costs = calculate_costs(positions, commission=commission, config=config, funding_rates=funding)

    assert np.allclose(costs.funding, [0.0, 20.0, 15.0])
