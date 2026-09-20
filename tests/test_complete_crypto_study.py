import numpy as np
import pandas as pd

from scripts.execute_complete_crypto_study import evaluate, held_weights


def test_next_open_activates_on_following_candle():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    target = pd.DataFrame({"BTCUSDT": [0.4, 0.2, -0.1, 0.0]}, index=index)
    held = held_weights(target, "daily", "next_open")
    assert held.BTCUSDT.tolist() == [0.0, 0.4, 0.2, -0.1]


def test_next_open_uses_forward_return_at_held_timestamp():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    columns = ["BTCUSDT"]
    prices = pd.DataFrame(1.0, index=index, columns=columns)
    forward_returns = pd.DataFrame({"BTCUSDT": [0.01, 0.10, -0.05, 0.0]}, index=index)
    unit = pd.DataFrame({"BTCUSDT": [1.0, 1.0, 1.0, 1.0]}, index=index)
    zero = pd.DataFrame(0.0, index=index, columns=columns)
    mask = {"all": pd.Series(True, index=index)}
    rates = pd.Series(0.0, index=columns)
    _, net, held, _ = evaluate(prices, forward_returns, unit, 1.0, 1.0, "daily", "next_open", 1.0, 0.0,
                                zero, zero, mask, rates, rates)
    assert held.iloc[0, 0] == 0
    assert held.iloc[1, 0] == 1
    assert np.isclose(net.iloc[1], 0.10)


def test_next_close_defers_funding_position_one_day():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    columns = ["BTCUSDT"]
    prices = pd.DataFrame(1.0, index=index, columns=columns)
    zero = pd.DataFrame(0.0, index=index, columns=columns)
    unit = pd.DataFrame(1.0, index=index, columns=columns)
    same = pd.DataFrame({"BTCUSDT": [0.0, -0.01, -0.02, -0.03]}, index=index)
    mask = {"all": pd.Series(True, index=index)}
    rates = pd.Series(0.0, index=columns)
    _, _, _, funding = evaluate(prices, zero, unit, 1.0, 1.0, "daily", "next_close", 1.0, 0.0,
                                 same, zero, mask, rates, rates)
    assert funding.iloc[1] == 0
    assert np.isclose(funding.iloc[2], -0.02)
