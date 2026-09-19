import importlib
import sys

import numpy as np
import pandas as pd


class _DummyBinanceClient:
    def __init__(self, *args, **kwargs):
        pass


def _import_strategy_modules(monkeypatch):
    """Import strategy modules without creating a real Binance client."""
    import binance.client as binance_client

    monkeypatch.setattr(binance_client, "Client", _DummyBinanceClient)
    sys.modules.pop("momo_bot.exchange", None)
    sys.modules.pop("portfolio_strategy", None)
    sys.modules.pop("reoptimize_all", None)

    cta = importlib.import_module("portfolio_strategy")
    ro = importlib.import_module("reoptimize_all")
    return cta, ro


def test_position_size_zero_forecast_is_flat(monkeypatch):
    cta, _ = _import_strategy_modules(monkeypatch)
    price = pd.Series(
        np.linspace(100.0, 120.0, 120),
        index=pd.date_range("2024-01-01", periods=120, freq="D"),
    )
    forecast = pd.Series(0.0, index=price.index)

    positions, positions_usd = cta._position_size_from_forecast(
        price,
        forecast,
        portfolio_value=100_000.0,
        target_vol_annual=0.2,
        data_frequency="1d",
        vol_lookback=30,
    )

    assert np.isclose(float(positions.abs().max()), 0.0)
    assert np.isclose(float(positions_usd.abs().max()), 0.0)


def test_position_size_respects_max_leverage_cap(monkeypatch):
    cta, _ = _import_strategy_modules(monkeypatch)
    price = pd.Series(
        np.linspace(100.0, 101.0, 120),
        index=pd.date_range("2024-01-01", periods=120, freq="D"),
    )
    forecast = pd.Series(20.0, index=price.index)

    _positions, positions_usd = cta._position_size_from_forecast(
        price,
        forecast,
        portfolio_value=100_000.0,
        target_vol_annual=0.2,
        data_frequency="1d",
        vol_lookback=30,
        max_leverage=0.25,
    )

    assert float(positions_usd.abs().max()) <= 25_000.0 + 1e-9


def test_split_helper_returns_first_oos_timestamp(monkeypatch):
    _, ro = _import_strategy_modules(monkeypatch)
    index = pd.date_range("2024-01-01", periods=10, freq="D")

    split_ts, oos_n = ro._split_train_test_index(
        index,
        oos_fraction=0.2,
        oos_periods=None,
    )

    assert oos_n == 2
    assert split_ts == index[-2]


def test_train_oos_split_is_disjoint(monkeypatch):
    _, ro = _import_strategy_modules(monkeypatch)
    index = pd.date_range("2024-01-01", periods=10, freq="D")

    split = ro.split_train_oos(index, oos_fraction=0.2, oos_periods=None)

    assert len(set(split.train_index).intersection(set(split.oos_index))) == 0
    assert split.first_oos_ts == index[-2]


def test_core_imports_do_not_create_binance_client(monkeypatch):
    import binance.client as binance_client

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Binance Client should not be created during core imports")

    monkeypatch.setattr(binance_client, "Client", fail_if_called)
    sys.modules.pop("momo_bot.exchange", None)
    sys.modules.pop("portfolio_strategy", None)
    sys.modules.pop("reoptimize_all", None)

    importlib.import_module("portfolio_strategy")
    importlib.import_module("reoptimize_all")


def test_walkforward_breakout_passes_cost_config_and_saves_cost_series(monkeypatch):
    _, ro = _import_strategy_modules(monkeypatch)
    from momo_bot.costs import BacktestCostConfig

    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    price = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    cost_config = BacktestCostConfig(enabled=True, source="default")
    captured_kwargs = []

    monkeypatch.setattr(
        ro,
        "_optimize_breakout_weights_single",
        lambda price, **kwargs: {"status": "success", "weights": [1.0], "sr": 1.0, "n_valid": 1},
    )
    monkeypatch.setattr(
        ro,
        "_build_breakout_signal",
        lambda price, weights, breakout_horizons, **kwargs: pd.Series(5.0, index=price.index),
    )

    def fake_run_backtest(price, signal, **kwargs):
        captured_kwargs.append(kwargs)
        gross = pd.Series(0.01, index=price.index)
        net = pd.Series(0.005, index=price.index)
        equity = (1.0 + net).cumprod()
        positions = pd.Series(1000.0, index=price.index)
        costs = pd.DataFrame(
            {
                "fees": 1.0,
                "slippage": 0.0,
                "funding": 0.0,
                "total_cost": 1.0,
                "turnover_notional": 2500.0,
            },
            index=price.index,
        )
        return gross, net, equity, positions, costs

    monkeypatch.setattr(ro, "_run_backtest", fake_run_backtest)

    result = ro._run_walkforward_breakout(
        price,
        ticker="BTCUSDT",
        breakout_horizons=[2],
        forecast_scalars={2: 1.0},
        vol_lookback=1,
        n_trials=1,
        data_frequency="1d",
        portfolio_value=100_000.0,
        target_vol_annual=0.2,
        max_leverage=None,
        diversification_multiplier=1.0,
        cap_final_forecast=True,
        opt_mode="best",
        top_percentile=0.2,
        train_window=5,
        test_window=2,
        step_window=2,
        train_mode="rolling",
        cost_config=cost_config,
    )

    assert result["status"] == "success"
    assert result["n_windows"] == 2
    assert all(kwargs["ticker"] == "BTCUSDT" for kwargs in captured_kwargs)
    assert all(kwargs["cost_config"] is cost_config for kwargs in captured_kwargs)
    assert {"gross_returns", "net_returns", "fees", "total_cost", "turnover_notional"}.issubset(result["series"].columns)
    assert len(result["series"]) == 4


def test_walkforward_payload_is_recorded_for_static_skipped_ticker(monkeypatch):
    _, ro = _import_strategy_modules(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    price = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    results = {}

    payload = ro._ensure_breakout_payload(
        results,
        "NEWUSDT",
        price=price,
        params={"status": "skipped_insufficient_data"},
        dm_value=1.24,
    )
    payload["walkforward"] = ro._normalise_walkforward_payload(
        {"status": "skipped_insufficient_data", "n_windows": 0},
        "1d",
    )

    assert "NEWUSDT" in results
    assert payload["params"]["status"] == "skipped_insufficient_data"
    assert payload["params"]["dm"] == 1.24
    assert payload["walkforward"]["status"] == "skipped_insufficient_data"
    assert {"price", "signal", "equity", "positions_usd"}.issubset(payload["series"].columns)
