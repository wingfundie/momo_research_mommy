import numpy as np
import pandas as pd
import pytest

from scripts.execute_complete_crypto_study import calibration_path
from scripts.backfill_funding_mark_prices import parse_mark_klines
from scripts.execute_full_crypto_study import funding_coefficients
from scripts.execute_production_like_walkforward import (
    fold_definitions,
    funding_tradability_mask,
    model_config_from_name,
    select_rosters,
    selection_window,
    simulate_arrays,
    simulate_search_batch,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ts_equal_vol60", {"rule_weights": "equal", "refit": "quarterly_scalars", "volatility_window": 60}),
        ("ts_shrink_80_primary_annual_vol360", {"rule_weights": "shrink_80_primary", "refit": "annual", "volatility_window": 360}),
        ("breakout_carver5_equal_fixed_fdm_vol90", {"method": "carver_fixed_scalars_equal", "fdm_mode": "fixed_1.20", "refit": "quarterly", "volatility_window": 90}),
        ("breakout_crypto_shrink_75_semiannual_vol180", {"method": "crypto_expanding_scalars_shrunk_weights", "weight_spec": "shrink_75", "fdm_mode": "causal_expanding", "refit": "semiannual", "volatility_window": 180}),
    ],
)
def test_model_names_round_trip_to_exact_configuration_fields(name, expected):
    assert model_config_from_name(name) == expected


def test_unsupported_or_320_day_model_name_is_rejected():
    with pytest.raises(ValueError):
        model_config_from_name("breakout_crypto_equal_quarterly_vol320")


def test_mark_price_kline_parser_uses_open_at_exact_event_timestamp():
    rows = [[1_700_000_000_000, "100.25", "101", "99", "100.5", "0", 0, "0", 0, "0", "0", "0"]]
    parsed = parse_mark_klines(rows)
    assert parsed.index[0] == pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC")
    assert parsed.iloc[0] == 100.25


def test_funding_tradability_starts_day_after_first_fully_priced_event():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    events = pd.DataFrame({
        "symbol": ["AAAUSDT", "AAAUSDT"],
        "funding_time": pd.to_datetime(["2024-01-01 08:00Z", "2024-01-02 08:00Z"]),
        "funding_rate": [0.001, 0.001], "mark_price": [np.nan, 100.0],
    })
    mask = funding_tradability_mask(index, ["AAAUSDT"], events)
    assert mask.AAAUSDT.tolist() == [False, False, True, True]


def test_funding_coefficients_reject_missing_mark_after_eligibility():
    prices = pd.DataFrame({"AAAUSDT": [100.0, 101.0, 102.0]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    events = pd.DataFrame({
        "symbol": ["AAAUSDT", "AAAUSDT"],
        "funding_time": pd.to_datetime(["2024-01-01 08:00Z", "2024-01-02 08:00Z"]),
        "funding_rate": [0.001, 0.001], "mark_price": [100.0, np.nan],
    })
    with pytest.raises(ValueError, match="post-eligibility"):
        funding_coefficients(prices, events)


def test_locked_outer_folds_are_annual_and_causal():
    index = pd.date_range("2019-09-08", "2026-09-18", freq="D")
    folds = fold_definitions(index)
    assert [fold["fold"] for fold in folds] == ["2022", "2023", "2024", "2025", "2026_ytd"]
    assert all(fold["cutoff"] < fold["test_start"] for fold in folds)
    assert folds[-1]["test_end"] == index.max()


def test_expanding_and_730_day_selection_windows_are_distinct():
    index = pd.date_range("2019-09-08", "2026-09-18", freq="D")
    cutoff = pd.Timestamp("2025-12-31")
    expanding = selection_window(index, cutoff, None)
    rolling = selection_window(index, cutoff, 730)
    assert expanding[0] == index.min() + pd.Timedelta(days=365)
    assert rolling[0] == cutoff - pd.Timedelta(days=729)
    assert expanding[1] == rolling[1] == cutoff


def test_rolling_calibration_ignores_observations_outside_lookback():
    index = pd.date_range("2020-01-01", periods=1500, freq="D")
    raw = np.ones((1500, 2, 2), dtype=float)
    raw[:900, :, 0] = 100.0
    returns = np.zeros((1500, 2), dtype=float)
    expanding = calibration_path(raw, returns, index, "annual", 1.0, 0.5, 1)
    rolling = calibration_path(raw, returns, index, "annual", 1.0, 0.5, 1, history_days=730)
    assert not np.allclose(expanding[0][-1], rolling[0][-1])


def test_selector_returns_three_distinct_models_per_family_and_fold():
    records = []
    for number in range(5):
        records.append({
            "variant": "nested_expanding", "fold": "2024", "family": "breakout",
            "model": f"model_{number}", "config_id": f"id_{number}",
            "selection_net_sharpe": 1.0 - number / 10,
            "selection_annual_turnover": 10.0 + number, "gross_cap": 1.0,
            "target_vol": 0.2, "ticker_risk_cap": 0.1, "rebalance": "daily",
        })
    selected = select_rosters(records)
    winners = selected[selected.selected]
    assert winners.model.tolist() == ["model_0", "model_1", "model_2"]
    assert winners["rank"].tolist() == [1, 2, 3]


def test_detailed_simulation_reconciles_asset_contributions():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    unit = np.tile(np.array([[1.0, -1.0]]), (len(index), 1))
    returns = np.tile(np.array([[0.01, -0.005]]), (len(index), 1))
    zeros = np.zeros_like(returns)
    simulation = simulate_arrays(
        unit, index, returns, zeros, zeros, np.array([0.0, 0.0]), np.array([0.0, 0.0]),
        target_vol=0.2, gross_cap=1.0, rebalance="daily", slippage_bps=0.0,
        detail=True, columns=pd.Index(["AAAUSDT", "BBBUSDT"]),
    )
    pd.testing.assert_series_equal(simulation.contribution.sum(axis=1), simulation.net)
    assert simulation.held.iloc[0].eq(0).all()
    assert simulation.held.abs().sum(axis=1).max() <= 1.0 + 1e-12


def test_batched_search_matches_single_configuration_simulation():
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    unit = np.tile(np.array([[0.4, -0.2]]), (len(index), 1))
    returns = np.tile(np.array([[0.01, -0.005]]), (len(index), 1))
    zeros = np.zeros_like(returns)
    combinations, net, turnover = simulate_search_batch(
        unit, index, returns, zeros, zeros, np.array([0.0, 0.0]), np.array([0.0, 0.0]), rebalance="weekly"
    )
    location = combinations.index((0.2, 1.0))
    single_net, single_turnover, _ = simulate_arrays(
        unit, index, returns, zeros, zeros, np.array([0.0, 0.0]), np.array([0.0, 0.0]),
        target_vol=0.2, gross_cap=1.0, rebalance="weekly", slippage_bps=5.0,
    )
    np.testing.assert_allclose(net[location], single_net)
    np.testing.assert_allclose(turnover[location], single_turnover)
