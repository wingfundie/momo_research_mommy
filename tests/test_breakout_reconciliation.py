from __future__ import annotations

import json

import pandas as pd
import pytest


def test_price_coverage_allows_prelisting_dates() -> None:
    from momo_bot.data_validation import validate_price_coverage

    frame = pd.DataFrame(
        {
            "OLDUSDT": [1.0, 2.0, 3.0],
            "NEWUSDT": [pd.NA, pd.NA, 9.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = validate_price_coverage(
        frame,
        required_tickers=["OLDUSDT", "NEWUSDT"],
        freq="1d",
        required_timestamp=pd.Timestamp("2024-01-03"),
    )
    assert result.is_publishable
    assert result.missing_internal_cells == 0


def test_price_coverage_rejects_partial_latest_row() -> None:
    from momo_bot.data_validation import validate_price_coverage

    frame = pd.DataFrame(
        {
            "AAAUSDT": [1.0, 2.0, 3.0],
            "BBBUSDT": [4.0, 5.0, pd.NA],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = validate_price_coverage(
        frame,
        required_tickers=["AAAUSDT", "BBBUSDT"],
        freq="1d",
        required_timestamp=pd.Timestamp("2024-01-03"),
    )
    assert not result.is_publishable
    assert result.missing_latest_tickers == ["BBBUSDT"]


def test_price_coverage_uses_futures_onboard_date() -> None:
    from momo_bot.data_validation import resolve_coverage_starts, validate_price_coverage

    frame = pd.DataFrame(
        {"AAAUSDT": [10.0, pd.NA, 12.0, 13.0, 14.0]},
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    coverage_starts = resolve_coverage_starts(
        frame,
        required_tickers=["AAAUSDT"],
        onboard_date_by_ticker={"AAAUSDT": pd.Timestamp("2024-01-02")},
    )
    result = validate_price_coverage(
        frame,
        required_tickers=["AAAUSDT"],
        freq="1d",
        required_timestamp=pd.Timestamp("2024-01-05"),
        expected_start_by_ticker=coverage_starts,
    )
    assert coverage_starts["AAAUSDT"] == pd.Timestamp("2024-01-03")
    assert result.is_publishable
    assert result.missing_internal_cells == 0


def test_confirmed_exchange_gap_is_publishable() -> None:
    from momo_bot.data_validation import validate_price_coverage

    frame = pd.DataFrame(
        {"AAAUSDT": [1.0, pd.NA, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = validate_price_coverage(
        frame,
        required_tickers=["AAAUSDT"],
        freq="1d",
        required_timestamp=pd.Timestamp("2024-01-03"),
        expected_start_by_ticker={"AAAUSDT": pd.Timestamp("2024-01-01")},
        allowed_missing_by_ticker={"AAAUSDT": [pd.Timestamp("2024-01-02")]},
    )
    assert result.is_publishable
    assert result.missing_internal_cells == 1
    assert result.unresolved_internal_cells == 0
    assert result.confirmed_unavailable_cells == 1


def test_refresh_preserves_existing_prices_and_writes_metadata(monkeypatch, tmp_path) -> None:
    import mommy_bot

    output = tmp_path / "crypto_tickers_1d.pkl"
    existing = pd.DataFrame(
        {"AAAUSDT": [1.0, pd.NA, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    monkeypatch.setattr(
        mommy_bot,
        "expected_last_complete_open_time",
        lambda freq: pd.Timestamp("2024-01-03"),
    )
    monkeypatch.setattr(
        mommy_bot.cta,
        "get_fresh_lookback_df",
        lambda ticker, start_date, freq="1d": pd.DataFrame(
            {"Close": [100.0, 2.0, 300.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        ),
    )

    updated = mommy_bot.update_historical_data(
        existing,
        freq="1d",
        out_path=output,
        target_tickers=["AAAUSDT"],
        require_complete=True,
        signal_tickers=["AAAUSDT"],
        expected_start_by_ticker={"AAAUSDT": pd.Timestamp("2024-01-01")},
    )
    assert updated["AAAUSDT"].tolist() == [1.0, 2.0, 3.0]
    saved = pd.read_pickle(output)
    assert saved["AAAUSDT"].tolist() == [1.0, 2.0, 3.0]
    metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["coverage"]["is_publishable"] is True
    assert metadata["signal_ticker_count"] == 1
    assert metadata["expected_start_by_ticker"]["AAAUSDT"] == "2024-01-01T00:00:00"


def test_refresh_records_exchange_confirmed_gap(monkeypatch, tmp_path) -> None:
    import mommy_bot

    output = tmp_path / "crypto_tickers_1d.pkl"
    existing = pd.DataFrame(
        {"AAAUSDT": [1.0, pd.NA, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    monkeypatch.setattr(
        mommy_bot,
        "expected_last_complete_open_time",
        lambda freq: pd.Timestamp("2024-01-03"),
    )
    monkeypatch.setattr(
        mommy_bot.cta,
        "get_fresh_lookback_df",
        lambda ticker, start_date, freq="1d": pd.DataFrame(
            {"Close": [1.0, 3.0]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-03"]),
        ),
    )

    mommy_bot.update_historical_data(
        existing,
        freq="1d",
        out_path=output,
        target_tickers=["AAAUSDT"],
        require_complete=True,
        expected_start_by_ticker={"AAAUSDT": pd.Timestamp("2024-01-01")},
    )
    metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["coverage"]["is_publishable"] is True
    assert metadata["coverage"]["confirmed_unavailable_cells"] == 1
    assert metadata["confirmed_unavailable_by_ticker"]["AAAUSDT"] == ["2024-01-02T00:00:00"]


def test_failed_coverage_keeps_last_good_file(monkeypatch, tmp_path) -> None:
    import mommy_bot

    output = tmp_path / "crypto_tickers_1d.pkl"
    last_good = pd.DataFrame(
        {"AAAUSDT": [1.0]},
        index=pd.DatetimeIndex(["2024-01-01"]),
    )
    last_good.to_pickle(output)
    monkeypatch.setattr(
        mommy_bot,
        "expected_last_complete_open_time",
        lambda freq: pd.Timestamp("2024-01-02"),
    )
    monkeypatch.setattr(
        mommy_bot.cta,
        "get_fresh_lookback_df",
        lambda ticker, start_date, freq="1d": pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="keeping the last good file"):
        mommy_bot.update_historical_data(
            last_good,
            freq="1d",
            out_path=output,
            target_tickers=["AAAUSDT"],
            require_complete=True,
        )
    pd.testing.assert_frame_equal(pd.read_pickle(output), last_good)
