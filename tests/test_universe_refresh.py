import pandas as pd


def test_update_historical_data_adds_new_tickers(monkeypatch):
    import mommy_bot

    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    existing = pd.DataFrame({"AAAUSDT": [1.0, 2.0]}, index=idx)

    def fake_get_fresh_lookback_df(ticker, start_date, freq="1d"):
        new_idx = pd.date_range("2024-01-03", periods=2, freq="D")
        base = 10.0 if ticker == "AAAUSDT" else 20.0
        return pd.DataFrame({"Close": [base, base + 1.0]}, index=new_idx)

    monkeypatch.setattr(mommy_bot.cta, "get_fresh_lookback_df", fake_get_fresh_lookback_df)

    updated = mommy_bot.update_historical_data(
        existing,
        freq="1d",
        out_path="",
        target_tickers=["AAAUSDT", "BBBUSDT"],
        start_date_for_new="2024-01-01",
        keep_inactive_existing=True,
    )

    assert "BBBUSDT" in updated.columns
    assert updated["BBBUSDT"].dropna().iloc[-1] == 21.0


def test_update_historical_data_keeps_inactive_existing(monkeypatch):
    import mommy_bot

    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    existing = pd.DataFrame({"OLDUSDT": [1.0, 2.0]}, index=idx)

    monkeypatch.setattr(
        mommy_bot.cta,
        "get_fresh_lookback_df",
        lambda ticker, start_date, freq="1d": pd.DataFrame({"Close": [3.0]}, index=pd.DatetimeIndex(["2024-01-03"])),
    )

    updated = mommy_bot.update_historical_data(
        existing,
        freq="1d",
        out_path="",
        target_tickers=["NEWUSDT"],
        start_date_for_new="2024-01-01",
        keep_inactive_existing=True,
    )

    assert "OLDUSDT" in updated.columns
    assert "NEWUSDT" in updated.columns


def test_update_historical_data_preserves_other_columns_on_overlapping_refresh(monkeypatch):
    import mommy_bot

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    existing = pd.DataFrame(
        {
            "AAAUSDT": [1.0, 2.0, 3.0],
            "BBBUSDT": [10.0, 11.0, 12.0],
        },
        index=idx,
    )

    monkeypatch.setattr(
        mommy_bot,
        "expected_last_complete_open_time",
        lambda freq: pd.Timestamp("2024-01-03"),
    )

    def fake_get_fresh_lookback_df(ticker, start_date, freq="1d"):
        if ticker == "AAAUSDT":
            return pd.DataFrame({"Close": [2.5]}, index=pd.DatetimeIndex(["2024-01-02"]))
        return pd.DataFrame()

    monkeypatch.setattr(mommy_bot.cta, "get_fresh_lookback_df", fake_get_fresh_lookback_df)

    existing.loc[pd.Timestamp("2024-01-02"), "AAAUSDT"] = pd.NA
    updated = mommy_bot.update_historical_data(
        existing,
        freq="1d",
        out_path="",
        target_tickers=["AAAUSDT", "BBBUSDT"],
        start_date_for_new="2024-01-01",
        keep_inactive_existing=True,
    )

    assert updated.loc[pd.Timestamp("2024-01-02"), "AAAUSDT"] == 2.5
    assert updated.loc[pd.Timestamp("2024-01-02"), "BBBUSDT"] == 11.0


def test_update_historical_data_repairs_internal_gap_when_last_valid_is_fresh(monkeypatch):
    import mommy_bot

    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    existing = pd.DataFrame({"AAAUSDT": [1.0, pd.NA, 3.0, 4.0]}, index=idx)

    monkeypatch.setattr(
        mommy_bot,
        "expected_last_complete_open_time",
        lambda freq: pd.Timestamp("2024-01-04"),
    )

    requested_starts = []

    def fake_get_fresh_lookback_df(ticker, start_date, freq="1d"):
        requested_starts.append(start_date)
        return pd.DataFrame({"Close": [2.0, 3.0, 4.0]}, index=pd.date_range("2024-01-02", periods=3, freq="D"))

    monkeypatch.setattr(mommy_bot.cta, "get_fresh_lookback_df", fake_get_fresh_lookback_df)

    updated = mommy_bot.update_historical_data(
        existing,
        freq="1d",
        out_path="",
        target_tickers=["AAAUSDT"],
        start_date_for_new="2024-01-01",
        keep_inactive_existing=True,
    )

    assert requested_starts == ["2024-01-02 00:00:00"]
    assert updated.loc[pd.Timestamp("2024-01-02"), "AAAUSDT"] == 2.0


def test_update_historical_data_targets_first_internal_gap(monkeypatch):
    import mommy_bot

    idx = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-01-03",
            "2024-01-04",
            "2024-06-10",
            "2024-06-11",
        ]
    )
    existing = pd.DataFrame({"AAAUSDT": [1.0, 3.0, 4.0, 10.0, 11.0]}, index=idx)

    monkeypatch.setattr(
        mommy_bot,
        "expected_last_complete_open_time",
        lambda freq: pd.Timestamp("2024-06-11"),
    )

    requested_starts = []

    def fake_get_fresh_lookback_df(ticker, start_date, freq="1d"):
        requested_starts.append(start_date)
        return pd.DataFrame({"Close": [5.0]}, index=pd.DatetimeIndex(["2024-01-05"]))

    monkeypatch.setattr(mommy_bot.cta, "get_fresh_lookback_df", fake_get_fresh_lookback_df)

    mommy_bot.update_historical_data(
        existing,
        freq="1d",
        out_path="",
        target_tickers=["AAAUSDT"],
        start_date_for_new="2024-01-01",
        keep_inactive_existing=True,
    )

    assert requested_starts == ["2024-01-02 00:00:00"]


def test_update_historical_data_stops_on_rest_circuit_open(monkeypatch):
    import mommy_bot
    from momo_bot.binance_rate_limit import BinanceRestCircuitOpen

    idx = pd.DatetimeIndex(["2024-01-01"])
    existing = pd.DataFrame({"AAAUSDT": [1.0], "BBBUSDT": [2.0]}, index=idx)

    monkeypatch.setattr(
        mommy_bot,
        "expected_last_complete_open_time",
        lambda freq: pd.Timestamp("2024-01-02"),
    )

    requested = []

    def fake_get_fresh_lookback_df(ticker, start_date, freq="1d"):
        requested.append(ticker)
        raise BinanceRestCircuitOpen("cooling down")

    monkeypatch.setattr(mommy_bot.cta, "get_fresh_lookback_df", fake_get_fresh_lookback_df)

    try:
        mommy_bot.update_historical_data(
            existing,
            freq="1d",
            out_path="",
            target_tickers=["AAAUSDT", "BBBUSDT"],
            keep_inactive_existing=True,
        )
    except BinanceRestCircuitOpen:
        pass
    else:
        raise AssertionError("Expected BinanceRestCircuitOpen")

    assert requested == ["AAAUSDT"]
