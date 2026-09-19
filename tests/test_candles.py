import pandas as pd

from momo_bot.candles import (
    check_freshness,
    expected_last_complete_open_time,
    normalize_timestamp,
)


def test_expected_last_complete_open_time_daily():
    now = pd.Timestamp("2025-01-02 12:34:56")
    required = expected_last_complete_open_time("1d", now=now)
    assert required == pd.Timestamp("2025-01-01 00:00:00")


def test_expected_last_complete_open_time_daily_at_midnight():
    # At an exact boundary, the most recently completed daily candle is the previous day.
    now = pd.Timestamp("2025-01-02 00:00:00")
    required = expected_last_complete_open_time("1d", now=now)
    assert required == pd.Timestamp("2025-01-01 00:00:00")


def test_expected_last_complete_open_time_4h():
    now = pd.Timestamp("2025-01-02 10:30:00")
    required = expected_last_complete_open_time("4h", now=now)
    assert required == pd.Timestamp("2025-01-02 04:00:00")


def test_check_freshness_true_when_latest_meets_required():
    now = pd.Timestamp("2025-01-02 12:00:00")
    required = expected_last_complete_open_time("1d", now=now)
    idx = pd.DatetimeIndex([required])
    f = check_freshness(idx, "1d", now=now)
    assert f.is_fresh is True


def test_check_freshness_false_when_latest_before_required():
    now = pd.Timestamp("2025-01-02 12:00:00")
    required = expected_last_complete_open_time("1d", now=now)
    idx = pd.DatetimeIndex([required - pd.Timedelta(days=1)])
    f = check_freshness(idx, "1d", now=now)
    assert f.is_fresh is False


def test_normalize_timestamp_tzaware_to_utc_naive():
    ts = pd.Timestamp("2025-01-02 00:00:00", tz="Asia/Singapore")
    out = normalize_timestamp(ts)
    assert out.tzinfo is None
    assert out == pd.Timestamp("2025-01-01 16:00:00")

