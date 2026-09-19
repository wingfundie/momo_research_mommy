from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


_FREQ_RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")


def _utcnow_naive() -> pd.Timestamp:
    # pandas returns tz-naive timestamps; interpret as UTC consistently across the project
    return pd.Timestamp.utcnow()


def normalize_timestamp(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def normalize_dt_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def _parse_freq(freq: str) -> tuple[int, str]:
    m = _FREQ_RE.match(freq)
    if not m:
        raise ValueError(f"Unsupported freq format: {freq!r} (expected like '1d', '4h', '1h')")
    n = int(m.group(1))
    unit = m.group(2).lower()
    return n, unit


def expected_last_complete_open_time(freq: str, now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """
    Returns the open timestamp of the most recent fully completed candle for `freq`.

    Assumptions:
    - Candle timestamps in stored data are candle OPEN times (Binance kline open time).
    - We use UTC (tz-naive timestamps interpreted as UTC).
    """
    n, unit = _parse_freq(freq)
    now_ts = normalize_timestamp(now or _utcnow_naive())

    if unit == "d":
        # Daily candles open at 00:00 UTC and complete at next 00:00 UTC.
        day_open = now_ts.floor("D")
        return day_open - pd.Timedelta(days=n)

    if unit == "h":
        hours = n
        period = f"{hours}h"
        boundary = now_ts.floor(period)
        return boundary - pd.Timedelta(hours=hours)

    raise ValueError(f"Unsupported freq unit: {unit!r} (from {freq!r})")


@dataclass(frozen=True)
class Freshness:
    latest: Optional[pd.Timestamp]
    required: pd.Timestamp
    is_fresh: bool


def check_freshness(index: pd.DatetimeIndex, freq: str, now: Optional[pd.Timestamp] = None) -> Freshness:
    if index is None or len(index) == 0:
        required = expected_last_complete_open_time(freq=freq, now=now)
        return Freshness(latest=None, required=required, is_fresh=False)

    idx = normalize_dt_index(index)
    latest = normalize_timestamp(idx.max())
    required = expected_last_complete_open_time(freq=freq, now=now)
    return Freshness(latest=latest, required=required, is_fresh=latest >= required)
