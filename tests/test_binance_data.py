from __future__ import annotations

import pandas as pd


def _kline(open_ms: int, close: float) -> list[object]:
    return [
        open_ms,
        "1",
        "1",
        "1",
        str(close),
        "1",
        open_ms + 1,
        "1",
        1,
        "1",
        "1",
        "0",
    ]


def test_fresh_lookback_paginates_futures_klines_without_boundary_gap() -> None:
    from momo_bot.binance_data import get_fresh_lookback_df

    first_open = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    first_page = [_kline(first_open + offset * day_ms, float(offset)) for offset in range(1000)]
    second_page = [_kline(first_open + 1000 * day_ms, 1000.0)]

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def futures_klines(self, **kwargs):
            self.calls.append(kwargs)
            return first_page if len(self.calls) == 1 else second_page

    client = FakeClient()
    result = get_fresh_lookback_df(
        "AAAUSDT",
        "2020-01-01 00:00:00",
        freq="1d",
        client=client,
    )

    assert len(result) == 1001
    assert not result.index.duplicated().any()
    assert client.calls[1]["startTime"] == first_page[-1][0] + 1
