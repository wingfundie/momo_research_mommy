from __future__ import annotations

import pandas as pd


def test_parse_closed_kline_event_ignores_unfinished_candles():
    from momo_bot.binance_stream import parse_closed_kline_event

    event = {
        "data": {
            "k": {
                "s": "BTCUSDT",
                "i": "1d",
                "t": 1717200000000,
                "c": "67500.5",
                "x": False,
            }
        }
    }

    assert parse_closed_kline_event(event) is None


def test_parse_closed_kline_event_returns_closed_candle():
    from momo_bot.binance_stream import parse_closed_kline_event

    event = {
        "data": {
            "k": {
                "s": "btcusdt",
                "i": "1d",
                "t": 1717200000000,
                "c": "67500.5",
                "x": True,
            }
        }
    }

    candle = parse_closed_kline_event(event)

    assert candle is not None
    assert candle.symbol == "BTCUSDT"
    assert candle.interval == "1d"
    assert candle.open_time == pd.Timestamp("2024-06-01 00:00:00")
    assert candle.close == 67500.5


def test_market_data_store_merges_closed_candle_and_ignores_unknown(tmp_path):
    from momo_bot.binance_stream import ClosedCandle
    from momo_bot.data_store import MarketDataStore

    path = tmp_path / "prices.pkl"
    initial = pd.DataFrame(
        {"BTCUSDT": [1.0]},
        index=pd.DatetimeIndex(["2024-06-01 00:00:00"]),
    )
    initial.to_pickle(path)
    store = MarketDataStore(path=path, freq="1d")

    assert store.merge_closed_candle(
        ClosedCandle(
            symbol="BTCUSDT",
            interval="1d",
            open_time=pd.Timestamp("2024-06-02 00:00:00", tz="UTC"),
            close=2.0,
        )
    )
    assert not store.merge_closed_candle(
        ClosedCandle(
            symbol="ETHUSDT",
            interval="1d",
            open_time=pd.Timestamp("2024-06-02 00:00:00"),
            close=3.0,
        )
    )

    saved = pd.read_pickle(path)
    assert saved.loc[pd.Timestamp("2024-06-02 00:00:00"), "BTCUSDT"] == 2.0
    assert "ETHUSDT" not in saved.columns
