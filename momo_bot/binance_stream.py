from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

USD_M_FUTURES_STREAM_BASE = "wss://fstream.binance.com/stream?streams="


@dataclass(frozen=True)
class ClosedCandle:
    symbol: str
    interval: str
    open_time: pd.Timestamp
    close: float


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    step = max(1, int(size))
    for start in range(0, len(values), step):
        yield values[start : start + step]


def parse_closed_kline_event(event: dict[str, Any]) -> ClosedCandle | None:
    payload = event.get("data", event)
    if not isinstance(payload, dict):
        return None
    kline = payload.get("k")
    if not isinstance(kline, dict) or not kline.get("x"):
        return None
    symbol = str(kline["s"]).upper()
    interval = str(kline["i"])
    open_time = pd.to_datetime(int(kline["t"]), unit="ms", utc=True).tz_localize(None)
    close = float(kline["c"])
    return ClosedCandle(symbol=symbol, interval=interval, open_time=open_time, close=close)


async def run_stream_shard(
    streams: list[str],
    store,
    *,
    connect: Callable[..., Any] | None = None,
    base_url: str = USD_M_FUTURES_STREAM_BASE,
) -> None:
    if not streams:
        return

    if connect is None:
        import websockets  # type: ignore

        connect = websockets.connect

    url = base_url + "/".join(streams)
    backoff_seconds = 1.0
    while True:
        try:
            async with connect(url, ping_interval=20) as ws:
                logger.info("Connected Binance futures stream shard with %d streams", len(streams))
                backoff_seconds = 1.0
                async for raw in ws:
                    event = json.loads(raw)
                    candle = parse_closed_kline_event(event)
                    if candle is None:
                        continue
                    if store.merge_closed_candle(candle):
                        logger.info(
                            "Stored closed Binance candle %s %s %s close=%s",
                            candle.symbol,
                            candle.interval,
                            candle.open_time,
                            candle.close,
                        )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Binance futures WebSocket shard failed; reconnecting")
            await asyncio.sleep(backoff_seconds)
            backoff_seconds = min(60.0, backoff_seconds * 2.0)


async def stream_closed_klines(
    symbols: Iterable[str],
    interval: str,
    store,
    *,
    shard_size: int = 100,
    create_task: Callable[[Awaitable[None]], asyncio.Task] | None = None,
) -> None:
    streams = [f"{symbol.lower()}@kline_{interval}" for symbol in sorted(set(symbols))]
    if not streams:
        return

    task_factory = create_task or asyncio.create_task
    tasks = [
        task_factory(run_stream_shard(shard, store))
        for shard in chunked(streams, shard_size)
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
