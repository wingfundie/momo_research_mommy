from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.exchange import get_binance_client

INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"


def main():
    snapshot = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    prices = pd.read_parquet(INPUTS / "research_prices.parquet")
    members = snapshot.loc[snapshot.member.astype(str).str.lower().isin(["true", "1"]), "symbol"]
    symbols = [symbol for symbol in members if symbol in prices and prices[symbol].notna().sum() >= 90]
    client = get_binance_client()
    end_ms = int((pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)).timestamp() * 1000)
    start_ms = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)
    series = {}
    for number, symbol in enumerate(symbols, 1):
        rows = []; cursor = start_ms
        while cursor < end_ms:
            page = client.futures_klines(symbol=symbol, interval="1d", startTime=cursor, endTime=end_ms, limit=1000)
            if not page:
                break
            rows.extend(page)
            next_cursor = int(page[-1][0]) + 1
            if next_cursor <= cursor:
                raise RuntimeError(f"Kline pagination failed to advance for {symbol}")
            cursor = next_cursor
            if len(page) < 1000:
                break
            time.sleep(.12)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            index = pd.to_datetime(frame.iloc[:, 0], unit="ms", utc=True).dt.tz_localize(None)
            series[symbol] = pd.Series(pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(), index=index)
        if number % 10 == 0:
            print(f"open prices {number}/{len(symbols)}", flush=True)
        time.sleep(.12)
    opens = pd.DataFrame(series).sort_index()
    opens.to_parquet(INPUTS / "portfolio_open_prices.parquet")
    print({"symbols": opens.shape[1], "rows": opens.shape[0], "start": str(opens.index.min()), "end": str(opens.index.max())}, flush=True)


if __name__ == "__main__":
    main()
