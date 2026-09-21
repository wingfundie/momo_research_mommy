from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from urllib3.exceptions import InsecureRequestWarning


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.exchange import get_binance_client


ENDPOINT = "GET /fapi/v1/markPriceKlines"
INTERVAL = "8h"
SEGMENT_DAYS = 199


def parse_mark_klines(rows: list[list]) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    timestamps = pd.to_datetime([row[0] for row in rows], unit="ms", utc=True)
    values = pd.to_numeric(pd.Series([row[1] for row in rows], index=timestamps), errors="raise")
    return values[~values.index.duplicated(keep="last")].sort_index()


def fetch_symbol_marks(client, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    pieces = []
    cursor = pd.Timestamp(start).tz_convert("UTC").floor(INTERVAL)
    finish = pd.Timestamp(end).tz_convert("UTC").ceil(INTERVAL)
    while cursor <= finish:
        segment_end = min(cursor + pd.Timedelta(days=SEGMENT_DAYS), finish)
        rows = client.futures_mark_price_klines(
            symbol=symbol, interval=INTERVAL, startTime=int(cursor.timestamp() * 1000),
            endTime=int(segment_end.timestamp() * 1000), limit=1000,
        )
        pieces.append(parse_mark_klines(rows))
        cursor = segment_end + pd.Timedelta(milliseconds=1)
        time.sleep(0.05)
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces)[lambda values: ~values.index.duplicated(keep="last")].sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing Binance funding-event mark prices from 8h mark-price klines.")
    parser.add_argument(
        "--funding", type=Path,
        default=ROOT / "data_store/crypto_momentum_research/inputs/funding_events.parquet",
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore", InsecureRequestWarning)
    funding = pd.read_parquet(args.funding)
    funding["funding_time"] = pd.to_datetime(funding["funding_time"], utc=True)
    funding["mark_price"] = pd.to_numeric(funding["mark_price"], errors="coerce")
    if "mark_price_source" not in funding:
        funding["mark_price_source"] = np.where(
            funding.mark_price.notna(), "binance_funding_history", None
        )
    client = get_binance_client()
    records = []
    for number, (symbol, missing) in enumerate(funding[funding.mark_price.isna()].groupby("symbol"), 1):
        marks = fetch_symbol_marks(client, symbol, missing.funding_time.min(), missing.funding_time.max())
        indices = missing.index
        # Binance funding timestamps can be a few milliseconds after the nominal
        # 00:00/08:00/16:00 boundary. They refer to the same settlement event.
        mapped = missing.funding_time.dt.floor(INTERVAL).map(marks)
        resolved = mapped.notna()
        funding.loc[indices[resolved], "mark_price"] = mapped[resolved].to_numpy(float)
        funding.loc[indices[resolved], "mark_price_source"] = "binance_mark_price_kline_8h_open"
        records.append({
            "symbol": symbol, "requested": int(len(missing)), "resolved": int(resolved.sum()),
            "unresolved": int((~resolved).sum()), "first_event": missing.funding_time.min().isoformat(),
            "last_event": missing.funding_time.max().isoformat(),
        })
        funding.to_parquet(args.funding, index=False)
        print(f"mark prices {number}: {symbol} {int(resolved.sum()):,}/{len(missing):,}", flush=True)
    unresolved = funding[funding.mark_price.isna()][["symbol", "funding_time"]]
    first_covered = funding[funding.mark_price.notna()].groupby("symbol")["funding_time"].min()
    late_unresolved = unresolved[
        unresolved.apply(
            lambda row: row.funding_time >= first_covered.get(row.symbol, pd.Timestamp.max.tz_localize("UTC")), axis=1
        )
    ]
    manifest = {
        "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(), "endpoint": ENDPOINT,
        "interval": INTERVAL, "price_field": "8h_kline_open_at_funding_timestamp",
        "source_url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data",
        "events": int(len(funding)), "finite_mark_prices": int(funding.mark_price.notna().sum()),
        "unresolved_pre_eligibility": int(len(unresolved)),
        "unresolved_post_eligibility": int(len(late_unresolved)), "symbols": records,
    }
    manifest_path = args.funding.with_name("funding_mark_price_backfill_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not late_unresolved.empty:
        examples = late_unresolved.head(10).astype(str).to_dict("records")
        raise RuntimeError(f"Funding mark-price backfill incomplete after eligibility: {len(late_unresolved):,} unresolved; examples={examples}")
    print(json.dumps({key: manifest[key] for key in (
        "events", "finite_mark_prices", "unresolved_pre_eligibility", "unresolved_post_eligibility"
    )}, indent=2))


if __name__ == "__main__":
    main()
