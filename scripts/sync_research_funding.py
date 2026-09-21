from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from urllib3.exceptions import InsecureRequestWarning

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.exchange import get_binance_client
from momo_bot.research.funding import paginate_funding_history, paginate_income_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data_store/crypto_momentum_research/inputs"))
    args = parser.parse_args()
    warnings.simplefilter("ignore", InsecureRequestWarning)
    args.output.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(args.prices)
    client = get_binance_client()
    destination = args.output / "funding_events.parquet"
    completed = set()
    existing = pd.DataFrame()
    if destination.exists():
        existing = pd.read_parquet(destination)
        completed = set(existing["symbol"].unique())
    all_frames = [existing] if not existing.empty else []
    for number, symbol in enumerate(prices.columns, 1):
        if symbol in completed:
            continue
        start = prices[symbol].first_valid_index()
        end = prices[symbol].last_valid_index() + pd.Timedelta(days=1)
        try:
            frame = paginate_funding_history(client.futures_funding_rate, symbol, start, end)
            all_frames.append(frame)
            combined = pd.concat(all_frames, ignore_index=True).drop_duplicates(["symbol", "funding_time"])
            combined.to_parquet(destination, index=False)
            print(f"funding {number}/{len(prices.columns)} {symbol}: {len(frame)} events")
        except Exception as exc:
            print(f"funding ERROR {symbol}: {exc}")
    now = pd.Timestamp.now(tz="UTC")
    try:
        income = paginate_income_history(client.futures_income_history, now - pd.Timedelta(days=89), now)
        income.to_parquet(args.output / "actual_funding_income_recent.parquet", index=False)
        print(f"actual account funding: {len(income)} events")
    except Exception as exc:
        print(f"actual funding ERROR: {exc}")


if __name__ == "__main__":
    main()
