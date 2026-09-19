from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.exchange import get_binance_client
from momo_bot.research.config import UniverseConfig
from momo_bot.research.universe import (
    archive_provider_response, build_monthly_snapshot, eligible_usdt_perpetuals,
    fetch_fdv_cascade, quote_volume_from_klines, resolve_provider_assets,
    trailing_median_quote_volume,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data_store/crypto_momentum_research/inputs"))
    args = parser.parse_args()
    load_dotenv()
    args.output.mkdir(parents=True, exist_ok=True)
    client = get_binance_client()
    exchange_info = client.futures_exchange_info()
    assets = eligible_usdt_perpetuals(exchange_info)
    active = {symbol: asset for symbol, asset in assets.items()
              if next(row for row in exchange_info["symbols"] if row["symbol"] == symbol).get("status") == "TRADING"}
    provider_session = requests.Session()
    provider_session.verify = os.getenv("MOMO_PROVIDER_VERIFY_SSL", "false").lower() in {"1", "true", "yes"}
    provider, provider_rows = fetch_fdv_cascade(session=provider_session,
                                                coinmarketcap_key=os.getenv("COINMARKETCAP_API_KEY"))
    resolved, mappings = resolve_provider_assets(active, provider_rows, provider)
    klines = {}
    for number, symbol in enumerate(sorted(active), 1):
        try:
            klines[symbol] = client.futures_klines(symbol=symbol, interval="1d", limit=31)
        except Exception as exc:
            print(f"volume warning {symbol}: {exc}")
        if number % 50 == 0:
            print(f"volume coverage {number}/{len(active)}")
    volumes = quote_volume_from_klines(klines)
    median_volume = trailing_median_quote_volume(volumes)
    as_of = pd.Timestamp.now(tz="UTC").normalize()
    snapshots = build_monthly_snapshot(as_of, resolved, provider_rows, median_volume,
                                       config=UniverseConfig(), provider=provider, point_in_time=False)
    snapshot_frame = pd.DataFrame([asdict(row) for row in snapshots])
    snapshot_frame.to_csv(args.output / "current_universe_snapshot.csv", index=False)
    mappings.to_csv(args.output / "symbol_mappings.csv", index=False)
    volumes.to_parquet(args.output / "quote_volume_30d.parquet")
    provider_rows.to_parquet(args.output / f"{provider}_fdv_snapshot.parquet", index=False)
    archive_provider_response(provider_rows.to_dict("records"), args.output / f"{provider}_fdv_response.json",
                              provider=provider, timestamp=as_of)
    prices = pd.read_pickle(args.prices)
    members = snapshot_frame.loc[snapshot_frame["member"], "symbol"].tolist()
    available = [symbol for symbol in members if symbol in prices]
    prices[available].to_parquet(args.output / "research_prices.parquet")
    metadata = {"as_of": as_of.isoformat(), "provider": provider, "point_in_time": False,
                "universe_method": "current_universe_historical_fallback", "members": len(members),
                "members_with_prices": len(available), "price_start": str(prices.index.min()),
                "price_end": str(prices.index.max())}
    (args.output / "input_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
