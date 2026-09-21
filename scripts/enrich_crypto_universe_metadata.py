from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from urllib3.exceptions import InsecureRequestWarning

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from momo_bot.exchange import get_binance_client

INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"


def main():
    warnings.simplefilter("ignore", InsecureRequestWarning)
    snapshot = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    members = snapshot.loc[snapshot.member.astype(str).str.lower().isin(["true", "1"]), "symbol"]
    mappings = pd.read_csv(INPUTS / "symbol_mappings.csv").set_index("binance_symbol")
    volumes = pd.read_parquet(INPUTS / "quote_volume_30d.parquet")
    exchange = get_binance_client().futures_exchange_info()
    exchange_rows = {row["symbol"]: row for row in exchange["symbols"]}
    session = requests.Session(); session.verify = os.getenv("MOMO_PROVIDER_VERIFY_SSL", "false").lower() in {"1", "true", "yes"}
    cache_path = INPUTS / "coingecko_asset_metadata.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    rows = []
    for number, symbol in enumerate(members, 1):
        provider_id = mappings.loc[symbol, "provider_id"] if symbol in mappings.index else None
        if provider_id and provider_id not in cache:
            for attempt in range(4):
                response = session.get(f"https://api.coingecko.com/api/v3/coins/{provider_id}",
                                       params={"localization": "false", "tickers": "false", "market_data": "false",
                                               "community_data": "false", "developer_data": "false"}, timeout=30)
                if response.status_code == 200:
                    payload = response.json(); cache[provider_id] = {"categories": payload.get("categories", []), "name": payload.get("name")}; break
                time.sleep(3 * (attempt + 1))
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(1.2)
        info = exchange_rows.get(symbol, {})
        rows.append({"symbol": symbol, "provider_id": provider_id, "provider_name": cache.get(provider_id, {}).get("name"),
                     "sector": (cache.get(provider_id, {}).get("categories") or ["Unclassified"])[0],
                     "categories_json": json.dumps(cache.get(provider_id, {}).get("categories", []), ensure_ascii=False),
                     "listing_time": pd.to_datetime(info.get("onboardDate"), unit="ms", utc=True, errors="coerce"),
                     "median_quote_volume_30d": float(volumes[symbol].median()) if symbol in volumes else None})
        if number % 20 == 0: print(f"metadata {number}/{len(members)}", flush=True)
    pd.DataFrame(rows).to_parquet(INPUTS / "universe_metadata.parquet", index=False)


if __name__ == "__main__": main()
