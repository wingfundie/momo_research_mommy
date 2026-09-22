from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from urllib3.exceptions import InsecureRequestWarning

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.binance_data import get_fresh_lookback_df
from momo_bot.candles import expected_last_complete_open_time
from momo_bot.exchange import get_binance_client
from momo_bot.research.funding import paginate_funding_history
from momo_bot.xsec20 import CONFIG_ID, XSec20Service


INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"
PRICE_PATH = ROOT / "data_store/crypto_research_prices_1d.pkl"
OPEN_PATH = INPUTS / "portfolio_open_prices.parquet"
FUNDING_PATH = INPUTS / "funding_events.parquet"
warnings.simplefilter("ignore", InsecureRequestWarning)


def _atomic_pickle(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        frame.to_pickle(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, destination: Path, *, index: bool = True) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=index)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _members() -> list[str]:
    universe = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    return sorted(
        universe.loc[
            universe["member"].astype(str).str.lower().isin(["true", "1"]),
            "symbol",
        ].astype(str)
    )


def _normalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index, utc=True).tz_localize(None)
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def refresh_daily_market_inputs(*, full_price_universe: bool, include_funding: bool) -> dict:
    required = expected_last_complete_open_time("1d")
    members = _members()
    prices = _normalise_index(pd.read_pickle(PRICE_PATH))
    opens = _normalise_index(pd.read_parquet(OPEN_PATH))
    symbols = sorted(set(prices.columns) | set(members)) if full_price_universe else members
    client = get_binance_client()
    failed: dict[str, str] = {}

    for number, symbol in enumerate(symbols, 1):
        last_close = prices[symbol].last_valid_index() if symbol in prices else None
        last_open = opens[symbol].last_valid_index() if symbol in opens else None
        close_is_fresh = last_close is not None and pd.Timestamp(last_close) >= required
        open_is_fresh = symbol not in members or (
            last_open is not None and pd.Timestamp(last_open) >= required
        )
        if close_is_fresh and open_is_fresh:
            continue
        starts = [pd.Timestamp(value) for value in (last_close, last_open) if value is not None]
        start = min(starts) if starts else pd.Timestamp("2019-01-01")
        try:
            klines = get_fresh_lookback_df(symbol, start, freq="1d", client=client)
            if klines.empty:
                raise ValueError("Binance returned no daily klines")
            klines = _normalise_index(klines).loc[:required]
            prices = prices.reindex(prices.index.union(klines.index))
            opens = opens.reindex(opens.index.union(klines.index))
            if symbol not in prices:
                prices[symbol] = pd.NA
            prices.loc[klines.index, symbol] = klines["Close"].astype(float)
            if symbol in members:
                if symbol not in opens:
                    opens[symbol] = pd.NA
                opens.loc[klines.index, symbol] = klines["Open"].astype(float)
        except Exception as exc:
            failed[symbol] = str(exc)
        if number % 50 == 0:
            print(f"daily klines {number}/{len(symbols)}", flush=True)
        time.sleep(0.02)

    prices = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    opens = opens.apply(pd.to_numeric, errors="coerce").sort_index()
    missing_close = [
        symbol
        for symbol in members
        if symbol not in prices or prices[symbol].last_valid_index() is None
        or pd.Timestamp(prices[symbol].last_valid_index()) < required
    ]
    missing_open = [
        symbol
        for symbol in members
        if symbol not in opens or opens[symbol].last_valid_index() is None
        or pd.Timestamp(opens[symbol].last_valid_index()) < required
    ]
    if missing_close or missing_open:
        raise RuntimeError(
            "Daily XSec20 inputs are incomplete; previous files were retained. "
            f"close={missing_close[:12]}, open={missing_open[:12]}, failures={dict(list(failed.items())[:5])}"
        )

    funding_added = 0
    funding = pd.read_parquet(FUNDING_PATH)
    funding["funding_time"] = pd.to_datetime(funding["funding_time"], utc=True)
    if include_funding:
        additions = []
        now = pd.Timestamp.now(tz="UTC")
        for number, symbol in enumerate(members, 1):
            existing = funding.loc[funding["symbol"].eq(symbol), "funding_time"]
            start = existing.max() + pd.Timedelta(milliseconds=1) if not existing.empty else pd.Timestamp("2019-01-01", tz="UTC")
            if start >= now:
                continue
            try:
                fresh = paginate_funding_history(
                    client.futures_funding_rate,
                    symbol,
                    start,
                    now,
                )
                if not fresh.empty:
                    additions.append(fresh)
            except Exception as exc:
                failed[f"funding:{symbol}"] = str(exc)
            if number % 25 == 0:
                print(f"funding {number}/{len(members)}", flush=True)
            time.sleep(0.02)
        if additions:
            funding_added = sum(len(frame) for frame in additions)
            funding = pd.concat([funding, *additions], ignore_index=True)
            funding = funding.drop_duplicates(["symbol", "funding_time"], keep="last")
            funding = funding.sort_values(["symbol", "funding_time"])

    # Publish only after the required close/open coverage has passed. The model
    # service detects these mtimes and rebuilds its immutable runtime snapshot.
    _atomic_pickle(prices, PRICE_PATH)
    _atomic_parquet(opens, OPEN_PATH)
    if include_funding:
        _atomic_parquet(funding, FUNDING_PATH, index=False)
    return {
        "required_candle": required.isoformat(),
        "model_symbols": len(members),
        "price_symbols": len(symbols),
        "funding_events_added": funding_added,
        "non_blocking_failures": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh completed daily XSec20 inputs and rebuild the Telegram runtime snapshot."
    )
    parser.add_argument(
        "--model-universe-only",
        action="store_true",
        help="Refresh current portfolio members only. The default refreshes the full calibration panel.",
    )
    parser.add_argument("--skip-funding", action="store_true")
    parser.add_argument(
        "--refresh-universe",
        action="store_true",
        help="Rebuild the point-in-time universe first; intended for the monthly scheduled run.",
    )
    args = parser.parse_args()

    if args.refresh_universe:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_crypto_research_inputs.py"),
                "--prices",
                str(PRICE_PATH),
                "--output",
                str(INPUTS),
            ],
            cwd=ROOT,
            check=True,
        )

    refresh = refresh_daily_market_inputs(
        full_price_universe=not args.model_universe_only,
        include_funding=not args.skip_funding,
    )
    snapshot = XSec20Service(root=ROOT).get_snapshot(refresh=True)
    output = {
        **refresh,
        "config_id": snapshot.manifest["config_id"],
        "data_cutoff": snapshot.manifest["data_cutoff"],
        "validation_net_sharpe": snapshot.manifest["validation_net_sharpe"],
        "holdout_net_sharpe": snapshot.manifest["holdout_net_sharpe"],
    }
    if output["config_id"] != CONFIG_ID:
        raise RuntimeError(f"Unexpected XSec20 configuration: {output['config_id']}")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
