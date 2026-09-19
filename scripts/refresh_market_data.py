from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mommy_bot as momo
import portfolio_strategy as cta
from momo_bot.config import settings
from momo_bot.data_validation import validate_price_frame


def _resolve_strategy_defaults(strategy: str) -> tuple[Path, str]:
    if strategy == "momentum":
        return settings.momentum_price_data_path, settings.momentum_data_frequency
    if strategy == "breakout":
        if settings.breakout_price_data_path is None:
            raise FileNotFoundError("Breakout price data path is not configured.")
        return settings.breakout_price_data_path, settings.breakout_data_frequency
    raise ValueError(f"Unknown strategy: {strategy}")


def _load_optimized_symbols(strategy: str) -> list[str]:
    if strategy == "momentum":
        path = settings.momentum_params_path
    elif strategy == "breakout":
        if settings.breakout_params_path is None:
            return []
        path = settings.breakout_params_path
    else:
        return []

    if not path.exists():
        return []
    raw = pd.read_pickle(path)
    if not isinstance(raw, dict):
        return []
    symbols = []
    for symbol, payload in raw.items():
        if isinstance(payload, dict) and ("weights" in payload or payload.get("status") == "success"):
            symbols.append(str(symbol))
    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh and expand stored Binance futures price data.")
    parser.add_argument("--strategy", choices=["momentum", "breakout"], required=True)
    parser.add_argument("--freq", default="auto")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--universe",
        choices=["optimized", "stored", "all"],
        default=settings.refresh_universe if settings.refresh_universe in {"optimized", "stored", "all"} else "optimized",
        help="Symbols to refresh. Use 'all' only for manual, throttled full-universe maintenance.",
    )
    parser.add_argument("--expand-universe", action="store_true", default=False, help="Deprecated alias for --universe all.")
    parser.add_argument("--max-symbols-per-run", type=int, default=0)
    parser.add_argument("--start-date-for-new", default=settings.new_ticker_start_date)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    default_path, configured_frequency = _resolve_strategy_defaults(args.strategy)
    path = Path(args.output) if args.output else default_path
    if not path.exists():
        raise FileNotFoundError(f"Price data not found: {path}")

    df = pd.read_pickle(path)
    validation = validate_price_frame(
        df,
        path=path,
        configured_frequency=configured_frequency if args.freq == "auto" else args.freq,
        strict=settings.strict_data_frequency,
    )
    freq = validation.effective_frequency
    if args.freq != "auto":
        freq = args.freq

    if args.expand_universe:
        args.universe = "all"

    target_tickers = None
    expected_start_by_ticker = None
    if args.universe == "all":
        expected_start_by_ticker = cta.get_usdt_perpetual_futures_onboard_dates(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
        )
        target_tickers = list(expected_start_by_ticker)
    elif args.universe == "optimized":
        target_tickers = _load_optimized_symbols(args.strategy)
        if not target_tickers:
            raise RuntimeError(f"No optimized symbols found for {args.strategy}; use --universe stored or --universe all.")
    elif args.universe == "stored":
        target_tickers = sorted(map(str, df.columns))

    if args.max_symbols_per_run and target_tickers:
        target_tickers = target_tickers[: max(1, args.max_symbols_per_run)]
        if expected_start_by_ticker:
            expected_start_by_ticker = {
                symbol: timestamp
                for symbol, timestamp in expected_start_by_ticker.items()
                if symbol in set(target_tickers)
            }

    updated = momo.update_historical_data(
        df,
        freq=freq,
        out_path=path,
        target_tickers=target_tickers,
        start_date_for_new=args.start_date_for_new,
        keep_inactive_existing=settings.keep_inactive_tickers,
        request_sleep_seconds=args.request_sleep_seconds,
        require_complete=args.strategy == "breakout",
        expected_start_by_ticker=expected_start_by_ticker,
    )
    print(f"Saved {path} rows={updated.shape[0]} cols={updated.shape[1]} freq={freq}")


if __name__ == "__main__":
    main()
