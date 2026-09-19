from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mommy_bot as momo
import portfolio_strategy as cta
from momo_bot.config import settings
from momo_bot.data_validation import resolve_coverage_starts, validate_price_coverage


DEFAULT_BOT_SOURCE = Path(
    r"C:\Users\HomePC\Desktop\acausal capital\momentum_run\mom bot\crypto_tickers_1d.pkl"
)
DEFAULT_OUTPUT = ROOT / "data_store" / "crypto_tickers_1d.pkl"


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame at {path}, found {type(frame)!r}")
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index()


def _latest_breakout_params(data_dir: Path) -> tuple[list[str], str]:
    bundles = sorted(
        data_dir.glob("backtest_results_bundle_*.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in bundles:
        bundle = pd.read_pickle(path)
        section = bundle.get("breakout", {}) if isinstance(bundle, dict) else {}
        results = section.get("results", {}) if isinstance(section, dict) else {}
        tickers = [
            ticker
            for ticker, payload in results.items()
            if isinstance(payload, dict)
            and isinstance(payload.get("params"), dict)
            and payload["params"].get("status", "success") == "success"
            and "weights" in payload["params"]
        ]
        if tickers:
            return sorted(tickers), f"bundle:{path.resolve()}"
    raise FileNotFoundError(f"No valid breakout bundle found in {data_dir}")


def _backup(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def _filled_cell_count(base: pd.DataFrame, merged: pd.DataFrame) -> int:
    aligned = base.reindex(index=merged.index, columns=merged.columns)
    return int((aligned.isna() & merged.notna()).sum().sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile the live bot breakout file with repaired history and Binance candles."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--bot-source", type=Path, default=DEFAULT_BOT_SOURCE)
    parser.add_argument("--repair-source", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()

    bot_source = args.bot_source.resolve()
    repair_source = args.repair_source.resolve()
    output = args.output.resolve()
    bot_frame = _load_frame(bot_source)
    repair_frame = _load_frame(repair_source)
    merged = bot_frame.combine_first(repair_frame).sort_index().sort_index(axis=1)

    active_universe = cta.get_usdt_perpetual_futures_onboard_dates(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
    )
    active_tickers = list(active_universe)
    if not active_tickers:
        raise RuntimeError("Could not resolve the active Binance USDT perpetual universe.")
    signal_tickers, params_source = _latest_breakout_params(settings.data_dir)

    bot_coverage_starts = resolve_coverage_starts(
        bot_frame,
        required_tickers=active_tickers,
        onboard_date_by_ticker=active_universe,
    )
    merged_coverage_starts = resolve_coverage_starts(
        merged,
        required_tickers=active_tickers,
        onboard_date_by_ticker=active_universe,
    )
    bot_coverage = validate_price_coverage(
        bot_frame,
        required_tickers=active_tickers,
        freq="1d",
        expected_start_by_ticker=bot_coverage_starts,
    )
    merged_coverage = validate_price_coverage(
        merged,
        required_tickers=active_tickers,
        freq="1d",
        expected_start_by_ticker=merged_coverage_starts,
    )
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "bot_source": str(bot_source),
        "repair_source": str(repair_source),
        "output": str(output),
        "bot_shape": list(bot_frame.shape),
        "merged_shape": list(merged.shape),
        "active_ticker_count": len(active_tickers),
        "signal_ticker_count": len(signal_tickers),
        "cells_filled_from_repaired_source": _filled_cell_count(bot_frame, merged),
        "bot_coverage": bot_coverage.to_dict(),
        "merged_coverage": merged_coverage.to_dict(),
        "params_source": params_source,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backups = {
        "bot_source_backup": _backup(bot_source, stamp),
        "output_backup": _backup(output, stamp),
    }
    print(json.dumps({key: str(value) if value else None for key, value in backups.items()}, indent=2))

    updated = momo.update_historical_data(
        merged,
        freq="1d",
        out_path=output,
        target_tickers=active_tickers,
        start_date_for_new=settings.new_ticker_start_date,
        keep_inactive_existing=True,
        request_sleep_seconds=args.request_sleep_seconds,
        require_complete=True,
        signal_tickers=signal_tickers,
        expected_start_by_ticker=active_universe,
        metadata_extra={
            "bot_source": str(bot_source),
            "repair_source": str(repair_source),
            "params_source": params_source,
            "cells_filled_from_repaired_source": summary["cells_filled_from_repaired_source"],
            "bot_source_backup": str(backups["bot_source_backup"]) if backups["bot_source_backup"] else None,
            "output_backup": str(backups["output_backup"]) if backups["output_backup"] else None,
        },
    )
    final_coverage_starts = resolve_coverage_starts(
        updated,
        required_tickers=active_tickers,
        onboard_date_by_ticker=active_universe,
    )
    published_metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    confirmed_unavailable = published_metadata.get("confirmed_unavailable_by_ticker", {})
    final_coverage = validate_price_coverage(
        updated,
        required_tickers=active_tickers,
        freq="1d",
        expected_start_by_ticker=final_coverage_starts,
        allowed_missing_by_ticker=confirmed_unavailable,
    )
    print(
        json.dumps(
            {
                "published": str(output),
                "shape": list(updated.shape),
                "coverage": final_coverage.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
