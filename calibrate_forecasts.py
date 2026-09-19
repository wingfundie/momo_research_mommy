from __future__ import annotations

"""Calibrate forecast scalars and diversification multipliers for the backtests.

This script inspects the configured price data and calculates:
- Per-rule scalars that target an absolute mean forecast of ~10.
- A diversification multiplier (DM) from forecast correlations or fixed values.

The output is printed to stdout and can be saved as JSON for reuse in
`reoptimize_all.py` via `--calibration-json`.

Examples:
  python calibrate_forecasts.py --mode both --oos-fraction 0 --output data_store/calibration.json
  python calibrate_forecasts.py --mode momentum --ewmac-dm-mode fixed --ewmac-dm-fixed 1.2
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import portfolio_strategy as cta
import reoptimize_all as ro
from momo_bot.config import settings


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stringify_ewmac_scalars(scalars: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """Convert EWMAC scalar keys to string form for JSON serialization."""
    return {str(k): float(v) for k, v in scalars.items()}


def _print_scalars(title: str, scalars: Dict[str, float]) -> None:
    """Print scalars in a deterministic order for quick inspection."""
    print(f"\n{title}")
    for key in sorted(scalars.keys()):
        print(f"  {key}: {scalars[key]:.4f}")


def _calibrate_momentum(args: argparse.Namespace) -> Dict[str, Any]:
    """Calibrate EWMAC scalars and DM for the momentum dataset.

    Args:
        args: Parsed CLI arguments that control scalar/DM modes and lookbacks.

    Returns:
        Dictionary payload with scalars, DM, and calibration metadata.

    Raises:
        FileNotFoundError: If momentum price data is missing.
        ValueError: If the dataset is empty or scalars cannot be computed.
    """
    momentum_path = ro._resolve_path(settings.momentum_price_data_path)
    if not momentum_path.exists():
        raise FileNotFoundError(f"Momentum price data not found: {momentum_path}")
    price_frame = pd.read_pickle(momentum_path).drop_duplicates()
    if price_frame.empty:
        raise ValueError("Momentum price data is empty.")

    base_ewmac_factors = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128)]
    data_frequency = "4h"
    ewmac_factors = ro._resolve_ewmac_factors(
        base_ewmac_factors,
        span_mode=args.ewmac_span_mode,
        data_frequency=data_frequency,
    )

    split_ts, _ = ro._split_train_test_index(
        price_frame.index,
        oos_fraction=args.oos_fraction,
        oos_periods=args.oos_periods or None,
    )
    train_frame = price_frame if split_ts is None else price_frame.loc[:split_ts].copy()

    if args.ewmac_scalar_mode == "calibrated":
        ewmac_scalars = ro._calibrate_ewmac_scalars(
            train_frame,
            ewmac_factors,
            vol_lookback=args.vol_lookback_momentum,
            fallback_scalars=cta.CARVER_EWMAC_FORECAST_SCALARS,
            base_factors=base_ewmac_factors,
        )
    else:
        ewmac_scalars = {
            actual: cta.CARVER_EWMAC_FORECAST_SCALARS.get(base, float("nan"))
            for base, actual in zip(base_ewmac_factors, ewmac_factors)
        }

    missing = [k for k, v in ewmac_scalars.items() if not pd.notnull(v)]
    if missing:
        raise ValueError(f"Missing EWMAC scalars for factors: {missing}")

    if args.ewmac_dm_mode == "correlation":
        if args.ewmac_dm_weighting != "equal":
            print("Note: Optuna-weighted DM requires optimized weights. Using equal weights for calibration.")
        corr_matrix = ro._compute_ewmac_corr_matrix(
            train_frame,
            ewmac_factors,
            scalars=ewmac_scalars,
            vol_lookback=args.vol_lookback_momentum,
        )
        weights = np.full(len(ewmac_factors), 1.0 / len(ewmac_factors))
        ewmac_dm = ro._forecast_diversification_multiplier(
            weights,
            corr_matrix,
            dm_cap=args.ewmac_dm_cap,
        )
    else:
        ewmac_dm = args.ewmac_dm_fixed

    return {
        "data_path": str(momentum_path),
        "data_frequency": data_frequency,
        "ewmac_factors": [str(f) for f in ewmac_factors],
        "ewmac_scalar_mode": args.ewmac_scalar_mode,
        "ewmac_scalars": _stringify_ewmac_scalars(ewmac_scalars),
        "ewmac_dm_mode": args.ewmac_dm_mode,
        "ewmac_dm_weighting": args.ewmac_dm_weighting,
        "ewmac_dm_value": float(ewmac_dm),
        "ewmac_dm_cap": float(args.ewmac_dm_cap),
    }


def _calibrate_breakout(args: argparse.Namespace) -> Dict[str, Any]:
    """Calibrate breakout scalars and DM for the breakout dataset.

    Args:
        args: Parsed CLI arguments that control scalar/DM modes.

    Returns:
        Dictionary payload with scalars, DM, and calibration metadata.

    Raises:
        FileNotFoundError: If breakout price data is missing.
        ValueError: If the dataset is empty or scalars cannot be computed.
    """
    if not settings.breakout_price_data_path:
        raise FileNotFoundError("Breakout price data not found; set MOMO_BREAKOUT_PRICE_DATA_PATH.")
    breakout_path = ro._resolve_path(settings.breakout_price_data_path)
    if not breakout_path.exists():
        raise FileNotFoundError(f"Breakout price data not found: {breakout_path}")
    price_frame = pd.read_pickle(breakout_path).drop_duplicates()
    if price_frame.empty:
        raise ValueError("Breakout price data is empty.")

    breakout_horizons = [10, 20, 40, 80, 160]
    data_frequency = "1d"

    split_ts, _ = ro._split_train_test_index(
        price_frame.index,
        oos_fraction=args.oos_fraction,
        oos_periods=args.oos_periods or None,
    )
    train_frame = price_frame if split_ts is None else price_frame.loc[:split_ts].copy()

    if args.breakout_scalar_mode == "calibrated":
        breakout_scalars = ro._calibrate_breakout_scalars(
            train_frame,
            breakout_horizons,
            fallback_scalars=cta.CARVER_BREAKOUT_FORECAST_SCALARS,
        )
    else:
        breakout_scalars = {
            h: cta.CARVER_BREAKOUT_FORECAST_SCALARS.get(h, float("nan")) for h in breakout_horizons
        }

    missing = [k for k, v in breakout_scalars.items() if not pd.notnull(v)]
    if missing:
        raise ValueError(f"Missing breakout scalars for horizons: {missing}")

    if args.breakout_dm_mode == "correlation":
        if args.breakout_dm_weighting != "equal":
            print("Note: Optuna-weighted DM requires optimized weights. Using equal weights for calibration.")
        corr_matrix = ro._compute_breakout_corr_matrix(
            train_frame,
            breakout_horizons,
            scalars=breakout_scalars,
        )
        weights = np.full(len(breakout_horizons), 1.0 / len(breakout_horizons))
        breakout_dm = ro._forecast_diversification_multiplier(
            weights,
            corr_matrix,
            dm_cap=args.breakout_dm_cap,
        )
    else:
        breakout_dm = args.breakout_dm_fixed

    return {
        "data_path": str(breakout_path),
        "data_frequency": data_frequency,
        "breakout_horizons": breakout_horizons,
        "breakout_scalar_mode": args.breakout_scalar_mode,
        "breakout_scalars": {str(k): float(v) for k, v in breakout_scalars.items()},
        "breakout_dm_mode": args.breakout_dm_mode,
        "breakout_dm_weighting": args.breakout_dm_weighting,
        "breakout_dm_value": float(breakout_dm),
        "breakout_dm_cap": float(args.breakout_dm_cap),
    }


def main() -> None:
    """CLI entrypoint for calibration-only runs."""
    parser = argparse.ArgumentParser(description="Calibrate scalars and diversification multipliers only.")
    parser.add_argument("--mode", choices=["momentum", "breakout", "both"], default="both")
    parser.add_argument("--oos-fraction", type=float, default=0.0)
    parser.add_argument("--oos-periods", type=int, default=0)
    parser.add_argument("--output", type=str, default="")

    parser.add_argument(
        "--ewmac-span-mode",
        choices=["native", "daily-equivalent"],
        default="native",
    )
    parser.add_argument(
        "--ewmac-scalar-mode",
        choices=["carver", "calibrated"],
        default="calibrated",
    )
    parser.add_argument("--vol-lookback-momentum", type=int, default=360)
    parser.add_argument(
        "--ewmac-dm-mode",
        choices=["fixed", "correlation"],
        default="correlation",
    )
    parser.add_argument(
        "--ewmac-dm-weighting",
        choices=["equal", "optuna"],
        default="equal",
    )
    parser.add_argument("--ewmac-dm-fixed", type=float, default=1.12)
    parser.add_argument("--ewmac-dm-cap", type=float, default=2.5)

    parser.add_argument(
        "--breakout-scalar-mode",
        choices=["carver", "calibrated"],
        default="calibrated",
    )
    parser.add_argument(
        "--breakout-dm-mode",
        choices=["fixed", "correlation"],
        default="correlation",
    )
    parser.add_argument(
        "--breakout-dm-weighting",
        choices=["equal", "optuna"],
        default="equal",
    )
    parser.add_argument("--breakout-dm-fixed", type=float, default=1.24)
    parser.add_argument("--breakout-dm-cap", type=float, default=2.5)
    args = parser.parse_args()

    payload: Dict[str, Any] = {"meta": {"oos_fraction": args.oos_fraction, "oos_periods": args.oos_periods}}

    if args.mode in ("momentum", "both"):
        momentum = _calibrate_momentum(args)
        payload["momentum"] = momentum
        _print_scalars("Momentum scalars", momentum["ewmac_scalars"])
        print(f"Momentum DM: {momentum['ewmac_dm_value']:.4f}")

    if args.mode in ("breakout", "both"):
        breakout = _calibrate_breakout(args)
        payload["breakout"] = breakout
        _print_scalars("Breakout scalars", breakout["breakout_scalars"])
        print(f"Breakout DM: {breakout['breakout_dm_value']:.4f}")

    if args.output:
        _write_json(Path(args.output), payload)
        print(f"\nSaved calibration to {args.output}")


if __name__ == "__main__":
    main()
