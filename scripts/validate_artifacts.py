from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.config import settings
from momo_bot.data_validation import validate_price_frame


def _validate_price(path: Path, configured_frequency: str, strict: bool) -> bool:
    if not path.exists():
        print(f"MISSING {path}")
        return False
    df = pd.read_pickle(path)
    result = validate_price_frame(
        df,
        path=path,
        configured_frequency=configured_frequency,
        strict=strict,
    )
    print(
        f"OK {path} rows={result.rows} cols={result.columns} "
        f"inferred={result.inferred_frequency} effective={result.effective_frequency} "
        f"missing_pct={result.missing_cell_pct:.3f}"
    )
    for warning in result.warnings:
        print(f"WARN {path}: {warning}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local model artifacts.")
    parser.add_argument("--all", action="store_true", help="Validate all configured artifacts.")
    parser.add_argument("--strict", action="store_true", default=settings.strict_data_frequency)
    args = parser.parse_args()

    ok = True
    ok &= _validate_price(settings.momentum_price_data_path, settings.momentum_data_frequency, args.strict)
    if settings.breakout_price_data_path:
        ok &= _validate_price(settings.breakout_price_data_path, settings.breakout_data_frequency, args.strict)

    for path in [settings.momentum_params_path, settings.breakout_params_path, settings.breakout_pairs_params_path]:
        if path is None:
            continue
        if path.exists():
            print(f"OK {path}")
        else:
            print(f"MISSING {path}")
            ok = False

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
