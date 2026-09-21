from __future__ import annotations

import argparse
from pathlib import Path

from momo_bot.exchange import get_binance_client
from momo_bot.research.funding import import_income_csv, paginate_income_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive Binance FUNDING_FEE income")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data_store/actual_funding_income.csv"))
    args = parser.parse_args()
    frame = import_income_csv(args.csv) if args.csv else paginate_income_history(
        get_binance_client().futures_income_history, args.start, args.end
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        import pandas as pd
        frame = pd.concat([pd.read_csv(args.output), frame]).drop_duplicates("transaction_id", keep="last")
    frame.to_csv(args.output, index=False)
    print(f"Archived {len(frame)} funding income records to {args.output}")


if __name__ == "__main__": main()
