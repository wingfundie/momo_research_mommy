from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data_store/crypto_momentum_research/complete_results"
OUTPUT = ROOT / "reports/crypto_momentum_telegram_summary.txt"


def build_summary() -> str:
    signals = pd.read_csv(RESULTS / "latest_signal_records.csv")
    funding = pd.read_parquet(RESULTS / "actual_funding_reconciliation.parquet")
    expected = pd.read_csv(RESULTS / "expected_funding_snapshot.csv")
    primary = signals[signals.model.eq("ts_shrink_80_primary_quarterly_vol90")].dropna(subset=["forecast"])
    strongest = primary.nlargest(5, "forecast")[["symbol", "forecast"]]
    weakest = primary.nsmallest(5, "forecast")[["symbol", "forecast"]]
    expected_primary = expected[expected.model.eq("ts_shrink_80_primary_quarterly_vol90")]
    lines = ["CRYPTO MOMENTUM · SHADOW", f"Signal cutoff: {primary.timestamp.max()}", "",
             "Strongest: " + ", ".join(f"{row.symbol} {row.forecast:+.1f}" for row in strongest.itertuples()),
             "Weakest: " + ", ".join(f"{row.symbol} {row.forecast:+.1f}" for row in weakest.itertuples()), "",
             f"Actual archived funding: ${funding.income_usd.sum():,.2f} across {len(funding):,} events",
             f"Expected next settlement (primary): {expected_primary.next_settlement_expected_return.sum():+.3%}",
             f"Expected next 24h (primary): {expected_primary.next_24h_expected_return.sum():+.3%}", "",
             "Research only · current-universe fallback · no production orders"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="Send using TELEGRAM_BOT_TOKEN and MOMO_TELEGRAM_CHAT_ID")
    args = parser.parse_args()
    load_dotenv()
    text = build_summary()
    OUTPUT.write_text(text, encoding="utf-8")
    if args.send:
        token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("MOMO_TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise RuntimeError("TELEGRAM_BOT_TOKEN and MOMO_TELEGRAM_CHAT_ID are required for --send")
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=30)
        response.raise_for_status()
    print(OUTPUT)


if __name__ == "__main__":
    main()
