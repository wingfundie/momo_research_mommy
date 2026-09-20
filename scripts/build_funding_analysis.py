from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.execute_full_crypto_study import funding_coefficients

INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"
OUT = ROOT / "data_store/crypto_momentum_research/complete_results"


def main():
    prices = pd.read_parquet(INPUTS / "research_prices.parquet")
    signals = pd.read_parquet(OUT / "daily_signal_records.parquet")
    symbols = sorted(signals.symbol.unique())
    prices = prices.reindex(columns=symbols)
    events = pd.read_parquet(INPUTS / "funding_events.parquet")
    same, midnight = funding_coefficients(prices, events)
    rows = []
    for model, model_signals in signals.groupby("model", sort=False):
        positions = model_signals.pivot(index="timestamp", columns="symbol", values="target_position").reindex(index=prices.index, columns=symbols).fillna(0)
        matrix = positions * same + positions.shift(1).fillna(0) * midnight
        long = matrix.stack().rename("funding_return").reset_index()
        long.columns = ["timestamp", "symbol", "funding_return"]
        position_long = positions.stack().rename("position_weight").reset_index(drop=True)
        long["position_weight"] = position_long
        long["side"] = np.where(long.position_weight > 0, "long", np.where(long.position_weight < 0, "short", "flat"))
        long.insert(0, "model", model)
        rows.append(long[long.funding_return.ne(0)])
    detail = pd.concat(rows, ignore_index=True)
    detail["period"] = pd.to_datetime(detail.timestamp).dt.to_period("Y").astype(str)
    detail.to_parquet(OUT / "modeled_funding_detail.parquet", index=False)
    grouped = detail.groupby(["model", "symbol", "side", "period"], dropna=False).funding_return.agg(
        paid=lambda x: x[x < 0].sum(), received=lambda x: x[x > 0].sum(), net="sum", events="count").reset_index()
    totals = detail.groupby(["model", "symbol", "side"], dropna=False).funding_return.agg(
        paid=lambda x: x[x < 0].sum(), received=lambda x: x[x > 0].sum(), net="sum", events="count").reset_index()
    totals["period"] = "ALL"
    summary = pd.concat([grouped, totals], ignore_index=True)
    summary.to_csv(OUT / "modeled_funding_summary.csv", index=False)
    expected = pd.read_parquet(OUT / "default_daily_funding.parquet")[detail.model.unique()].sum()
    actual = detail.groupby("model").funding_return.sum()
    if not np.allclose(expected.reindex(actual.index), actual, atol=1e-12):
        raise AssertionError("Ticker funding detail does not reconcile to portfolio funding")
    print({"detail_rows": len(detail), "summary_rows": len(summary), "models": detail.model.nunique()})


if __name__ == "__main__":
    main()
