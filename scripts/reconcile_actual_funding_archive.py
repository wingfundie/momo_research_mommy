from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"
OUT = ROOT / "data_store/crypto_momentum_research/complete_results"


def main():
    frames = []
    archive_path = INPUTS / "actual_funding_income_archive_raw.parquet"
    if archive_path.exists():
        raw = pd.read_parquet(archive_path)
        frames.append(pd.DataFrame({"transaction_id": raw["Transaction ID"].astype(str), "symbol": raw["Symbol"].astype(str),
                                    "asset": raw["Asset"].astype(str), "timestamp": pd.to_datetime(raw["Date(UTC)"], utc=True),
                                    "income_usd": pd.to_numeric(raw["Amount"], errors="coerce"), "archive_source": "binance_async"}))
    recent_path = INPUTS / "actual_funding_income_recent.parquet"
    if recent_path.exists():
        recent = pd.read_parquet(recent_path).copy()
        recent["transaction_id"] = recent["transaction_id"].astype(str)
        recent["archive_source"] = "binance_income_api_recent"
        frames.append(recent[["transaction_id", "symbol", "asset", "timestamp", "income_usd", "archive_source"]])
    actual = pd.concat(frames, ignore_index=True)
    actual["timestamp"] = pd.to_datetime(actual["timestamp"], utc=True)
    actual = actual.dropna(subset=["timestamp", "income_usd"]).drop_duplicates("transaction_id", keep="last").sort_values("timestamp")
    actual.to_parquet(INPUTS / "actual_funding_income_all.parquet", index=False)

    public = pd.read_parquet(INPUTS / "funding_events.parquet")
    public["timestamp"] = pd.to_datetime(public["funding_time"], utc=True)
    public = public[["symbol", "timestamp", "funding_rate", "mark_price"]].drop_duplicates(["symbol", "timestamp"])
    reconciled = actual.merge(public, on=["symbol", "timestamp"], how="left")
    reconciled["effective_day"] = reconciled.timestamp.dt.tz_localize(None).dt.normalize()
    reconciled.loc[reconciled.timestamp.dt.hour.eq(0), "effective_day"] -= pd.Timedelta(days=1)
    signals = pd.read_parquet(OUT / "daily_signal_records.parquet")
    signals = signals[signals.model.eq("ts_shrink_80_primary_quarterly_vol90")][["timestamp", "symbol", "target_position"]]
    signals = signals.rename(columns={"timestamp": "effective_day", "target_position": "model_position_weight"})
    reconciled = reconciled.merge(signals, on=["effective_day", "symbol"], how="left")
    reconciled["modeled_funding_return"] = -reconciled.model_position_weight * reconciled.funding_rate
    reconciled["actual_implied_notional_usd"] = np.divide(reconciled.income_usd.abs(), reconciled.funding_rate.abs(),
                                                           out=np.full(len(reconciled), np.nan), where=reconciled.funding_rate.abs().gt(0))
    agrees = np.sign(reconciled.income_usd) == np.sign(reconciled.modeled_funding_return)
    has_model = reconciled.model_position_weight.fillna(0).ne(0)
    reconciled["match_status"] = np.where(~reconciled.funding_rate.notna(), "unmatched_missing_public_rate",
                                           np.where(~has_model, "unmatched_no_model_exposure",
                                                    np.where(agrees, "direction_agrees_actual_notional_reconstructed", "unmatched_direction_disagrees")))
    reconciled["unmatched_reason"] = np.where(reconciled.match_status.str.startswith("unmatched"), reconciled.match_status, None)
    reconciled.to_parquet(OUT / "actual_funding_reconciliation.parquet", index=False)
    reconciled.groupby(["archive_source", "symbol", "match_status"], dropna=False)["income_usd"].agg(["count", "sum"]).reset_index().to_csv(
        OUT / "actual_funding_summary.csv", index=False)
    print({"events": len(reconciled), "net_usd": float(reconciled.income_usd.sum()),
           "start": str(reconciled.timestamp.min()), "end": str(reconciled.timestamp.max()),
           "direction_agrees": int(reconciled.match_status.str.startswith("direction_agrees").sum())})


if __name__ == "__main__":
    main()
