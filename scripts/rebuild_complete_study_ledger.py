from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_store/crypto_momentum_research/complete_results"
INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"


def main():
    manifest = json.loads((OUT / "study_manifest.json").read_text(encoding="utf-8"))
    run_id = hashlib.sha256(json.dumps({"generated_at": manifest["generated_at"], "inputs": manifest["input_hashes"]}, sort_keys=True).encode()).hexdigest()[:24]
    canonical = OUT / "crypto_momentum_research.sqlite"
    archive = OUT / "crypto_momentum_research.pre_open_execution_fix.sqlite"
    if canonical.exists() and not archive.exists():
        canonical.replace(archive)
    elif canonical.exists():
        canonical.unlink()
    configs = json.loads((OUT / "tested_configurations.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(OUT / "default_daily_returns.parquet")
    funding = pd.read_parquet(OUT / "default_daily_funding.parquet")
    signals = pd.read_parquet(OUT / "daily_signal_records.parquet")
    ticker = pd.read_parquet(OUT / "ticker_attribution.parquet")
    groups = pd.read_parquet(OUT / "group_attribution.parquet")
    actual = pd.read_parquet(OUT / "actual_funding_reconciliation.parquet")
    expected = pd.read_csv(OUT / "expected_funding_snapshot.csv")
    modeled_funding = pd.read_csv(OUT / "modeled_funding_summary.csv")
    universe = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    with sqlite3.connect(canonical) as connection:
        pd.DataFrame([{"run_id": run_id, "generated_at": manifest["generated_at"], "manifest_json": json.dumps(manifest, sort_keys=True)}]).to_sql("study_runs", connection, if_exists="append", index=False)
        pd.DataFrame([{"run_id": run_id, "config_id": row["config_id"], "config_json": json.dumps(row["config"], sort_keys=True), "metrics_json": json.dumps(row["metrics"], sort_keys=True)} for row in configs]).to_sql("tested_configurations", connection, if_exists="append", index=False, chunksize=5000)
        for name, frame, value in (("daily_returns", daily, "net_return"), ("daily_funding", funding, "funding_return")):
            long = frame.rename_axis("timestamp").stack().rename(value).reset_index(); long.insert(0, "run_id", run_id)
            long.to_sql(name, connection, if_exists="append", index=False, chunksize=10000)
        for name, frame in (("signal_records", signals), ("ticker_attribution", ticker), ("group_attribution", groups),
                            ("actual_funding", actual), ("expected_funding", expected), ("modeled_funding_summary", modeled_funding),
                            ("universe_snapshots", universe)):
            frame = frame.copy(); frame.insert(0, "run_id", run_id)
            frame.to_sql(name, connection, if_exists="append", index=False, chunksize=10000)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_configs_run_id ON tested_configurations(run_id, config_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_signals_model_time ON signal_records(model, timestamp)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signal_records(symbol, timestamp)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_actual_symbol_time ON actual_funding(symbol, timestamp)")
    print({"run_id": run_id, "signals": len(signals), "configs": len(configs), "ledger": str(canonical)})


if __name__ == "__main__":
    main()
