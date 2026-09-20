from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_store/crypto_momentum_research/complete_results"
FIT_ASSETS = 581
PORTFOLIO_ASSETS = 97


def config_id(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]


def main():
    rows = json.loads((OUT / "tested_configurations.json").read_text(encoding="utf-8"))
    for row in rows:
        config = row["config"]
        if "fit_universe_assets" in config:
            config["fit_universe_assets"] = FIT_ASSETS
        if "portfolio_assets" in config:
            config["portfolio_assets"] = PORTFOLIO_ASSETS
        if config.get("model") in {"ts_legacy_optuna", "breakout_legacy_optuna"}:
            config["valid_optuna_tickers"] = 64
            config["fallback_equal_tickers"] = 33
            config["optimization_scope"] = "legacy_static_full_sample_reference"
        row["config_id"] = config_id(config)
    rows = list({row["config_id"]: row for row in rows}.values())
    families = sorted({row["config"].get("family") for row in rows})
    selected = []
    for family in families:
        candidates = [row for row in rows if row["config"].get("family") == family
                      and row["metrics"]["validation"]["net_sharpe"] is not None
                      and row["config"].get("taker_share", 1.) == 1.
                      and row["config"].get("slippage_bps", 5.) == 5.
                      and row["config"].get("activation", "next_open") == "next_open"
                      and not str(row["config"].get("optimization_scope", "")).startswith("legacy_static")]
        if candidates:
            selected.append(max(candidates, key=lambda row: row["metrics"]["validation"]["net_sharpe"]))
    manifest = json.loads((OUT / "study_manifest.json").read_text(encoding="utf-8"))
    manifest["fit_universe_assets"] = FIT_ASSETS
    manifest["portfolio_assets"] = PORTFOLIO_ASSETS
    manifest["configuration_count"] = len(rows)
    manifest["metadata_correction"] = (
        "asset counts use column counts; legacy Optuna coverage explicitly records 64 valid and 33 equal fallbacks; "
        "performance arrays unchanged"
    )
    (OUT / "tested_configurations.json").write_text(json.dumps(rows, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    (OUT / "selected_configurations.json").write_text(json.dumps(selected, indent=2, allow_nan=False), encoding="utf-8")
    deployable = {row["config"]["family"]: {"config_id": row["config_id"], **row["config"]} for row in selected}
    (OUT / "deployable_configurations.json").write_text(json.dumps(deployable, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    flat = []
    for row in rows:
        record = {"config_id": row["config_id"], **row["config"]}
        for split, values in row["metrics"].items():
            record.update({f"{split}_{key}": value for key, value in values.items()})
        flat.append(record)
    pd.DataFrame(flat).to_parquet(OUT / "tested_configurations.parquet", index=False)

    canonical = OUT / "crypto_momentum_research.sqlite"
    backup = OUT / "crypto_momentum_research.pre_metadata_fix.sqlite"
    if canonical.exists() and not backup.exists():
        canonical.replace(backup)
    elif canonical.exists():
        canonical.unlink()
    run_id = hashlib.sha256(json.dumps({"generated_at": manifest["generated_at"], "inputs": manifest["input_hashes"],
                                       "metadata_corrected": True}, sort_keys=True).encode()).hexdigest()[:24]
    daily = pd.read_parquet(OUT / "default_daily_returns.parquet")
    funding = pd.read_parquet(OUT / "default_daily_funding.parquet")
    with sqlite3.connect(canonical) as connection:
        pd.DataFrame([{"run_id": run_id, "generated_at": manifest["generated_at"],
                       "manifest_json": json.dumps(manifest, sort_keys=True)}]).to_sql("study_runs", connection, if_exists="append", index=False)
        pd.DataFrame([{"run_id": run_id, "config_id": row["config_id"],
                       "config_json": json.dumps(row["config"], sort_keys=True),
                       "metrics_json": json.dumps(row["metrics"], sort_keys=True)} for row in rows]).to_sql(
            "tested_configurations", connection, if_exists="append", index=False)
        daily_long = daily.rename_axis("timestamp").stack().rename("net_return").reset_index()
        daily_long.insert(0, "run_id", run_id); daily_long.to_sql("daily_returns", connection, if_exists="append", index=False)
        funding_long = funding.rename_axis("timestamp").stack().rename("funding_return").reset_index()
        funding_long.insert(0, "run_id", run_id); funding_long.to_sql("daily_funding", connection, if_exists="append", index=False)
    print(json.dumps({"configuration_count": len(rows), "fit_assets": FIT_ASSETS, "portfolio_assets": PORTFOLIO_ASSETS,
                      "selected_families": [row["config"]["family"] for row in selected]}, indent=2))


if __name__ == "__main__":
    main()
