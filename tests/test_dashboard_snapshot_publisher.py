from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.publish_dashboard_snapshot import build_bundle


def test_build_bundle_validates_and_packages_runtime(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    runtime = root / "data_store/xsec20_runtime"
    runtime.mkdir(parents=True)
    manifest = {
        "schema_version": 3,
        "model": "xsm_ic20_dollar_neutral_vol60",
        "config_id": "config-1",
        "generated_at": "2026-09-22T02:53:26+00:00",
        "data_cutoff": "2026-09-21T00:00:00",
        "gross_cap": 2.0,
    }
    (runtime / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-09-21")], "symbol": ["BTCUSDT"], "target_weight": [0.2]}
    ).to_parquet(runtime / "ticker_history.parquet", index=False)
    pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-09-21")], "net_return": [0.01], "gross_exposure": [0.2]}
    ).to_parquet(runtime / "portfolio_daily.parquet", index=False)
    pd.DataFrame({"symbol": ["BTCUSDT"], "standalone_sr": [1.2]}).to_parquet(
        runtime / "standalone_sr.parquet", index=False
    )

    bundle = build_bundle(root=root, output_root=tmp_path / "exports", version="test-v1")

    bundle_manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert bundle_manifest["version"] == "test-v1"
    assert bundle_manifest["config_id"] == "config-1"
    assert len(bundle_manifest["files"]) == 4
    assert (bundle / "xsec20/ticker_history.parquet").is_file()


def test_build_bundle_creates_memory_bounded_research_extracts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    runtime = root / "data_store/xsec20_runtime"
    runtime.mkdir(parents=True)
    (runtime / "manifest.json").write_text(json.dumps({
        "schema_version": 3,
        "model": "xsm_ic20_dollar_neutral_vol60",
        "config_id": "config-1",
        "generated_at": "2026-09-22T02:53:26+00:00",
        "data_cutoff": "2026-09-21T00:00:00",
        "gross_cap": 2.0,
    }), encoding="utf-8")
    pd.DataFrame({"timestamp": [pd.Timestamp("2026-09-21")], "symbol": ["BTCUSDT"], "target_weight": [.2]}).to_parquet(runtime / "ticker_history.parquet", index=False)
    pd.DataFrame({"timestamp": [pd.Timestamp("2026-09-21")], "net_return": [.01], "gross_exposure": [.2]}).to_parquet(runtime / "portfolio_daily.parquet", index=False)
    pd.DataFrame({"symbol": ["BTCUSDT"], "standalone_sr": [1.2]}).to_parquet(runtime / "standalone_sr.parquet", index=False)
    research = root / "data_store/crypto_momentum_research/complete_results"
    research.mkdir(parents=True)
    pd.DataFrame([{
        "family": "time_series", "model": "ts", "target_vol": .15,
        "gross_cap": 2, "ticker_risk_cap": .25, "rebalance": "daily",
        "taker_share": 1, "slippage_bps": 5, "validation_net_sharpe": 1.2,
        "holdout_2026_net_sharpe": .8, "validation_max_drawdown": -.1,
    }]).to_parquet(research / "tested_configurations.parquet", index=False)
    pd.DataFrame([{"timestamp": "2026-09-21", "symbol": "BTCUSDT", "income_usd": 1.0, "match_status": "matched"}]).to_parquet(research / "actual_funding_reconciliation.parquet", index=False)
    pd.DataFrame([{"model": "ts", "symbol": "BTCUSDT", "position_weight": .1, "reference_funding_rate": .0001, "next_settlement_expected_return": -.00001, "next_24h_expected_return": -.00003}]).to_csv(research / "expected_funding_snapshot.csv", index=False)
    pd.DataFrame([{"period": "ALL", "model": "ts", "paid": -.1, "received": .2, "net": .1}]).to_csv(research / "modeled_funding_summary.csv", index=False)

    bundle = build_bundle(root=root, output_root=tmp_path / "exports", version="test-v2")
    complete = bundle / "research/complete"
    summary = json.loads((complete / "research_summary.json").read_text(encoding="utf-8"))
    assert summary["tested_configurations"] == 1
    assert len(pd.read_parquet(complete / "headline_risk_grid.parquet")) == 1
    assert len(pd.read_parquet(complete / "actual_funding_reconciliation_head.parquet")) == 1
    assert len(pd.read_csv(complete / "expected_funding_snapshot_head.csv")) == 1
    assert len(pd.read_csv(complete / "modeled_funding_all.csv")) == 1
