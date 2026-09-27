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
