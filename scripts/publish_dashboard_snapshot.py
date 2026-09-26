from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    import truststore
except ImportError:  # Optional on platforms that already use the system CA store.
    truststore = None


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REQUIRED = (
    "manifest.json",
    "ticker_history.parquet",
    "portfolio_daily.parquet",
    "standalone_sr.parquet",
)
RESEARCH_FILES = (
    "tested_configurations.parquet",
    "default_daily_returns.parquet",
    "default_daily_funding.parquet",
    "modeled_funding_summary.csv",
    "latest_signal_records.csv",
    "group_attribution.parquet",
    "ticker_attribution.parquet",
    "capacity_analysis.csv",
    "actual_funding_reconciliation.parquet",
    "expected_funding_snapshot.csv",
    "selected_configurations.json",
    "study_manifest.json",
)
WALKFORWARD_FILES = (
    "study_manifest.json",
    "methodology_config.json",
    "configurations/selected_deployable_configurations.json",
    "configurations/selected_outer_fold_rosters.json",
    "metrics/rolling_selection_ledger.csv",
    "metrics/outer_fold_metrics.csv",
    "metrics/stitched_oos_metrics.csv",
)
RESEARCH_INPUT_FILES = (
    "symbol_mappings.csv",
    "coingecko_asset_metadata.json",
)
CHUNK_SIZE = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _validate_runtime(runtime: Path) -> dict[str, object]:
    missing = [name for name in RUNTIME_REQUIRED if not (runtime / name).is_file()]
    if missing:
        raise ValueError(f"Missing XSec20 runtime files: {', '.join(missing)}")
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise ValueError(f"Unsupported XSec20 schema: {manifest.get('schema_version')}")
    ticker = pd.read_parquet(runtime / "ticker_history.parquet", columns=["timestamp", "symbol", "target_weight"])
    portfolio = pd.read_parquet(runtime / "portfolio_daily.parquet", columns=["timestamp", "net_return", "gross_exposure"])
    standalone = pd.read_parquet(runtime / "standalone_sr.parquet", columns=["symbol", "standalone_sr"])
    if ticker.empty or portfolio.empty or standalone.empty:
        raise ValueError("XSec20 runtime tables must be non-empty")
    if ticker[["timestamp", "symbol"]].duplicated().any():
        raise ValueError("Ticker history contains duplicate timestamp/symbol rows")
    latest = pd.to_datetime(ticker["timestamp"]).max()
    cutoff = pd.Timestamp(str(manifest["data_cutoff"]))
    if latest != cutoff:
        raise ValueError(f"Ticker cutoff {latest} does not match manifest {cutoff}")
    latest_weights = ticker.loc[pd.to_datetime(ticker["timestamp"]).eq(latest), "target_weight"]
    if latest_weights.abs().sum() > float(manifest["gross_cap"]) + 1e-6:
        raise ValueError("Latest target weights exceed the declared gross cap")
    return manifest


def build_bundle(*, root: Path = ROOT, output_root: Path | None = None, version: str | None = None) -> Path:
    runtime = root / "data_store/xsec20_runtime"
    runtime_manifest = _validate_runtime(runtime)
    generated = pd.Timestamp(str(runtime_manifest["generated_at"]))
    version = version or generated.strftime("%Y%m%dT%H%M%SZ")
    destination = (output_root or root / "data_store/dashboard_exports") / version
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for name in RUNTIME_REQUIRED:
        _copy(runtime / name, destination / "xsec20" / name)

    research = root / "data_store/crypto_momentum_research/complete_results"
    copied_research: list[str] = []
    for name in RESEARCH_FILES:
        source = research / name
        if source.is_file():
            _copy(source, destination / "research/complete" / name)
            copied_research.append(name)

    inputs = root / "data_store/crypto_momentum_research/inputs"
    copied_inputs: list[str] = []
    for name in RESEARCH_INPUT_FILES:
        source = inputs / name
        if source.is_file():
            _copy(source, destination / "research/inputs" / name)
            copied_inputs.append(name)

    walkforward = root / "reports/production_like_walkforward_v1"
    copied_walkforward: list[str] = []
    for name in WALKFORWARD_FILES:
        source = walkforward / name
        if source.is_file():
            _copy(source, destination / "research/walkforward" / name)
            copied_walkforward.append(name)

    file_rows = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            file_rows.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": runtime_manifest["model"],
        "config_id": runtime_manifest["config_id"],
        "data_cutoff": runtime_manifest["data_cutoff"],
        "runtime_schema_version": runtime_manifest["schema_version"],
        "research_files": copied_research,
        "research_input_files": copied_inputs,
        "walkforward_files": copied_walkforward,
        "files": file_rows,
    }
    (destination / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return destination


def _upload_file(session: requests.Session, base_url: str, source: Path, remote_path: str, overwrite: bool) -> None:
    url = f"{base_url}/api/storage/files/{quote(remote_path, safe='/')}"
    head = session.head(url, timeout=60)
    if head.status_code not in {200, 404}:
        head.raise_for_status()
    remote_size = int(head.headers.get("content-length", "0")) if head.status_code == 200 else 0
    local_size = source.stat().st_size
    if remote_size == local_size and not overwrite:
        return
    offset = 0 if overwrite or remote_size > local_size else remote_size
    with source.open("rb") as input_file:
        input_file.seek(offset)
        while offset < local_size:
            chunk = input_file.read(CHUNK_SIZE)
            params: dict[str, object] = {"offset": offset}
            if overwrite and offset == 0:
                params["overwrite"] = "true"
            for attempt in range(5):
                try:
                    response = session.put(url, params=params, data=chunk, timeout=180)
                    response.raise_for_status()
                    offset = int(response.json()["size"])
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
    if offset != local_size:
        raise RuntimeError(f"Upload size mismatch for {remote_path}: {offset} != {local_size}")


def publish_bundle(bundle: Path, *, base_url: str, token: str, overwrite: bool = False) -> dict[str, object]:
    if truststore is not None:
        truststore.inject_into_ssl()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    remote_root = f"systematic/crypto/momentum/versions/{bundle.name}"
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            remote_path = f"{remote_root}/{path.relative_to(bundle).as_posix()}"
            _upload_file(session, base_url.rstrip("/"), path, remote_path, overwrite)
    response = session.post(
        f"{base_url.rstrip('/')}/api/systematic/crypto/momentum/snapshots/activate",
        params={"version": bundle.name},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and optionally publish the dashboard momentum snapshot.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("MOMO_DASHBOARD_URL", ""))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv(args.root / ".env")
    bundle = build_bundle(root=args.root.resolve(), output_root=args.output_root, version=args.version)
    print(json.dumps({"bundle": str(bundle), "version": bundle.name}, indent=2))
    if args.publish:
        if not args.base_url:
            raise ValueError("--base-url or MOMO_DASHBOARD_URL is required for publishing")
        token = os.environ.get("RENDER_DATA_UPLOAD_TOKEN", "")
        if not token:
            raise ValueError("RENDER_DATA_UPLOAD_TOKEN is required for publishing")
        result = publish_bundle(
            bundle,
            base_url=args.base_url,
            token=token,
            overwrite=args.overwrite_existing,
        )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
