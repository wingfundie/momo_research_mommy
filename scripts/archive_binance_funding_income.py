from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.exchange import get_binance_client

INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"
STATE = INPUTS / "actual_funding_async_jobs.json"


def parse_download(content: bytes) -> pd.DataFrame:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
        frames = [pd.read_csv(archive.open(name)) for name in archive.namelist() if name.lower().endswith(".csv")]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except zipfile.BadZipFile:
        return pd.read_csv(io.BytesIO(content))


def main():
    INPUTS.mkdir(parents=True, exist_ok=True)
    client = get_binance_client()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    now = pd.Timestamp.now(tz="UTC")
    new_jobs = 0
    max_new_jobs = int(os.getenv("MOMO_MAX_NEW_ASYNC_JOBS", "1"))
    for year in range(2019, now.year + 1):
        key = str(year)
        if key not in state:
            if new_jobs >= max_new_jobs:
                break
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = min(pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC"), now)
            response = client.futures_v1_get_income_asyn(startTime=int(start.timestamp() * 1000), endTime=int(end.timestamp() * 1000))
            state[key] = {"download_id": str(response["downloadId"]), "status": "requested", "start": start.isoformat(), "end": end.isoformat()}
            STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
            new_jobs += 1
            if new_jobs >= max_new_jobs:
                break
    raw_history_path = INPUTS / "actual_income_history_raw.parquet"
    frames = [pd.read_parquet(raw_history_path)] if raw_history_path.exists() else []
    max_polls = int(os.getenv("MOMO_MAX_ASYNC_POLLS", "20"))
    polls = 0
    for year, job in state.items():
        if job.get("downloaded"):
            continue
        if polls >= max_polls:
            break
        response = client.futures_v1_get_income_asyn_id(downloadId=job["download_id"])
        polls += 1
        job["status"] = response.get("status")
        job["expiration_timestamp"] = response.get("expirationTimestamp")
        url = response.get("url") or response.get("s3Link")
        if url and response.get("status") == "completed":
            verify = os.getenv("MOMO_PROVIDER_VERIFY_SSL", "false").lower() in {"1", "true", "yes"}
            frame = parse_download(requests.get(url, timeout=120, verify=verify).content)
            frame["archive_year"] = int(year)
            frames.append(frame)
            job["downloaded"] = True
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if frames:
        raw = pd.concat(frames, ignore_index=True).drop_duplicates()
        raw.to_parquet(raw_history_path, index=False)
        lowered = {column.lower().replace(" ", "").replace("_", ""): column for column in raw.columns}
        income_type = lowered.get("incometype") or lowered.get("type")
        if income_type:
            raw = raw[raw[income_type].astype(str).str.upper().eq("FUNDING_FEE")]
        raw.to_parquet(INPUTS / "actual_funding_income_archive_raw.parquet", index=False)
    print(json.dumps({year: job["status"] for year, job in state.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
