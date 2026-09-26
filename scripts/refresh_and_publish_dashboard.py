from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh XSec20 market data, rebuild the runtime snapshot and publish it to the dashboard."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model-universe-only", action="store_true")
    parser.add_argument("--skip-funding", action="store_true")
    parser.add_argument("--refresh-universe", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    refresh_command = [sys.executable, str(ROOT / "scripts/refresh_xsec20_snapshot.py")]
    for flag in ("model_universe_only", "skip_funding", "refresh_universe"):
        if getattr(args, flag):
            refresh_command.append(f"--{flag.replace('_', '-')}")
    subprocess.run(refresh_command, cwd=ROOT, check=True)

    publish_command = [
        sys.executable,
        str(ROOT / "scripts/publish_dashboard_snapshot.py"),
        "--publish",
        "--base-url",
        args.base_url,
        "--env-file",
        str(args.env_file.resolve()),
    ]
    if args.overwrite_existing:
        publish_command.append("--overwrite-existing")
    subprocess.run(publish_command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
