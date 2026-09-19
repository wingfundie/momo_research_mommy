from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS universe_snapshots (
  as_of TEXT NOT NULL, symbol TEXT NOT NULL, provider TEXT NOT NULL, fdv REAL, rank INTEGER,
  eligible INTEGER NOT NULL, member INTEGER NOT NULL, reasons_json TEXT NOT NULL,
  source_timestamp TEXT, snapshot_hash TEXT NOT NULL, point_in_time INTEGER NOT NULL,
  inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(as_of, symbol, snapshot_hash)
);
CREATE TABLE IF NOT EXISTS signal_records (
  timestamp TEXT NOT NULL, symbol TEXT NOT NULL, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
  data_hash TEXT NOT NULL, config_hash TEXT NOT NULL, component_json TEXT NOT NULL,
  combined_forecast REAL, cross_sectional_percentile REAL, direction TEXT NOT NULL,
  quality_flags_json TEXT NOT NULL, target_quantity REAL, target_notional REAL,
  expected_funding_next REAL, expected_funding_24h REAL, inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(timestamp, symbol, model_name, model_version, data_hash, config_hash)
);
CREATE TABLE IF NOT EXISTS funding_events (
  symbol TEXT NOT NULL, funding_time TEXT NOT NULL, funding_rate REAL NOT NULL, mark_price REAL NOT NULL,
  position_quantity REAL NOT NULL, funding_cost_usd REAL NOT NULL, funding_cashflow_usd REAL NOT NULL,
  source TEXT NOT NULL, rate_type TEXT NOT NULL, quality_flags_json TEXT NOT NULL,
  inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(symbol, funding_time, source)
);
CREATE TABLE IF NOT EXISTS actual_funding (
  transaction_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, asset TEXT NOT NULL, timestamp TEXT NOT NULL,
  income_usd REAL NOT NULL, matched_model TEXT, matched_notional REAL NOT NULL,
  unmatched_notional REAL NOT NULL, unmatched_reason TEXT, source TEXT NOT NULL,
  inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS portfolio_runs (
  run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, sleeve TEXT NOT NULL, model_version TEXT NOT NULL,
  refit_schedule TEXT NOT NULL, risk_scenario_json TEXT NOT NULL, input_hashes_json TEXT NOT NULL,
  config_json TEXT NOT NULL, provider_coverage_json TEXT NOT NULL, status TEXT NOT NULL,
  metrics_json TEXT NOT NULL, manifest_json TEXT NOT NULL, inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS daily_portfolio (
  run_id TEXT NOT NULL, timestamp TEXT NOT NULL, gross_return REAL, net_return REAL, equity REAL,
  fees REAL, slippage REAL, funding_cashflow REAL, turnover REAL,
  PRIMARY KEY(run_id, timestamp), FOREIGN KEY(run_id) REFERENCES portfolio_runs(run_id)
);
"""


class ResearchLedger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def append_universe(self, records: Iterable) -> None:
        rows = [(r.as_of.isoformat(), r.symbol, r.provider, r.fully_diluted_valuation, r.rank,
                 int(r.eligible), int(r.member), json.dumps(r.reasons),
                 r.source_timestamp.isoformat() if r.source_timestamp else None, r.snapshot_hash,
                 int(r.point_in_time)) for r in records]
        self.connection.executemany("INSERT OR IGNORE INTO universe_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)", rows)
        self.connection.commit()

    def append_signals(self, records: Iterable) -> None:
        rows = [(r.timestamp.isoformat(), r.symbol, r.model_name, r.model_version, r.data_hash, r.config_hash,
                 json.dumps(dict(r.component_forecasts), sort_keys=True), r.combined_forecast,
                 r.cross_sectional_percentile, r.direction, json.dumps(r.quality_flags), r.target_quantity,
                 r.target_notional, r.expected_funding_next, r.expected_funding_24h) for r in records]
        self.connection.executemany("INSERT OR IGNORE INTO signal_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)", rows)
        self.connection.commit()

    def append_funding(self, records: Iterable) -> None:
        rows = [(r.symbol, r.funding_time.isoformat(), r.funding_rate, r.mark_price, r.position_quantity,
                 r.funding_cost_usd, r.funding_cashflow_usd, r.source, r.rate_type,
                 json.dumps(r.quality_flags)) for r in records]
        self.connection.executemany("INSERT OR IGNORE INTO funding_events VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)", rows)
        self.connection.commit()

    def append_actual_funding(self, records: Iterable) -> None:
        rows = [(r.transaction_id, r.symbol, r.asset, r.timestamp.isoformat(), r.income_usd, r.matched_model,
                 r.matched_notional, r.unmatched_notional, r.unmatched_reason, r.source) for r in records]
        self.connection.executemany("INSERT OR IGNORE INTO actual_funding VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)", rows)
        self.connection.commit()

    def append_run(self, run, daily: pd.DataFrame, manifest: dict) -> None:
        row = (run.run_id, run.created_at.isoformat(), run.sleeve, run.model_version, run.refit_schedule,
               _json(run.risk_scenario), _json(run.input_hashes), _json(run.config),
               _json(run.provider_coverage), run.status, _json(run.metrics), _json(manifest))
        self.connection.execute("INSERT OR IGNORE INTO portfolio_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)", row)
        daily_rows = [(run.run_id, pd.Timestamp(i).isoformat(), *[None if pd.isna(v) else float(v) for v in values])
                      for i, values in daily[["gross_return", "net_return", "equity", "fees", "slippage",
                                             "funding_cashflow", "turnover"]].iterrows()]
        self.connection.executemany("INSERT OR IGNORE INTO daily_portfolio VALUES (?,?,?,?,?,?,?,?,?)", daily_rows)
        self.connection.commit()

    def read(self, table: str) -> pd.DataFrame:
        if table not in {"universe_snapshots", "signal_records", "funding_events", "actual_funding",
                         "portfolio_runs", "daily_portfolio"}:
            raise ValueError("Unsupported ledger table")
        return pd.read_sql_query(f"SELECT * FROM {table}", self.connection)

    def export(self, directory: Path | str) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        for table in ("universe_snapshots", "signal_records", "funding_events", "actual_funding",
                      "portfolio_runs", "daily_portfolio"):
            frame = self.read(table)
            frame.to_csv(destination / f"{table}.csv", index=False)
            try:
                frame.to_parquet(destination / f"{table}.parquet", index=False)
            except ImportError:
                pass


def _json(value) -> str:
    if is_dataclass(value): value = asdict(value)
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
