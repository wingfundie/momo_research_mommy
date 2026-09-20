from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data_store/crypto_momentum_research/production_like_walkforward_v1"
LEGACY_MANIFEST = ROOT / "data_store/crypto_momentum_research/complete_results/study_manifest.json"
LEGACY_REPORT = ROOT / "reports/crypto_momentum_complete_study_20260920.html"
EXPECTED_LEGACY_HASHES = {
    LEGACY_MANIFEST: "0b80acfcf2a5ae91f2ec45334e916c31521b58f465d63b395e1b990bb40ada5c",
    LEGACY_REPORT: "c7b73229834802185e75306c68419138fd8060bba579fc37caf10a08a5c0d1b3",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((RESULTS / "study_manifest.json").read_text(encoding="utf-8"))
    candidates = pd.read_parquet(RESULTS / "candidate_score_ledger.parquet")
    selection = pd.read_parquet(RESULTS / "rolling_selection_ledger.parquet")
    selected = selection[selection.selected]
    state = pd.read_parquet(RESULTS / "daily_portfolio_state.parquet")
    positions = pd.read_parquet(RESULTS / "daily_ticker_positions.parquet")
    rolling = pd.read_parquet(RESULTS / "rolling_metrics.parquet", columns=["window_days"])

    assert len(candidates) == manifest["candidate_fold_scores"] == 892_800
    assert candidates.config_id.nunique() == manifest["tested_configuration_count"] == 178_560
    assert not candidates.duplicated(["variant", "fold", "family", "model", "config_id"]).any()
    assert len(selected) == manifest["selected_outer_fold_configurations"] == 60
    grouped = selected.groupby(["variant", "fold", "family"])
    assert grouped.size().eq(3).all() and grouped.model.nunique().eq(3).all()
    assert manifest["selection_uses_holdout_metrics"] is False
    coverage = manifest["funding_coverage"]
    assert coverage["events"] == coverage["finite_mark_prices"] + coverage["unresolved_pre_eligibility"]
    assert coverage["unresolved_post_eligibility"] == 0
    first_outer_year = pd.to_datetime(state.timestamp).dt.year.eq(2022)
    assert state.loc[first_outer_year, "funding"].abs().sum() > 0
    assert set(rolling.window_days) == {90, 365}
    assert not selected.model.str.contains("320|control|optuna|legacy|unsmoothed", case=False, regex=True).any()
    assert all(pd.Timestamp(fold["cutoff"]) < pd.Timestamp(fold["test_start"]) for fold in manifest["outer_folds"])

    aggregates = positions.groupby(["timestamp", "variant", "strategy"], sort=False).agg(
        contribution=("net_return_contribution", "sum"),
        gross=("held_weight", lambda values: values.abs().sum()),
        net=("held_weight", "sum"),
    ).reset_index()
    reconciled = state.merge(aggregates, on=["timestamp", "variant", "strategy"], validate="one_to_one")
    assert (reconciled.net_return - reconciled.contribution).abs().max() < 1e-12
    assert (reconciled.gross_exposure - reconciled.gross).abs().max() < 1e-12
    assert (reconciled.net_exposure - reconciled.net).abs().max() < 1e-12
    accounting = reconciled.gross_return - reconciled.fee - reconciled.slippage + reconciled.funding
    assert (reconciled.net_return - accounting).abs().max() < 1e-12

    catalog = json.loads((RESULTS / "all_tested_configurations.json").read_text(encoding="utf-8"))
    deployable = json.loads((RESULTS / "selected_deployable_configurations.json").read_text(encoding="utf-8"))
    assert catalog["configuration_count"] == len(catalog["configurations"]) == 178_560
    assert len(deployable["configurations"]) == 60
    with sqlite3.connect(RESULTS / "production_shadow.sqlite") as connection:
        shadow_rows = connection.execute("SELECT COUNT(*) FROM emitted_signals").fetchone()[0]
        shadow_models = connection.execute("SELECT COUNT(DISTINCT model) FROM emitted_signals").fetchone()[0]
    assert shadow_rows == 582 and shadow_models == 6 and manifest["prospective_observations"] == 1
    for path, expected in EXPECTED_LEGACY_HASHES.items():
        assert digest(path) == expected, f"Legacy production artifact changed: {path}"
    assert manifest["production_signals_changed"] is False
    print("PASS: causal selection, configuration catalog, accounting, shadow ledger, and production baseline reconcile.")


if __name__ == "__main__":
    main()
