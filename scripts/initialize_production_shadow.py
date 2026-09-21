from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.execute_complete_crypto_study import VOL_WINDOWS, fast_risk_unit, load_full_price_panel
from scripts.execute_full_crypto_study import funding_coefficients
from scripts.execute_production_like_walkforward import (
    INPUTS, LOCKED, OUT, funding_tradability_mask, iter_breakout_models, iter_momentum_models, simulate_arrays,
)


def main() -> None:
    database = OUT / "production_shadow.sqlite"
    manifest_path = OUT / "study_manifest.json"
    if not database.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run execute_production_like_walkforward.py before initializing shadow signals")
    locked = json.loads(LOCKED.read_text(encoding="utf-8"))
    seed = locked["prospective_seed_roster"]
    seed_ids = {record["config_id"] for family in ("time_series_momentum", "breakout") for record in seed[family]}
    momentum = pd.read_parquet(ROOT / "data_store/crypto_momentum_research/complete_results/tested_configurations.parquet")
    breakout = pd.read_parquet(ROOT / "data_store/crypto_momentum_research/complete_results_carver5_v2/tested_configurations.parquet")
    configs = pd.concat([momentum[momentum.config_id.isin(seed_ids)], breakout[breakout.config_id.isin(seed_ids)]], ignore_index=True)
    if set(configs.config_id) != seed_ids:
        raise ValueError(f"Missing seed configurations: {sorted(seed_ids - set(configs.config_id))}")

    all_prices = load_full_price_panel()
    fit_prices = all_prices.loc[:, all_prices.notna().sum() >= 90]
    universe = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    members = universe.loc[universe.member.astype(str).str.lower().isin(["true", "1"]), "symbol"]
    symbols = [symbol for symbol in members if symbol in fit_prices]
    prices = fit_prices[symbols]
    opens = pd.read_parquet(INPUTS / "portfolio_open_prices.parquet").reindex(index=prices.index, columns=symbols)
    close_returns = prices.pct_change(fill_method=None).fillna(0)
    open_returns = opens.shift(-1).div(opens).sub(1).fillna(0)
    events = pd.read_parquet(INPUTS / "funding_events.parquet")
    events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True)
    same, midnight = funding_coefficients(prices, events)
    tradable = funding_tradability_mask(prices.index, symbols, events)
    commissions = pd.read_csv(INPUTS / "commission_rates.csv").set_index("symbol")
    maker = commissions["maker"].reindex(symbols).fillna(.0002)
    taker = commissions["taker"].reindex(symbols).fillna(.0004)
    volatility = {window: close_returns.rolling(window, min_periods=window).std() * (365 ** .5) for window in VOL_WINDOWS}

    model_names = set(configs.model)
    rows = []
    emitted_at = pd.Timestamp.now(tz="UTC").isoformat()
    momentum_names = set(configs.loc[configs.model.str.startswith("ts_"), "model"])
    breakout_names = model_names - momentum_names
    generators = [
        iter_momentum_models(fit_prices, prices, None, allowed_models=momentum_names),
        iter_breakout_models(fit_prices, prices, maker, taker, None, allowed_models=breakout_names),
    ]
    for generator in generators:
        for model in generator:
            if model.name not in model_names:
                continue
            config = configs[configs.model.eq(model.name)].iloc[0]
            risk_window = int(config.volatility_window)
            unit = fast_risk_unit(model.forecast, close_returns, volatility[risk_window], float(config.ticker_risk_cap), risk_window).where(tradable, 0.0)
            simulation = simulate_arrays(
                unit.to_numpy(float), prices.index, open_returns.to_numpy(float), same.to_numpy(float), midnight.to_numpy(float),
                maker.to_numpy(float), taker.to_numpy(float), target_vol=float(config.target_vol),
                gross_cap=float(config.gross_cap), rebalance=config.rebalance, detail=True, columns=prices.columns,
            )
            cutoff = prices.index.max()
            for symbol in symbols:
                rows.append({
                    "timestamp": cutoff.isoformat(), "family": model.family, "model": model.name,
                    "symbol": symbol, "forecast": float(model.forecast.loc[cutoff, symbol]) if pd.notna(model.forecast.loc[cutoff, symbol]) else None,
                    "target_weight": float(simulation.target.loc[cutoff, symbol]),
                    "held_weight": float(simulation.held.loc[cutoff, symbol]),
                    "config_hash": config.config_id, "data_cutoff": cutoff.isoformat(), "emitted_at": emitted_at,
                })
    if {row["model"] for row in rows} != model_names:
        raise RuntimeError(f"Not every seed model emitted: {sorted(model_names - {row['model'] for row in rows})}")
    emitted = pd.DataFrame(rows)
    with sqlite3.connect(database) as connection:
        existing = pd.read_sql_query("SELECT COUNT(*) AS observations FROM emitted_signals", connection).iloc[0, 0]
        if existing == 0:
            emitted.head(0).to_sql("emitted_signals", connection, if_exists="replace", index=False)
        else:
            connection.execute(
                """DELETE FROM emitted_signals
                   WHERE rowid NOT IN (
                       SELECT MIN(rowid) FROM emitted_signals GROUP BY data_cutoff, model, symbol
                   )"""
            )
        cutoff_exists = connection.execute(
            "SELECT 1 FROM emitted_signals WHERE data_cutoff = ? LIMIT 1", (prices.index.max().isoformat(),)
        ).fetchone()
        if cutoff_exists is None:
            emitted.to_sql("emitted_signals", connection, if_exists="append", index=False)
        prospective_observations = connection.execute(
            "SELECT COUNT(DISTINCT data_cutoff) FROM emitted_signals"
        ).fetchone()[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prospective_shadow_initialized"] = True
    manifest["prospective_observations"] = prospective_observations
    manifest["latest_shadow_data_cutoff"] = prices.index.max().isoformat()
    manifest["latest_shadow_emitted_at"] = emitted_at
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    action = "Already stored" if cutoff_exists is not None else "Stored"
    print(f"{action} {len(emitted):,} ticker-model signal rows for {len(model_names)} seed models at {prices.index.max().date()}")


if __name__ == "__main__":
    main()
