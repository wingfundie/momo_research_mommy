from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.research.signals import (
    basket_rank_weights,
    beta_constrained_weights,
    btc_hedged_weights,
    btc_residualized_forecasts,
    continuous_rank_weights,
    cross_sectional_percentiles,
    long_only_rank_weights,
    rolling_betas,
)
from scripts.execute_carver_breakout_v2 import (
    HORIZONS as BREAKOUT_HORIZONS,
    breakout_components,
    eligibility_path,
    parameter_path,
)
from scripts.execute_complete_crypto_study import (
    GROSS_CAPS,
    TARGETS,
    TICKER_CAPS,
    VOL_WINDOWS,
    buffered_basket,
    calibration_path,
    capped_simplex,
    components,
    fast_risk_unit,
    load_full_price_panel,
    row_correlation,
    signed_normalized,
)
from scripts.execute_full_crypto_study import funding_coefficients
from scripts.execute_production_like_walkforward import (
    HEADLINE_SLIPPAGE_BPS,
    INPUTS,
    REBALANCES,
    Simulation,
    delayed_positions,
    equal_risk_held,
    fold_definitions,
    funding_tradability_mask,
    rich_metrics,
    rolling_metrics,
    search_score,
    selection_key,
    selection_window,
    simulate_arrays,
    simulate_search_batch,
    simulation_from_held,
)


OUT = ROOT / "data_store/crypto_momentum_research/cross_sectional_trend_v1"
METHODOLOGY = "cross-sectional-trend-v1"
IC_SCHEMES = {"ic1": (1,), "ic5": (5,), "ic20": (20,), "ic5_20": (5, 20)}
FAMILIES = ("xs_momentum", "xs_breakout")
ROLLING_HISTORY_DAYS = 730


@dataclass(frozen=True)
class CrossSectionalModel:
    family: str
    name: str
    forecast: pd.DataFrame
    config: dict


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def config_id(config: dict) -> str:
    payload = json.dumps(json_safe(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def frame_digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def ic_weighted_forecast(
    scaled_components: np.ndarray,
    prices: pd.DataFrame,
    outcome_prices: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    history_days: int | None,
) -> pd.DataFrame:
    """Causal, quarterly cross-sectional IC weights with optional trailing history."""
    index, columns = prices.index, prices.columns
    rule_count = scaled_components.shape[2]
    component_ranks = np.stack(
        [
            pd.DataFrame(scaled_components[:, :, rule], index=index, columns=columns)
            .rank(axis=1, pct=True)
            .to_numpy(float)
            for rule in range(rule_count)
        ],
        axis=2,
    )
    future_ranks = {
        horizon: outcome_prices.shift(-(horizon + 1)).div(outcome_prices.shift(-1)).sub(1)
        .rank(axis=1, pct=True).to_numpy(float)
        for horizon in horizons
    }
    steps = np.full((len(index), rule_count), np.nan)
    equal = np.ones(rule_count) / rule_count
    quarterly = []
    for stamp in pd.date_range(index.min(), index.max(), freq="QS"):
        location = int(index.searchsorted(stamp, side="left"))
        if location >= 365 and location < len(index):
            quarterly.append(location)
    for location in quarterly:
        history_start = 0 if history_days is None else int(
            index.searchsorted(index[location] - pd.Timedelta(days=history_days), side="left")
        )
        scores = []
        for rule in range(rule_count):
            horizon_ics = []
            for horizon in horizons:
                usable = max(location - horizon - 1, history_start)
                correlations = row_correlation(
                    component_ranks[history_start:usable, :, rule],
                    future_ranks[horizon][history_start:usable],
                )
                finite = correlations[np.isfinite(correlations)]
                horizon_ics.append(float(finite.mean()) if len(finite) else np.nan)
            finite_ics = [value for value in horizon_ics if np.isfinite(value)]
            scores.append(max(float(np.mean(finite_ics)), 0.0) if finite_ics else 0.0)
        raw = np.asarray(scores, float)
        fitted = raw / raw.sum() if raw.sum() > 0 else equal
        weights = capped_simplex(0.20 * fitted + 0.80 * equal, 0.25)
        steps[min(location + 1, len(index) - 1)] = weights
    weights = pd.DataFrame(steps, index=index).ffill().fillna(pd.Series(equal))
    weights = weights.ewm(span=125, adjust=False).mean()
    weights = weights.div(weights.sum(axis=1), axis=0).to_numpy(float)
    valid = np.isfinite(scaled_components)
    numerator = np.nansum(scaled_components * weights[:, None, :], axis=2)
    denominator = np.sum(valid * weights[:, None, :], axis=2)
    final = np.divide(
        numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0
    )
    return pd.DataFrame(np.clip(final, -20, 20), index=index, columns=columns)


def cross_sectional_constructions(
    forecast: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    include_buffers: bool,
) -> dict[str, tuple[pd.DataFrame, float]]:
    ranks = cross_sectional_percentiles(forecast)
    # pandas percentile ranks average above 0.5 in a finite cross-section.  Demean
    # each row explicitly so the construction is genuinely dollar neutral.
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    continuous = centered.div(centered.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    betas = rolling_betas(returns, lookback=90)
    result: dict[str, tuple[pd.DataFrame, float]] = {
        "signed_unhedged": (signed_normalized(forecast), 0.0),
        "dollar_neutral": (continuous, 0.0),
        "basket_10": (basket_rank_weights(ranks, 0.10), 0.0),
        "basket_20": (basket_rank_weights(ranks, 0.20), 0.0),
        "basket_30": (basket_rank_weights(ranks, 0.30), 0.0),
        "long_only_20": (long_only_rank_weights(ranks, forecast, 0.20), 0.0),
        "btc_hedged": (btc_hedged_weights(continuous, betas), 0.0),
        "beta_constrained": (beta_constrained_weights(continuous, betas), 0.0),
    }
    residual = btc_residualized_forecasts(forecast, returns, lookback=90)
    residual_ranks = cross_sectional_percentiles(residual)
    residual_centered = residual_ranks.sub(residual_ranks.mean(axis=1), axis=0)
    result["btc_residualized"] = (
        residual_centered.div(
            residual_centered.abs().sum(axis=1).replace(0.0, np.nan), axis=0
        ).fillna(0.0),
        0.0,
    )
    if include_buffers:
        result["basket20_buffer5"] = (buffered_basket(ranks, 0.20, 0.05), 0.05)
        result["basket20_buffer10"] = (buffered_basket(ranks, 0.20, 0.10), 0.10)
    return result


def iter_models(
    family: str,
    fit_prices: pd.DataFrame,
    portfolio_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    maker: pd.Series,
    taker: pd.Series,
    *,
    history_days: int | None,
    allowed_names: set[str] | None = None,
) -> Iterator[CrossSectionalModel]:
    locations = [fit_prices.columns.get_loc(column) for column in portfolio_prices]
    fit_returns = fit_prices.pct_change(fill_method=None).to_numpy(float)
    portfolio_returns = portfolio_prices.pct_change(fill_method=None)
    breakout_raw = breakout_components(fit_prices) if family == "xs_breakout" else None
    for risk_window in VOL_WINDOWS:
        suffix = f"_vol{risk_window}"
        if allowed_names is not None and not any(name.endswith(suffix) for name in allowed_names):
            continue
        if family == "xs_momentum":
            raw = components(fit_prices, risk_window)
            scalars, _, _ = calibration_path(
                raw, fit_returns, fit_prices.index, "quarterly", 0.80, 0.25, 125,
                history_days=history_days,
            )
            scaled = np.clip(raw[:, locations] * scalars[:, None, :], -20, 20)
            eligibility_note = "all_components"
        elif family == "xs_breakout":
            scalars, _, _ = parameter_path(
                breakout_raw, fit_returns, fit_prices.index, "quarterly",
                fixed_scalars=False, fit_weights=False, history_days=history_days,
            )
            scaled = np.clip(breakout_raw[:, locations] * scalars[:, None, :], -20, 20)
            eligible, _ = eligibility_path(
                portfolio_prices, maker, taker, taker_share=1.0,
                slippage_bps=HEADLINE_SLIPPAGE_BPS, risk_window=risk_window,
                schedule="quarterly",
            )
            scaled = np.where(eligible, scaled, np.nan)
            eligibility_note = "carver_speed_filter"
        else:
            raise ValueError(family)
        for ic_name, horizons in IC_SCHEMES.items():
            prefix = "xsm" if family == "xs_momentum" else "xsb"
            potential_names = [f"{prefix}_{ic_name}_{construction}_vol{risk_window}" for construction in (
                "signed_unhedged", "dollar_neutral", "basket_10", "basket_20", "basket_30",
                "long_only_20", "btc_hedged", "beta_constrained", "btc_residualized",
                "basket20_buffer5", "basket20_buffer10",
            )]
            if allowed_names is not None and not any(name in allowed_names for name in potential_names):
                continue
            forecast = ic_weighted_forecast(
                scaled, portfolio_prices, open_prices, horizons, history_days=history_days
            )
            for construction, (frame, rank_buffer) in cross_sectional_constructions(
                forecast, portfolio_returns, include_buffers=ic_name == "ic5_20"
            ).items():
                name = f"{prefix}_{ic_name}_{construction}_vol{risk_window}"
                if allowed_names is not None and name not in allowed_names:
                    continue
                yield CrossSectionalModel(
                    family=family,
                    name=name,
                    forecast=frame,
                    config={
                        "signal_family": family,
                        "component_horizons": (
                            [f"{fast}/{slow}" for fast, slow in ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128))]
                            if family == "xs_momentum" else list(BREAKOUT_HORIZONS)
                        ),
                        "ic_horizons": list(horizons),
                        "construction": construction,
                        "rank_buffer": rank_buffer,
                        "volatility_window": risk_window,
                        "component_eligibility": eligibility_note,
                    },
                )


def model_count() -> int:
    return len(VOL_WINDOWS) * (len(IC_SCHEMES) * 9 + 2)


def exact_config(
    model: CrossSectionalModel,
    *,
    phase: str,
    history_days: int | None,
    target: float,
    gross: float,
    cap: float,
    rebalance: str,
) -> dict:
    return {
        "methodology_version": METHODOLOGY,
        "phase": phase,
        "history_days": history_days,
        "family": model.family,
        "model": model.name,
        **model.config,
        "target_vol": target,
        "gross_cap": gross,
        "ticker_risk_cap": cap,
        "rebalance": rebalance,
        "activation": "next_open",
        "taker_share": 1.0,
        "slippage_bps": HEADLINE_SLIPPAGE_BPS,
        "funding_mode": "event_level_historical",
        "universe": "top100_current_fdv_fallback",
    }


def scan_phase(
    *,
    phase: str,
    families: tuple[str, ...],
    history_days: int | None,
    fit_prices: pd.DataFrame,
    portfolio_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    open_returns: pd.DataFrame,
    close_returns: pd.DataFrame,
    volatility_cache: dict[int, pd.DataFrame],
    same: pd.DataFrame,
    midnight: pd.DataFrame,
    maker: pd.Series,
    taker: pd.Series,
    tradable: pd.DataFrame,
    folds: list[dict],
) -> tuple[pd.DataFrame, Path]:
    path = OUT / f"{phase}_candidate_scores.parquet"
    catalog_path = OUT / f"{phase}_tested_configurations.jsonl.gz"
    if path.exists():
        path.unlink()
    writer: pq.ParquetWriter | None = None
    best_records: dict[tuple[str, str, str], dict] = {}
    index = portfolio_prices.index
    arrays = (
        open_returns.to_numpy(float), same.to_numpy(float), midnight.to_numpy(float),
        maker.to_numpy(float), taker.to_numpy(float),
    )
    with gzip.open(catalog_path, "wt", encoding="utf-8") as catalog:
        for family in families:
            for number, model in enumerate(
                iter_models(
                    family, fit_prices, portfolio_prices, open_prices, maker, taker,
                    history_days=history_days,
                ),
                1,
            ):
                print(f"{phase} {family} model {number}/{model_count()} {model.name}", flush=True)
                risk_window = int(model.config["volatility_window"])
                rows = []
                for cap in TICKER_CAPS:
                    unit = fast_risk_unit(
                        model.forecast, close_returns, volatility_cache[risk_window], cap, risk_window
                    ).where(tradable, 0.0).to_numpy(float)
                    for frequency in REBALANCES:
                        if model.config["rank_buffer"] and frequency != "weekly":
                            continue
                        combinations, net_batch, turnover_batch = simulate_search_batch(
                            unit, index, *arrays, rebalance=frequency
                        )
                        for combination_number, (target, gross) in enumerate(combinations):
                            config = exact_config(
                                model, phase=phase, history_days=history_days, target=target,
                                gross=gross, cap=float(cap), rebalance=frequency,
                            )
                            cid = config_id(config)
                            catalog.write(json.dumps({"config_id": cid, **json_safe(config)}, sort_keys=True) + "\n")
                            net = net_batch[combination_number]
                            turnover = turnover_batch[combination_number]
                            if phase == "standard_holdout":
                                validation = search_score(
                                    net, turnover, index, pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
                                )
                                holdout = search_score(
                                    net, turnover, index, pd.Timestamp("2026-01-01"), index.max()
                                )
                                record = {
                                    "phase": phase, "family": family, "model": model.name, "config_id": cid,
                                    "target_vol": target, "gross_cap": gross, "ticker_risk_cap": float(cap),
                                    "rebalance": frequency,
                                    "validation_net_sharpe": validation[0],
                                    "validation_annual_turnover": validation[1],
                                    "validation_observations": validation[2],
                                    "holdout_net_sharpe": holdout[0],
                                    "holdout_annual_turnover": holdout[1],
                                    "holdout_observations": holdout[2],
                                }
                                rows.append(record)
                            else:
                                for fold in folds:
                                    start, end = selection_window(index, fold["cutoff"], history_days)
                                    score, annual_turnover, observations = search_score(
                                        net, turnover, index, start, end
                                    )
                                    record = {
                                        "phase": phase, "fold": fold["fold"],
                                        "selection_cutoff": fold["cutoff"], "selection_start": start,
                                        "family": family, "model": model.name, "config_id": cid,
                                        "selection_net_sharpe": score,
                                        "selection_annual_turnover": annual_turnover,
                                        "selection_observations": observations,
                                        "target_vol": target, "gross_cap": gross,
                                        "ticker_risk_cap": float(cap), "rebalance": frequency,
                                    }
                                    rows.append(record)
                                    if score is not None:
                                        candidate = {**record, "config_json": json.dumps(json_safe(config), sort_keys=True)}
                                        key = (fold["fold"], family, model.name)
                                        current = best_records.get(key)
                                        if current is None or selection_key(candidate) < selection_key(current):
                                            best_records[key] = candidate
                table = pa.Table.from_pylist(rows)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
    if writer is not None:
        writer.close()
    if phase == "standard_holdout":
        candidates = pd.read_parquet(path)
        model_best = []
        for (_, model_name), group in candidates.groupby(["family", "model"]):
            valid = group.dropna(subset=["validation_net_sharpe"])
            if valid.empty:
                continue
            chosen = valid.sort_values(
                ["validation_net_sharpe", "validation_annual_turnover", "gross_cap", "target_vol", "config_id"],
                ascending=[False, True, True, True, True], kind="mergesort",
            ).iloc[0]
            model_best.append(chosen.to_dict())
        return pd.DataFrame(model_best), path
    return pd.DataFrame(best_records.values()), path


def select_standard(model_best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in model_best.groupby("family"):
        ranked = group.sort_values(
            ["validation_net_sharpe", "validation_annual_turnover", "gross_cap", "target_vol", "config_id"],
            ascending=[False, True, True, True, True], kind="mergesort",
        ).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["selected"] = ranked["rank"].eq(1)
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def select_rolling(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, family), group in best.groupby(["fold", "family"], sort=False):
        ranked = group.sort_values(
            ["selection_net_sharpe", "selection_annual_turnover", "gross_cap", "target_vol", "config_id"],
            ascending=[False, True, True, True, True], kind="mergesort",
        ).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["selected"] = ranked["rank"].le(3)
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def reconstruct(
    selected: pd.DataFrame,
    *,
    phase: str,
    history_days: int | None,
    fit_prices: pd.DataFrame,
    portfolio_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    open_returns: pd.DataFrame,
    close_returns: pd.DataFrame,
    volatility_cache: dict[int, pd.DataFrame],
    same: pd.DataFrame,
    midnight: pd.DataFrame,
    maker: pd.Series,
    taker: pd.Series,
    tradable: pd.DataFrame,
) -> dict[str, tuple[Simulation, pd.DataFrame]]:
    selected_rows = selected[selected.selected].copy()
    needed = set(selected_rows.config_id)
    allowed_names = set(selected_rows.model)
    output: dict[str, tuple[Simulation, pd.DataFrame]] = {}
    arrays = (
        open_returns.to_numpy(float), same.to_numpy(float), midnight.to_numpy(float),
        maker.to_numpy(float), taker.to_numpy(float),
    )
    for family in FAMILIES:
        family_names = set(selected_rows[selected_rows.family.eq(family)].model)
        if not family_names:
            continue
        for model in iter_models(
            family, fit_prices, portfolio_prices, open_prices, maker, taker,
            history_days=history_days, allowed_names=allowed_names,
        ):
            rows = selected_rows[selected_rows.model.eq(model.name)].drop_duplicates("config_id")
            for cap, block in rows.groupby("ticker_risk_cap"):
                risk_window = int(model.config["volatility_window"])
                unit = fast_risk_unit(
                    model.forecast, close_returns, volatility_cache[risk_window], float(cap), risk_window
                ).where(tradable, 0.0).to_numpy(float)
                for row in block.itertuples():
                    sim = simulate_arrays(
                        unit, portfolio_prices.index, *arrays,
                        target_vol=float(row.target_vol), gross_cap=float(row.gross_cap),
                        rebalance=row.rebalance, detail=True, columns=portfolio_prices.columns,
                    )
                    output[row.config_id] = (sim, model.forecast)
    missing = needed - set(output)
    if missing:
        raise RuntimeError(f"Selected configurations were not reconstructed: {sorted(missing)}")
    return output


def write_standard_outputs(
    selection: pd.DataFrame,
    simulations: dict[str, tuple[Simulation, pd.DataFrame]],
    index: pd.DatetimeIndex,
    btc_returns: pd.Series,
) -> None:
    masks = {
        "post_training": pd.Series(index >= index.min() + pd.Timedelta(days=365), index=index),
        "validation": pd.Series((index >= "2024-01-01") & (index < "2026-01-01"), index=index),
        "holdout_2026": pd.Series(index >= "2026-01-01", index=index),
    }
    rows = []
    state = []
    positions = []
    for selected in selection[selection.selected].itertuples():
        sim, forecast = simulations[selected.config_id]
        for period, mask in masks.items():
            rows.append({
                "family": selected.family, "model": selected.model,
                "config_id": selected.config_id, "period": period,
                **rich_metrics(sim, mask, btc_returns),
            })
        daily = pd.DataFrame({
            "timestamp": index, "family": selected.family, "model": selected.model,
            "net_return": sim.net, "gross_return": sim.gross_return,
            "fee": sim.fee, "slippage": sim.slippage, "funding": sim.funding,
            "turnover": sim.turnover, "gross_exposure": sim.held.abs().sum(axis=1),
            "net_exposure": sim.held.sum(axis=1),
        })
        state.append(daily)
        long = sim.held.stack().rename("held_weight").reset_index()
        long.columns = ["timestamp", "symbol", "held_weight"]
        long["family"] = selected.family
        long["model"] = selected.model
        long["forecast"] = forecast.stack(dropna=False).reset_index(drop=True)
        positions.append(long)
    pd.DataFrame(rows).to_parquet(OUT / "standard_selected_metrics.parquet", index=False)
    pd.concat(state, ignore_index=True).to_parquet(OUT / "standard_daily_state.parquet", index=False)
    pd.concat(positions, ignore_index=True).to_parquet(OUT / "standard_daily_positions.parquet", index=False)


def write_rolling_outputs(
    selection: pd.DataFrame,
    simulations: dict[str, tuple[Simulation, pd.DataFrame]],
    folds: list[dict],
    index: pd.DatetimeIndex,
    columns: pd.Index,
    open_returns: pd.DataFrame,
    same: pd.DataFrame,
    midnight: pd.DataFrame,
    maker: pd.Series,
    taker: pd.Series,
) -> None:
    stitched: dict[str, dict[str, pd.DataFrame]] = {}
    for fold in folds:
        mask = pd.Series((index >= fold["test_start"]) & (index <= fold["test_end"]), index=index)
        family_held: dict[str, pd.DataFrame] = {}
        for family in FAMILIES:
            roster = selection[
                selection.fold.eq(fold["fold"]) & selection.family.eq(family) & selection.selected
            ].sort_values("rank")
            selected_sims = []
            for row in roster.itertuples():
                sim, forecast = simulations[row.config_id]
                strategy = f"{family}_slot_{int(row.rank)}"
                bucket = stitched.setdefault(
                    strategy,
                    {
                        "held": pd.DataFrame(0.0, index=index, columns=columns),
                        "forecast": pd.DataFrame(np.nan, index=index, columns=columns),
                    },
                )
                bucket["held"].loc[mask] = sim.held.loc[mask]
                bucket["forecast"].loc[mask] = forecast.loc[mask]
                selected_sims.append((sim, float(row.target_vol)))
            family_held[family] = equal_risk_held(selected_sims)
            bucket = stitched.setdefault(
                f"{family}_ensemble", {"held": pd.DataFrame(0.0, index=index, columns=columns)}
            )
            bucket["held"].loc[mask] = family_held[family].loc[mask]
        combination = (family_held["xs_momentum"] + family_held["xs_breakout"]) / 2.0
        bucket = stitched.setdefault(
            "xs_combined_50_50", {"held": pd.DataFrame(0.0, index=index, columns=columns)}
        )
        bucket["held"].loc[mask] = combination.loc[mask]

    btc_returns = open_returns.get("BTCUSDT", pd.Series(0.0, index=index))
    outer_mask = pd.Series(index >= folds[0]["test_start"], index=index)
    metric_rows = []
    fold_rows = []
    state_rows = []
    position_rows = []
    rolling_rows = []
    for strategy, payload in stitched.items():
        sim = simulation_from_held(payload["held"], open_returns, same, midnight, maker, taker)
        metric_rows.append({"strategy": strategy, **rich_metrics(sim, outer_mask, btc_returns)})
        rolling_rows.append(rolling_metrics(sim, strategy, "rolling_730", btc_returns, [90, 365]))
        for fold in folds:
            mask = pd.Series((index >= fold["test_start"]) & (index <= fold["test_end"]), index=index)
            fold_rows.append({"strategy": strategy, "fold": fold["fold"], **rich_metrics(sim, mask, btc_returns)})
        gross = sim.held.abs().sum(axis=1)
        top5 = sim.held.abs().apply(lambda row: row.nlargest(5).sum(), axis=1).div(gross.replace(0, np.nan)).fillna(0)
        state_rows.append(pd.DataFrame({
            "timestamp": index, "strategy": strategy, "net_return": sim.net,
            "gross_return": sim.gross_return, "fee": sim.fee, "slippage": sim.slippage,
            "funding": sim.funding, "turnover": sim.turnover,
            "gross_exposure": gross, "net_exposure": sim.held.sum(axis=1),
            "long_count": (sim.held > 0).sum(axis=1), "short_count": (sim.held < 0).sum(axis=1),
            "top5_concentration": top5,
        }))
        long = sim.held.stack().rename("held_weight").reset_index()
        long.columns = ["timestamp", "symbol", "held_weight"]
        long["strategy"] = strategy
        long["net_return_contribution"] = sim.contribution.stack().reset_index(drop=True)
        forecast = payload.get("forecast")
        long["forecast"] = forecast.stack(dropna=False).reset_index(drop=True) if forecast is not None else np.nan
        position_rows.append(long)
    pd.DataFrame(metric_rows).to_parquet(OUT / "rolling_stitched_metrics.parquet", index=False)
    pd.DataFrame(fold_rows).to_parquet(OUT / "rolling_fold_metrics.parquet", index=False)
    pd.concat(state_rows, ignore_index=True).to_parquet(OUT / "rolling_daily_state.parquet", index=False)
    pd.concat(position_rows, ignore_index=True).to_parquet(OUT / "rolling_daily_positions.parquet", index=False)
    pd.concat(rolling_rows, ignore_index=True).to_parquet(OUT / "rolling_metrics.parquet", index=False)


def selected_configs(selection: pd.DataFrame, phase: str) -> list[dict]:
    rows = []
    for row in selection[selection.selected].itertuples():
        payload = json.loads(row.config_json) if isinstance(row.config_json, str) else {}
        rows.append({
            "selection": {
                "phase": phase,
                "fold": getattr(row, "fold", None),
                "rank": int(row.rank),
                "selection_score": getattr(row, "selection_net_sharpe", getattr(row, "validation_net_sharpe", None)),
            },
            "config_id": row.config_id,
            **json_safe(payload),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cross-sectional momentum and breakout study")
    parser.add_argument(
        "--rolling-only", action="store_true",
        help="Reuse a completed standard holdout phase and rebuild only rolling outputs",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_prices = load_full_price_panel()
    fit_prices = all_prices.loc[:, all_prices.notna().sum().ge(90)]
    universe = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    members = universe.loc[universe.member.astype(str).str.lower().isin(["true", "1"]), "symbol"]
    symbols = [symbol for symbol in members if symbol in fit_prices]
    portfolio_prices = fit_prices[symbols]
    open_prices = pd.read_parquet(INPUTS / "portfolio_open_prices.parquet").reindex(
        index=portfolio_prices.index, columns=symbols
    )
    events = pd.read_parquet(INPUTS / "funding_events.parquet")
    events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True)
    same, midnight = funding_coefficients(portfolio_prices, events)
    tradable = funding_tradability_mask(portfolio_prices.index, symbols, events)
    commissions = pd.read_csv(INPUTS / "commission_rates.csv").set_index("symbol")
    maker = commissions.maker.reindex(symbols).fillna(0.0002)
    taker = commissions.taker.reindex(symbols).fillna(0.0004)
    close_returns = portfolio_prices.pct_change(fill_method=None).fillna(0.0)
    open_returns = open_prices.shift(-1).div(open_prices).sub(1).fillna(0.0)
    volatility_cache = {
        window: close_returns.rolling(window, min_periods=window).std() * np.sqrt(365)
        for window in VOL_WINDOWS
    }
    folds = fold_definitions(portfolio_prices.index)

    standard_path = OUT / "standard_holdout_candidate_scores.parquet"
    if args.rolling_only:
        required = [
            standard_path,
            OUT / "standard_holdout_tested_configurations.jsonl.gz",
            OUT / "standard_selection_ledger.parquet",
            OUT / "standard_selected_metrics.parquet",
            OUT / "standard_daily_state.parquet",
            OUT / "standard_daily_positions.parquet",
        ]
        missing = [path for path in required if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Rolling-only recovery requires completed standard outputs: {missing}")
        standard_selection = pd.read_parquet(OUT / "standard_selection_ledger.parquet")
        print("Reusing completed standard holdout outputs; rebuilding rolling phase only", flush=True)
    else:
        standard_best, standard_path = scan_phase(
            phase="standard_holdout", families=FAMILIES, history_days=None,
            fit_prices=fit_prices, portfolio_prices=portfolio_prices, open_prices=open_prices,
            open_returns=open_returns, close_returns=close_returns, volatility_cache=volatility_cache,
            same=same, midnight=midnight, maker=maker, taker=taker, tradable=tradable, folds=folds,
        )
        standard_selection = select_standard(standard_best)
        standard_selection["config_json"] = None
        for selected_index, row in standard_selection[standard_selection.selected].iterrows():
            model = next(iter(iter_models(
                row.family, fit_prices, portfolio_prices, open_prices, maker, taker,
                history_days=None, allowed_names={row.model},
            )))
            config = exact_config(
                model, phase="standard_holdout", history_days=None, target=float(row.target_vol),
                gross=float(row.gross_cap), cap=float(row.ticker_risk_cap), rebalance=row.rebalance,
            )
            if config_id(config) != row.config_id:
                raise RuntimeError(f"Configuration reconstruction mismatch for {row.config_id}")
            standard_selection.at[selected_index, "config_json"] = json.dumps(json_safe(config), sort_keys=True)
        standard_selection.to_parquet(OUT / "standard_selection_ledger.parquet", index=False)
        standard_sims = reconstruct(
            standard_selection, phase="standard_holdout", history_days=None,
            fit_prices=fit_prices, portfolio_prices=portfolio_prices, open_prices=open_prices,
            open_returns=open_returns, close_returns=close_returns, volatility_cache=volatility_cache,
            same=same, midnight=midnight, maker=maker, taker=taker, tradable=tradable,
        )
        write_standard_outputs(
            standard_selection, standard_sims, portfolio_prices.index,
            open_returns.get("BTCUSDT", pd.Series(0.0, index=portfolio_prices.index)),
        )

    rolling_best, rolling_path = scan_phase(
        phase="rolling_730", families=FAMILIES, history_days=ROLLING_HISTORY_DAYS,
        fit_prices=fit_prices, portfolio_prices=portfolio_prices, open_prices=open_prices,
        open_returns=open_returns, close_returns=close_returns, volatility_cache=volatility_cache,
        same=same, midnight=midnight, maker=maker, taker=taker, tradable=tradable, folds=folds,
    )
    rolling_selection = select_rolling(rolling_best)
    rolling_selection.to_parquet(OUT / "rolling_selection_ledger.parquet", index=False)
    rolling_sims = reconstruct(
        rolling_selection, phase="rolling_730", history_days=ROLLING_HISTORY_DAYS,
        fit_prices=fit_prices, portfolio_prices=portfolio_prices, open_prices=open_prices,
        open_returns=open_returns, close_returns=close_returns, volatility_cache=volatility_cache,
        same=same, midnight=midnight, maker=maker, taker=taker, tradable=tradable,
    )
    write_rolling_outputs(
        rolling_selection, rolling_sims, folds, portfolio_prices.index, portfolio_prices.columns,
        open_returns, same, midnight, maker, taker,
    )

    exact_selected = {
        "schema_version": 1,
        "methodology_version": METHODOLOGY,
        "standard_holdout": selected_configs(standard_selection, "standard_holdout"),
        "rolling_730": selected_configs(rolling_selection, "rolling_730"),
    }
    (OUT / "selected_configurations.json").write_text(
        json.dumps(json_safe(exact_selected), indent=2), encoding="utf-8"
    )
    standard_count = len(pd.read_parquet(standard_path, columns=["config_id"]))
    rolling_count = len(pd.read_parquet(rolling_path, columns=["config_id"]))
    manifest = {
        "schema_version": 1,
        "methodology_version": METHODOLOGY,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_cutoff": portfolio_prices.index.max().isoformat(),
        "families": list(FAMILIES),
        "models_per_family": model_count(),
        "standard_candidate_rows": standard_count,
        "rolling_candidate_fold_rows": rolling_count,
        "configuration_catalogs": {
            "standard_holdout": str(OUT / "standard_holdout_tested_configurations.jsonl.gz"),
            "rolling_730": str(OUT / "rolling_730_tested_configurations.jsonl.gz"),
        },
        "standard_selection_uses_holdout": False,
        "standard_validation": ["2024-01-01", "2025-12-31"],
        "standard_holdout": ["2026-01-01", portfolio_prices.index.max().isoformat()],
        "rolling_history_days": ROLLING_HISTORY_DAYS,
        "rolling_outer_folds": json_safe(folds),
        "risk_grid": {
            "volatility_windows": list(VOL_WINDOWS), "targets": list(TARGETS),
            "gross_caps": list(GROSS_CAPS), "ticker_caps": list(TICKER_CAPS),
            "rebalances": list(REBALANCES),
        },
        "execution": {"activation": "next_open", "taker_share": 1.0, "slippage_bps": 5.0},
        "funding": "event_level_historical",
        "tradability": "day_after_first_fully_priced_funding_event",
        "universe": "top100_current_fdv_fallback",
        "fit_assets": fit_prices.shape[1], "portfolio_assets": portfolio_prices.shape[1],
        "input_hashes": {
            "fit_prices": frame_digest(fit_prices), "portfolio_prices": frame_digest(portfolio_prices),
            "open_prices": frame_digest(open_prices), "funding": frame_digest(events),
        },
        "production_signals_changed": False,
    }
    (OUT / "study_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
