from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
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

from scripts.execute_carver_breakout_v2 import (
    EXPECTED_TURNOVER,
    FDM_CAP,
    FIXED_FDM,
    FIXED_SCALARS,
    HORIZONS,
    breakout_components,
    combine_components,
    eligibility_path,
    parameter_path,
)
from scripts.execute_complete_crypto_study import (
    GROSS_CAPS,
    REFITS,
    REBALANCES,
    RULES,
    SPECS,
    TARGETS,
    TICKER_CAPS,
    VOL_WINDOWS,
    calibration_path,
    combine,
    components,
    digest_frame,
    fast_risk_unit,
    load_full_price_panel,
)
from scripts.execute_full_crypto_study import funding_coefficients


LOCKED = ROOT / "configs/production_like_walkforward_v1.json"
INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"
OUT = ROOT / "data_store/crypto_momentum_research/production_like_walkforward_v1"
METHODOLOGY = "crypto-trend-production-like-walkforward-v1"
HEADLINE_TAKER_SHARE = 1.0
HEADLINE_SLIPPAGE_BPS = 5.0


@dataclass(frozen=True)
class ModelForecast:
    family: str
    name: str
    forecast: pd.DataFrame
    config: dict


@dataclass
class Simulation:
    net: pd.Series
    turnover: pd.Series
    funding: pd.Series
    fee: pd.Series
    slippage: pd.Series
    gross_return: pd.Series
    target: pd.DataFrame
    held: pd.DataFrame
    contribution: pd.DataFrame


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def config_id(config: dict) -> str:
    return hashlib.sha256(json.dumps(json_safe(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def model_config_from_name(name: str) -> dict:
    volatility_suffix = re.search(r"_vol(\d+)$", name)
    if volatility_suffix and int(volatility_suffix.group(1)) not in VOL_WINDOWS:
        raise ValueError(f"Unsupported volatility window in model name: {name}")
    equal = re.fullmatch(r"ts_equal_vol(\d+)", name)
    if equal:
        return {"rule_weights": "equal", "refit": "quarterly_scalars", "volatility_window": int(equal.group(1))}
    momentum = re.fullmatch(r"ts_(shrink_75|shrink_80_primary|shrink_90)_(quarterly|semiannual|annual|frozen)_vol(\d+)", name)
    if momentum:
        return {"rule_weights": momentum.group(1), "refit": momentum.group(2), "volatility_window": int(momentum.group(3))}
    carver = re.fullmatch(r"breakout_carver5_equal_(causal|fixed)_fdm_vol(\d+)", name)
    if carver:
        return {
            "method": "carver_fixed_scalars_equal",
            "fdm_mode": "causal_expanding" if carver.group(1) == "causal" else "fixed_1.20",
            "refit": "quarterly",
            "volatility_window": int(carver.group(2)),
        }
    breakout_equal = re.fullmatch(r"breakout_crypto_equal_(quarterly|semiannual|annual|frozen)_vol(\d+)", name)
    if breakout_equal:
        return {
            "method": "crypto_expanding_scalars_equal", "fdm_mode": "causal_expanding",
            "refit": breakout_equal.group(1), "volatility_window": int(breakout_equal.group(2)),
        }
    breakout_shrunk = re.fullmatch(
        r"breakout_crypto_(shrink_75|shrink_80_primary|shrink_90)_(quarterly|semiannual|annual|frozen)_vol(\d+)", name
    )
    if breakout_shrunk:
        return {
            "method": "crypto_expanding_scalars_shrunk_weights", "weight_spec": breakout_shrunk.group(1),
            "fdm_mode": "causal_expanding", "refit": breakout_shrunk.group(2),
            "volatility_window": int(breakout_shrunk.group(3)),
        }
    raise ValueError(f"Unsupported model name: {name}")


def exact_configuration(record) -> dict:
    history_days = None if record.variant == "nested_expanding" else 730
    return {
        "methodology_version": METHODOLOGY, "history_variant": record.variant,
        "history_days": history_days, "family": record.family, "model": record.model,
        **model_config_from_name(record.model), "target_vol": float(record.target_vol),
        "gross_cap": float(record.gross_cap), "ticker_risk_cap": float(record.ticker_risk_cap),
        "rebalance": record.rebalance, "activation": "next_open", "taker_share": HEADLINE_TAKER_SHARE,
        "slippage_bps": HEADLINE_SLIPPAGE_BPS,
    }


def write_configuration_catalog(candidate_path: Path, selection: pd.DataFrame | None = None) -> int:
    columns = ["variant", "family", "model", "config_id", "target_vol", "gross_cap", "ticker_risk_cap", "rebalance"]
    unique = pd.read_parquet(candidate_path, columns=columns).drop_duplicates("config_id").sort_values("config_id")
    path = OUT / "all_tested_configurations.json"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema_version": 1, "methodology_version": METHODOLOGY,
                                 "configuration_count": len(unique)}, separators=(",", ":"))[:-1])
        stream.write(',"configurations":[')
        for number, record in enumerate(unique.itertuples(index=False)):
            config = exact_configuration(record)
            if config_id(config) != record.config_id:
                raise ValueError(f"Configuration hash mismatch for {record.config_id}")
            if number:
                stream.write(",")
            stream.write(json.dumps({"config_id": record.config_id, **json_safe(config)}, separators=(",", ":")))
        stream.write("]}")
    if selection is not None:
        selected_records = []
        for record in selection[selection.selected].sort_values(["variant", "fold", "family", "rank"]).itertuples(index=False):
            selected_records.append({
                "fold": record.fold, "rank": int(record.rank),
                "selection_cutoff": json_safe(record.selection_cutoff),
                "selection_net_sharpe": json_safe(record.selection_net_sharpe),
                "selection_annual_turnover": json_safe(record.selection_annual_turnover),
                "config_id": record.config_id, **json_safe(exact_configuration(record)),
            })
        (OUT / "selected_deployable_configurations.json").write_text(
            json.dumps({"schema_version": 1, "methodology_version": METHODOLOGY,
                        "selection_uses_holdout_metrics": False, "configurations": selected_records}, indent=2),
            encoding="utf-8",
        )
    return len(unique)


def fold_definitions(index: pd.DatetimeIndex) -> list[dict]:
    folds = []
    for year in range(2022, 2027):
        start = pd.Timestamp(f"{year}-01-01")
        end = min(pd.Timestamp(f"{year}-12-31"), index.max())
        if start <= index.max():
            folds.append({"fold": str(year) if year < 2026 else "2026_ytd", "cutoff": start - pd.Timedelta(days=1),
                          "test_start": start, "test_end": end})
    return folds


def selection_window(index: pd.DatetimeIndex, cutoff: pd.Timestamp, history_days: int | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    available_start = index.min() + pd.Timedelta(days=365)
    start = available_start if history_days is None else max(available_start, cutoff - pd.Timedelta(days=history_days - 1))
    return start, cutoff


def funding_tradability_mask(index: pd.DatetimeIndex, symbols: list[str], events: pd.DataFrame) -> pd.DataFrame:
    finite = events[events.mark_price.notna() & events.symbol.isin(symbols)].copy()
    first_covered = finite.groupby("symbol")["funding_time"].min()
    missing = sorted(set(symbols) - set(first_covered.index))
    if missing:
        raise ValueError(f"No fully priced funding event is available for: {missing}")
    mask = pd.DataFrame(False, index=index, columns=symbols)
    normalized_index = pd.DatetimeIndex(index).normalize()
    for symbol in symbols:
        first_tradable_day = pd.Timestamp(first_covered[symbol]).tz_convert(None).normalize() + pd.Timedelta(days=1)
        mask[symbol] = normalized_index >= first_tradable_day
    return mask


def search_score(net: np.ndarray, turnover: np.ndarray, index: pd.DatetimeIndex,
                 start: pd.Timestamp, end: pd.Timestamp) -> tuple[float | None, float | None, int]:
    mask = (index >= start) & (index <= end)
    values = net[mask]
    valid = np.isfinite(values)
    if valid.sum() < 180:
        return None, None, int(valid.sum())
    values = values[valid]
    vol = float(values.std(ddof=1) * np.sqrt(365))
    annual_return = float(values.mean() * 365)
    sharpe = annual_return / vol if vol > 0 else None
    annual_turnover = float(np.nanmean(turnover[mask]) * 365)
    return sharpe, annual_turnover, int(valid.sum())


def delayed_positions(desired: np.ndarray, index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    if frequency == "daily":
        rebalance = np.ones(len(index), dtype=bool)
    elif frequency == "weekly":
        rebalance = index.weekday == 0
    elif frequency == "monthly":
        rebalance = np.asarray(~index.to_period("M").duplicated())
    else:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    source = np.maximum.accumulate(np.where(rebalance, np.arange(len(index)), -1))
    delayed_source = np.full(len(index), -1, dtype=int)
    delayed_source[1:] = source[:-1]
    held = np.zeros_like(desired)
    valid = delayed_source >= 0
    held[valid] = desired[delayed_source[valid]]
    return held


def simulate_arrays(
    unit: np.ndarray,
    index: pd.DatetimeIndex,
    returns: np.ndarray,
    same: np.ndarray,
    midnight: np.ndarray,
    maker: np.ndarray,
    taker: np.ndarray,
    *,
    target_vol: float,
    gross_cap: float,
    rebalance: str,
    taker_share: float = HEADLINE_TAKER_SHARE,
    slippage_bps: float = HEADLINE_SLIPPAGE_BPS,
    detail: bool = False,
    columns: pd.Index | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Simulation:
    gross = np.abs(unit).sum(axis=1)
    multiplier = np.minimum(target_vol, np.divide(gross_cap, gross, out=np.full_like(gross, target_vol), where=gross > 0))
    desired = unit * multiplier[:, None]
    held = delayed_positions(desired, index, rebalance)
    trades = np.abs(np.diff(held, axis=0, prepend=np.zeros((1, held.shape[1]))))
    turnover = trades.sum(axis=1)
    blended = taker * taker_share + maker * (1 - taker_share)
    fee_matrix = trades * blended[None, :]
    slip_matrix = trades * slippage_bps / 10_000.0
    funding_matrix = held * same + np.vstack([np.zeros((1, held.shape[1])), held[:-1]]) * midnight
    gross_matrix = held * returns
    contribution = gross_matrix - fee_matrix - slip_matrix + funding_matrix
    net = contribution.sum(axis=1)
    funding = funding_matrix.sum(axis=1)
    if not detail:
        return net, turnover, funding
    if columns is None:
        raise ValueError("columns are required for detailed simulation")
    frame = lambda values: pd.DataFrame(values, index=index, columns=columns)
    series = lambda values: pd.Series(values, index=index)
    return Simulation(
        net=series(net), turnover=series(turnover), funding=series(funding),
        fee=series(fee_matrix.sum(axis=1)), slippage=series(slip_matrix.sum(axis=1)),
        gross_return=series(gross_matrix.sum(axis=1)), target=frame(desired), held=frame(held),
        contribution=frame(contribution),
    )


def simulate_search_batch(
    unit: np.ndarray,
    index: pd.DatetimeIndex,
    returns: np.ndarray,
    same: np.ndarray,
    midnight: np.ndarray,
    maker: np.ndarray,
    taker: np.ndarray,
    *,
    rebalance: str,
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    """Evaluate every target/gross pair together for one unit portfolio and frequency."""
    combinations = [(float(target), float(gross)) for target in TARGETS for gross in GROSS_CAPS]
    targets = np.asarray([item[0] for item in combinations])[:, None]
    gross_caps = np.asarray([item[1] for item in combinations])[:, None]
    unit_gross = np.abs(unit).sum(axis=1)[None, :]
    multiplier = np.minimum(
        targets,
        np.divide(gross_caps, unit_gross, out=np.broadcast_to(targets, (len(combinations), len(index))).copy(), where=unit_gross > 0),
    )
    desired = unit[None, :, :] * multiplier[:, :, None]
    if rebalance == "daily":
        rebalance_mask = np.ones(len(index), dtype=bool)
    elif rebalance == "weekly":
        rebalance_mask = index.weekday == 0
    elif rebalance == "monthly":
        rebalance_mask = np.asarray(~index.to_period("M").duplicated())
    else:
        raise ValueError(rebalance)
    source = np.maximum.accumulate(np.where(rebalance_mask, np.arange(len(index)), -1))
    delayed_source = np.full(len(index), -1, dtype=int)
    delayed_source[1:] = source[:-1]
    held = np.zeros_like(desired)
    valid = delayed_source >= 0
    held[:, valid, :] = desired[:, delayed_source[valid], :]
    trades = np.abs(np.diff(held, axis=1, prepend=np.zeros((len(combinations), 1, held.shape[2]))))
    turnover = trades.sum(axis=2)
    fee = (trades * taker[None, None, :]).sum(axis=2)
    funding = (held * same[None, :, :] + np.concatenate([np.zeros((len(combinations), 1, held.shape[2])), held[:, :-1]], axis=1) * midnight[None, :, :]).sum(axis=2)
    net = (held * returns[None, :, :]).sum(axis=2) - fee - turnover * HEADLINE_SLIPPAGE_BPS / 10_000.0 + funding
    return combinations, net, turnover


def iter_momentum_models(
    fit_prices: pd.DataFrame,
    portfolio_prices: pd.DataFrame,
    history_days: int | None,
    allowed_models: set[str] | None = None,
) -> Iterator[ModelForecast]:
    fit_returns = fit_prices.pct_change(fill_method=None).to_numpy(float)
    locations = [fit_prices.columns.get_loc(column) for column in portfolio_prices]
    for vol_window in VOL_WINDOWS:
        if allowed_models is not None and not any(name.endswith(f"_vol{vol_window}") for name in allowed_models):
            continue
        raw = components(fit_prices, vol_window)
        for schedule in REFITS:
            for spec_name, (shrink, cap, smoothing) in SPECS.items():
                name = f"ts_{spec_name}_{schedule}_vol{vol_window}"
                if allowed_models is not None and name not in allowed_models:
                    continue
                scalars, weights, dfm = calibration_path(
                    raw, fit_returns, fit_prices.index, schedule, shrink, cap, smoothing, history_days=history_days
                )
                final = combine(raw[:, locations], scalars, weights, dfm)
                yield ModelForecast("time_series", name, pd.DataFrame(final / 20, index=fit_prices.index, columns=portfolio_prices.columns),
                                    {"rule_weights": spec_name, "refit": schedule, "volatility_window": vol_window})
        name = f"ts_equal_vol{vol_window}"
        if allowed_models is None or name in allowed_models:
            scalars, _, dfm = calibration_path(
                raw, fit_returns, fit_prices.index, "quarterly", 1.0, 0.20, 1, history_days=history_days
            )
            equal = np.ones((len(fit_prices), len(RULES))) / len(RULES)
            final = combine(raw[:, locations], scalars, equal, dfm)
            yield ModelForecast("time_series", name, pd.DataFrame(final / 20, index=fit_prices.index, columns=portfolio_prices.columns),
                                {"rule_weights": "equal", "refit": "quarterly_scalars", "volatility_window": vol_window})


def iter_breakout_models(
    fit_prices: pd.DataFrame,
    portfolio_prices: pd.DataFrame,
    maker: pd.Series,
    taker: pd.Series,
    history_days: int | None,
    allowed_models: set[str] | None = None,
) -> Iterator[ModelForecast]:
    locations = [fit_prices.columns.get_loc(column) for column in portfolio_prices]
    raw_fit = breakout_components(fit_prices)
    raw = raw_fit[:, locations]
    returns = fit_prices.pct_change(fill_method=None).to_numpy(float)
    parameter_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    eligibility_cache: dict[tuple, np.ndarray] = {}

    def parameters(schedule: str, fixed: bool, fitted: bool, shrink: float, cap: float, smoothing: int):
        key = (schedule, fixed, fitted, shrink, cap, smoothing)
        if key not in parameter_cache:
            parameter_cache[key] = parameter_path(
                raw_fit, returns, fit_prices.index, schedule, fixed_scalars=fixed, fit_weights=fitted,
                shrink=shrink, weight_cap=cap, smoothing=smoothing, history_days=history_days,
            )
        return parameter_cache[key]

    def active(risk_window: int, schedule: str) -> np.ndarray:
        key = (risk_window, schedule)
        if key not in eligibility_cache:
            eligibility_cache[key] = eligibility_path(
                portfolio_prices, maker, taker, taker_share=HEADLINE_TAKER_SHARE,
                slippage_bps=HEADLINE_SLIPPAGE_BPS, risk_window=risk_window, schedule=schedule,
            )[0]
        return eligibility_cache[key]

    def build(name: str, config: dict, param_key: tuple, fixed_fdm: float | None = None) -> ModelForecast:
        schedule = param_key[0]
        scalars, weights, corr = parameters(*param_key)
        forecast, _, _ = combine_components(
            raw, scalars, weights, corr, active(int(config["volatility_window"]), schedule),
            fit_prices.index, portfolio_prices.columns, fixed_fdm=fixed_fdm,
        )
        return ModelForecast("breakout", name, forecast, config)

    for risk_window in VOL_WINDOWS:
        if allowed_models is not None and not any(name.endswith(f"_vol{risk_window}") for name in allowed_models):
            continue
        name = f"breakout_carver5_equal_causal_fdm_vol{risk_window}"
        if allowed_models is None or name in allowed_models:
            yield build(
                name,
                {"method": "carver_fixed_scalars_equal", "fdm_mode": "causal_expanding", "refit": "quarterly", "volatility_window": risk_window},
                ("quarterly", True, False, 1.0, 0.20, 1),
            )
        name = f"breakout_carver5_equal_fixed_fdm_vol{risk_window}"
        if allowed_models is None or name in allowed_models:
            yield build(
                name,
                {"method": "carver_fixed_scalars_equal", "fdm_mode": "fixed_1.20", "refit": "quarterly", "volatility_window": risk_window},
                ("quarterly", True, False, 1.0, 0.20, 1), fixed_fdm=FIXED_FDM,
            )
        for schedule in REFITS:
            name = f"breakout_crypto_equal_{schedule}_vol{risk_window}"
            if allowed_models is None or name in allowed_models:
                yield build(
                    name,
                    {"method": "crypto_expanding_scalars_equal", "fdm_mode": "causal_expanding", "refit": schedule,
                     "volatility_window": risk_window},
                    (schedule, False, False, 1.0, 0.20, 1),
                )
            for spec_name, (shrink, cap, smoothing) in SPECS.items():
                name = f"breakout_crypto_{spec_name}_{schedule}_vol{risk_window}"
                if allowed_models is None or name in allowed_models:
                    yield build(
                        name,
                        {"method": "crypto_expanding_scalars_shrunk_weights", "weight_spec": spec_name,
                         "fdm_mode": "causal_expanding", "refit": schedule, "volatility_window": risk_window},
                        (schedule, False, True, shrink, cap, smoothing),
                    )


def candidate_models(family: str, fit_prices: pd.DataFrame, portfolio_prices: pd.DataFrame,
                     maker: pd.Series, taker: pd.Series, history_days: int | None) -> Iterator[ModelForecast]:
    if family == "time_series":
        yield from iter_momentum_models(fit_prices, portfolio_prices, history_days)
    elif family == "breakout":
        yield from iter_breakout_models(fit_prices, portfolio_prices, maker, taker, history_days)
    else:
        raise ValueError(family)


def selection_key(record: dict) -> tuple:
    score = record.get("selection_net_sharpe")
    return (-score if score is not None else math.inf, record.get("selection_annual_turnover", math.inf),
            record["gross_cap"], record["target_vol"], record["config_id"])


def scan_candidates(
    *,
    variant: str,
    history_days: int | None,
    family: str,
    fit_prices: pd.DataFrame,
    portfolio_prices: pd.DataFrame,
    close_returns: pd.DataFrame,
    open_returns: pd.DataFrame,
    volatility_cache: dict[int, pd.DataFrame],
    same: pd.DataFrame,
    midnight: pd.DataFrame,
    maker: pd.Series,
    taker: pd.Series,
    tradable: pd.DataFrame,
    folds: list[dict],
    writer: pq.ParquetWriter | None,
) -> tuple[dict, pq.ParquetWriter]:
    best_by_model: dict[tuple, dict] = {}
    index = portfolio_prices.index
    returns_array = open_returns.to_numpy(float)
    same_array, midnight_array = same.to_numpy(float), midnight.to_numpy(float)
    maker_array, taker_array = maker.to_numpy(float), taker.to_numpy(float)
    total = len(VOL_WINDOWS) * (len(REFITS) * len(SPECS) + 1) if family == "time_series" else len(VOL_WINDOWS) * (2 + len(REFITS) * (1 + len(SPECS)))
    for model_number, model in enumerate(candidate_models(family, fit_prices, portfolio_prices, maker, taker, history_days), 1):
        print(f"{variant} {family} model {model_number}/{total} {model.name}", flush=True)
        risk_window = int(model.config["volatility_window"])
        unit_by_cap = {
            cap: fast_risk_unit(model.forecast, close_returns, volatility_cache[risk_window], cap, risk_window).where(tradable, 0.0).to_numpy(float)
            for cap in TICKER_CAPS
        }
        score_rows = []
        for cap, unit in unit_by_cap.items():
            for frequency in REBALANCES:
                combinations, net_batch, turnover_batch = simulate_search_batch(
                    unit, index, returns_array, same_array, midnight_array, maker_array, taker_array,
                    rebalance=frequency,
                )
                for combination_number, (target, gross) in enumerate(combinations):
                    net = net_batch[combination_number]
                    turnover = turnover_batch[combination_number]
                    cfg = {
                        "methodology_version": METHODOLOGY, "history_variant": variant,
                        "history_days": history_days, "family": family, "model": model.name, **model.config,
                        "target_vol": target, "gross_cap": gross, "ticker_risk_cap": cap,
                        "rebalance": frequency, "activation": "next_open", "taker_share": HEADLINE_TAKER_SHARE,
                        "slippage_bps": HEADLINE_SLIPPAGE_BPS,
                    }
                    cid = config_id(cfg)
                    for fold in folds:
                        start, end = selection_window(index, fold["cutoff"], history_days)
                        score, annual_turnover, observations = search_score(net, turnover, index, start, end)
                        record = {
                            "variant": variant, "fold": fold["fold"], "selection_cutoff": fold["cutoff"],
                            "selection_start": start, "family": family, "model": model.name, "config_id": cid,
                            "selection_net_sharpe": score, "selection_annual_turnover": annual_turnover,
                            "selection_observations": observations, "target_vol": target, "gross_cap": gross,
                            "ticker_risk_cap": cap, "rebalance": frequency,
                        }
                        score_rows.append(record)
                        key = (fold["fold"], family, model.name)
                        if score is not None and (key not in best_by_model or selection_key(record) < selection_key(best_by_model[key])):
                            best_by_model[key] = record
        table = pa.Table.from_pylist(score_rows)
        if writer is None:
            writer = pq.ParquetWriter(OUT / "candidate_score_ledger.parquet", table.schema, compression="zstd")
        writer.write_table(table)
    if writer is None:
        raise RuntimeError("No candidate scores were generated")
    return best_by_model, writer


def select_rosters(best_records: list[dict]) -> pd.DataFrame:
    rows = []
    for (variant, fold, family), group in pd.DataFrame(best_records).groupby(["variant", "fold", "family"], sort=False):
        ranked = group.sort_values(
            ["selection_net_sharpe", "selection_annual_turnover", "gross_cap", "target_vol", "config_id"],
            ascending=[False, True, True, True, True], kind="mergesort",
        ).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["selected"] = ranked["rank"] <= 3
        ranked["selection_reason"] = np.where(ranked["selected"], "top_three_distinct_models", "below_top_three")
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def selected_simulations(
    *, variant: str, history_days: int | None, family: str, selected: pd.DataFrame,
    fit_prices: pd.DataFrame, portfolio_prices: pd.DataFrame, close_returns: pd.DataFrame,
    open_returns: pd.DataFrame, volatility_cache: dict[int, pd.DataFrame], same: pd.DataFrame,
    midnight: pd.DataFrame, maker: pd.Series, taker: pd.Series, tradable: pd.DataFrame,
) -> dict[str, tuple[Simulation, pd.DataFrame]]:
    needed = set(selected.loc[(selected.variant == variant) & (selected.family == family) & selected.selected, "config_id"])
    by_model = selected.loc[selected.config_id.isin(needed)].groupby("model")
    output: dict[str, tuple[Simulation, pd.DataFrame]] = {}
    index = portfolio_prices.index
    arrays = (open_returns.to_numpy(float), same.to_numpy(float), midnight.to_numpy(float), maker.to_numpy(float), taker.to_numpy(float))
    for model in candidate_models(family, fit_prices, portfolio_prices, maker, taker, history_days):
        if model.name not in by_model.groups:
            continue
        configs = selected.loc[by_model.groups[model.name]].drop_duplicates("config_id")
        risk_window = int(model.config["volatility_window"])
        for cap, group in configs.groupby("ticker_risk_cap"):
            unit = fast_risk_unit(model.forecast, close_returns, volatility_cache[risk_window], float(cap), risk_window).where(tradable, 0.0).to_numpy(float)
            for row in group.itertuples():
                sim = simulate_arrays(
                    unit, index, *arrays, target_vol=float(row.target_vol), gross_cap=float(row.gross_cap),
                    rebalance=row.rebalance, detail=True, columns=portfolio_prices.columns,
                )
                output[row.config_id] = (sim, model.forecast)
    missing = needed - set(output)
    if missing:
        raise RuntimeError(f"Selected configurations were not reconstructed: {sorted(missing)}")
    return output


def simulation_from_held(
    held: pd.DataFrame, open_returns: pd.DataFrame, same: pd.DataFrame, midnight: pd.DataFrame,
    maker: pd.Series, taker: pd.Series,
) -> Simulation:
    trades = held.diff().abs().fillna(held.abs())
    fee_matrix = trades.mul(taker, axis=1)
    slip_matrix = trades * HEADLINE_SLIPPAGE_BPS / 10_000.0
    funding_matrix = held * same + held.shift(1).fillna(0) * midnight
    gross_matrix = held * open_returns
    contribution = gross_matrix - fee_matrix - slip_matrix + funding_matrix
    return Simulation(
        net=contribution.sum(axis=1), turnover=trades.sum(axis=1), funding=funding_matrix.sum(axis=1),
        fee=fee_matrix.sum(axis=1), slippage=slip_matrix.sum(axis=1), gross_return=gross_matrix.sum(axis=1),
        target=held.copy(), held=held, contribution=contribution,
    )


def equal_risk_held(simulations: list[tuple[Simulation, float]], target_vol: float = 0.20) -> pd.DataFrame:
    unit_positions = [simulation.held / selected_target for simulation, selected_target in simulations]
    raw = sum(unit_positions) / len(unit_positions) * target_vol
    gross = raw.abs().sum(axis=1)
    raw = raw.mul((3.0 / gross.replace(0, np.nan)).clip(upper=1).fillna(0), axis=0)
    raw = raw.clip(-0.40, 0.40)
    return raw


def rich_metrics(simulation: Simulation, mask: pd.Series, btc_return: pd.Series) -> dict:
    x = simulation.net[mask].dropna()
    if x.empty:
        return {}
    equity = (1 + x).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1)
    annual_return = float(x.mean() * 365)
    annual_vol = float(x.std(ddof=1) * np.sqrt(365))
    downside = float(x[x < 0].std(ddof=1) * np.sqrt(365)) if (x < 0).sum() > 1 else np.nan
    weekly = (1 + x).resample("W").prod().sub(1)
    monthly = (1 + x).resample("ME").prod().sub(1)
    aligned_btc = btc_return.reindex(x.index).fillna(0)
    btc_var = float(aligned_btc.var())
    beta = float(x.cov(aligned_btc) / btc_var) if btc_var > 0 else np.nan
    max_dd = float(drawdown.min())
    return {
        "annual_return": annual_return, "annual_volatility": annual_vol,
        "net_sharpe": annual_return / annual_vol if annual_vol > 0 else np.nan,
        "sortino": annual_return / downside if downside > 0 else np.nan,
        "calmar": annual_return / abs(max_dd) if max_dd < 0 else np.nan,
        "max_drawdown": max_dd, "cumulative_return": float(equity.iloc[-1] - 1),
        "worst_week": float(weekly.min()), "worst_month": float(monthly.min()),
        "cvar_95": float(x[x <= x.quantile(.05)].mean()), "positive_day_rate": float((x > 0).mean()),
        "annual_turnover": float(simulation.turnover[mask].mean() * 365),
        "fees": float(simulation.fee[mask].sum()), "slippage": float(simulation.slippage[mask].sum()),
        "funding": float(simulation.funding[mask].sum()), "btc_beta": beta,
        "btc_correlation": float(x.corr(aligned_btc)), "observations": int(len(x)),
    }


def rolling_metrics(simulation: Simulation, strategy: str, variant: str, btc_return: pd.Series, windows: list[int]) -> pd.DataFrame:
    rows = []
    for window in windows:
        minimum = max(window // 2, 30)
        ret = simulation.net.rolling(window, min_periods=minimum).mean() * 365
        vol = simulation.net.rolling(window, min_periods=minimum).std() * np.sqrt(365)
        covariance = simulation.net.rolling(window, min_periods=minimum).cov(btc_return)
        btc_variance = btc_return.rolling(window, min_periods=minimum).var()
        frame = pd.DataFrame({
            "timestamp": simulation.net.index, "variant": variant, "strategy": strategy, "window_days": window,
            "annual_return": ret, "annual_volatility": vol, "sharpe": ret.div(vol.replace(0, np.nan)),
            "btc_beta": covariance.div(btc_variance.replace(0, np.nan)),
            "btc_correlation": simulation.net.rolling(window, min_periods=minimum).corr(btc_return),
            "annual_turnover": simulation.turnover.rolling(window, min_periods=minimum).mean() * 365,
            "funding_return": simulation.funding.rolling(window, min_periods=minimum).sum(),
        })
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    locked = json.loads(LOCKED.read_text(encoding="utf-8"))
    all_prices = load_full_price_panel()
    fit_prices = all_prices.loc[:, all_prices.notna().sum() >= 90]
    universe = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    members = universe.loc[universe.member.astype(str).str.lower().isin(["true", "1"]), "symbol"]
    symbols = [symbol for symbol in members if symbol in fit_prices]
    portfolio_prices = fit_prices[symbols]
    open_prices = pd.read_parquet(INPUTS / "portfolio_open_prices.parquet").reindex(index=portfolio_prices.index, columns=symbols)
    if open_prices.notna().sum().lt(90).any():
        raise ValueError("Every portfolio asset requires at least 90 daily opens")
    events = pd.read_parquet(INPUTS / "funding_events.parquet")
    events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True)
    if events.duplicated(["symbol", "funding_time"]).any():
        raise ValueError("Duplicated funding events are not permitted")
    gaps = events.sort_values(["symbol", "funding_time"]).groupby("symbol")["funding_time"].diff().dt.total_seconds().div(3600)
    if (gaps > 24).any() or set(symbols) - set(events.symbol.unique()):
        raise ValueError("Complete funding history is required for every portfolio asset")
    commissions = pd.read_csv(INPUTS / "commission_rates.csv").set_index("symbol")
    maker = commissions["maker"].reindex(symbols).fillna(.0002)
    taker = commissions["taker"].reindex(symbols).fillna(.0004)
    same, midnight = funding_coefficients(portfolio_prices, events)
    tradable = funding_tradability_mask(portfolio_prices.index, symbols, events)
    close_returns = portfolio_prices.pct_change(fill_method=None).fillna(0)
    open_returns = open_prices.shift(-1).div(open_prices).sub(1).fillna(0)
    volatility_cache = {window: close_returns.rolling(window, min_periods=window).std() * np.sqrt(365) for window in VOL_WINDOWS}
    folds = fold_definitions(portfolio_prices.index)
    variants = [(item["id"], item.get("training_history_days")) for item in locked["methodology_variants"]]

    candidate_path = OUT / "candidate_score_ledger.parquet"
    if candidate_path.exists():
        candidate_path.unlink()
    writer = None
    best_records = []
    for variant, history_days in variants:
        for family in ("time_series", "breakout"):
            best, writer = scan_candidates(
                variant=variant, history_days=history_days, family=family, fit_prices=fit_prices,
                portfolio_prices=portfolio_prices, close_returns=close_returns, open_returns=open_returns,
                volatility_cache=volatility_cache, same=same, midnight=midnight, maker=maker, taker=taker,
                folds=folds, writer=writer,
                tradable=tradable,
            )
            best_records.extend({"variant": variant, **record} for record in best.values())
    if writer is not None:
        writer.close()
    selection = select_rosters(best_records)
    selection.to_parquet(OUT / "rolling_selection_ledger.parquet", index=False)
    selected_only = selection[selection.selected].copy()
    selected_only.to_json(OUT / "selected_outer_fold_rosters.json", orient="records", indent=2, date_format="iso")
    tested_configuration_count = write_configuration_catalog(candidate_path, selection)

    simulations: dict[tuple[str, str], dict[str, tuple[Simulation, pd.DataFrame]]] = {}
    for variant, history_days in variants:
        for family in ("time_series", "breakout"):
            simulations[(variant, family)] = selected_simulations(
                variant=variant, history_days=history_days, family=family, selected=selected_only,
                fit_prices=fit_prices, portfolio_prices=portfolio_prices, close_returns=close_returns,
                open_returns=open_returns, volatility_cache=volatility_cache, same=same, midnight=midnight,
                maker=maker, taker=taker, tradable=tradable,
            )

    state_rows = []
    position_rows = []
    strategy_sims: dict[tuple[str, str], Simulation] = {}
    fold_metric_rows = []
    for variant, _ in variants:
        stitched: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
        for fold in folds:
            fold_mask = pd.Series((portfolio_prices.index >= fold["test_start"]) & (portfolio_prices.index <= fold["test_end"]), index=portfolio_prices.index)
            family_held = {}
            for family in ("time_series", "breakout"):
                roster = selected_only[(selected_only.variant == variant) & (selected_only.fold == fold["fold"]) & (selected_only.family == family)].sort_values("rank")
                selected_sims = []
                for row in roster.itertuples():
                    sim, forecast = simulations[(variant, family)][row.config_id]
                    strategy = f"{family}_slot_{int(row.rank)}"
                    selected_sims.append((sim, float(row.target_vol)))
                    bucket = stitched.setdefault(strategy, {
                        "held": pd.DataFrame(0.0, index=portfolio_prices.index, columns=symbols),
                        "forecast": pd.DataFrame(np.nan, index=portfolio_prices.index, columns=symbols),
                    })
                    bucket["held"].loc[fold_mask] = sim.held.loc[fold_mask]
                    bucket["forecast"].loc[fold_mask] = forecast.loc[fold_mask]
                family_held[family] = equal_risk_held(selected_sims)
                family_name = f"{family}_ensemble"
                bucket = stitched.setdefault(family_name, {"held": pd.DataFrame(0.0, index=portfolio_prices.index, columns=symbols)})
                bucket["held"].loc[fold_mask] = family_held[family].loc[fold_mask]
            combined = (family_held["time_series"] + family_held["breakout"]) / 2
            bucket = stitched.setdefault("combined_50_50", {"held": pd.DataFrame(0.0, index=portfolio_prices.index, columns=symbols)})
            bucket["held"].loc[fold_mask] = combined.loc[fold_mask]

        for strategy, payload in stitched.items():
            sim = simulation_from_held(payload["held"], open_returns, same, midnight, maker, taker)
            strategy_sims[(variant, strategy)] = sim
            forecast = payload.get("forecast")
            for fold in folds:
                mask = pd.Series((portfolio_prices.index >= fold["test_start"]) & (portfolio_prices.index <= fold["test_end"]), index=portfolio_prices.index)
                fold_metric_rows.append({"variant": variant, "strategy": strategy, "fold": fold["fold"], **rich_metrics(sim, mask, open_returns.get("BTCUSDT", pd.Series(0., index=portfolio_prices.index)))})
            gross = sim.held.abs().sum(axis=1)
            top5 = sim.held.abs().apply(lambda row: row.nlargest(5).sum(), axis=1).div(gross.replace(0, np.nan)).fillna(0)
            daily = pd.DataFrame({
                "timestamp": portfolio_prices.index, "variant": variant, "strategy": strategy,
                "net_return": sim.net, "gross_return": sim.gross_return, "fee": sim.fee,
                "slippage": sim.slippage, "funding": sim.funding, "turnover": sim.turnover,
                "gross_exposure": gross, "net_exposure": sim.held.sum(axis=1),
                "long_count": (sim.held > 0).sum(axis=1), "short_count": (sim.held < 0).sum(axis=1),
                "top5_concentration": top5,
            })
            state_rows.append(daily)
            long = sim.held.stack().rename("held_weight").reset_index()
            long.columns = ["timestamp", "symbol", "held_weight"]
            long["variant"] = variant; long["strategy"] = strategy
            contributions = sim.contribution.stack().rename("net_return_contribution").reset_index(drop=True)
            long["net_return_contribution"] = contributions
            if forecast is not None:
                long["forecast"] = forecast.stack(dropna=False).reset_index(drop=True)
            else:
                long["forecast"] = np.nan
            position_rows.append(long)

    daily_state = pd.concat(state_rows, ignore_index=True)
    positions = pd.concat(position_rows, ignore_index=True)
    daily_state.to_parquet(OUT / "daily_portfolio_state.parquet", index=False)
    positions.to_parquet(OUT / "daily_ticker_positions.parquet", index=False)
    pd.DataFrame(fold_metric_rows).to_parquet(OUT / "outer_fold_metrics.parquet", index=False)

    stitched_metric_rows = []
    rolling_rows = []
    btc_return = open_returns.get("BTCUSDT", pd.Series(0., index=portfolio_prices.index))
    outer_mask = pd.Series(portfolio_prices.index >= folds[0]["test_start"], index=portfolio_prices.index)
    for (variant, strategy), sim in strategy_sims.items():
        stitched_metric_rows.append({"variant": variant, "strategy": strategy, **rich_metrics(sim, outer_mask, btc_return)})
        rolling_rows.append(rolling_metrics(sim, strategy, variant, btc_return, locked["rolling_metric_windows_days"]))
    pd.DataFrame(stitched_metric_rows).to_parquet(OUT / "stitched_oos_metrics.parquet", index=False)
    pd.concat(rolling_rows, ignore_index=True).to_parquet(OUT / "rolling_metrics.parquet", index=False)
    daily_state.pivot(index="timestamp", columns=["variant", "strategy"], values="net_return").to_parquet(OUT / "stitched_oos_returns.parquet")

    seed_roster = locked["prospective_seed_roster"]
    with sqlite3.connect(OUT / "production_shadow.sqlite") as connection:
        pd.DataFrame([
            {"family": family, **record}
            for family in ("time_series_momentum", "breakout") for record in seed_roster[family]
        ]).to_sql("seed_roster", connection, if_exists="replace", index=False)
        pd.DataFrame(columns=["timestamp", "model", "symbol", "forecast", "target_weight", "config_hash", "emitted_at"]).to_sql(
            "emitted_signals", connection, if_exists="replace", index=False
        )

    manifest = {
        "schema_version": 1, "methodology_version": METHODOLOGY,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "locked_config": str(LOCKED.relative_to(ROOT)),
        "locked_config_sha256": hashlib.sha256(LOCKED.read_bytes()).hexdigest(),
        "input_hashes": {"fit_prices": digest_frame(fit_prices), "portfolio_prices": digest_frame(portfolio_prices),
                         "open_prices": digest_frame(open_prices), "funding": digest_frame(events)},
        "variants": [{"id": variant, "history_days": history_days} for variant, history_days in variants],
        "outer_folds": json_safe(folds), "rolling_metric_windows_days": locked["rolling_metric_windows_days"],
        "fit_assets": fit_prices.shape[1], "portfolio_assets": portfolio_prices.shape[1],
        "candidate_fold_scores": int(len(pd.read_parquet(candidate_path, columns=["config_id"]))),
        "tested_configuration_count": tested_configuration_count,
        "selected_outer_fold_configurations": int(len(selected_only)),
        "selection_uses_holdout_metrics": False,
        "headline_costs": {"taker_share": HEADLINE_TAKER_SHARE, "slippage_bps": HEADLINE_SLIPPAGE_BPS},
        "current_universe_historical_fallback": True, "prospective_shadow_initialized": True,
        "tradability": "day_after_first_fully_priced_funding_event",
        "funding_coverage": {
            "events": int(len(events)),
            "finite_mark_prices": int(pd.to_numeric(events["mark_price"], errors="coerce").notna().sum()),
            "unresolved_pre_eligibility": int(pd.to_numeric(events["mark_price"], errors="coerce").isna().sum()),
            "unresolved_post_eligibility": 0,
            "mark_price_sources": {
                str(source): int(count)
                for source, count in events.get("mark_price_source", pd.Series("unspecified", index=events.index))
                .fillna("unspecified").value_counts().items()
            },
        },
        "prospective_observations": 0, "production_signals_changed": False,
    }
    (OUT / "study_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    print(json.dumps(json_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
