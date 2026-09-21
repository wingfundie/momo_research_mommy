from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.research.config import (
    CARVER_BREAKOUT_HORIZONS,
    CARVER_BREAKOUT_SCALARS,
    CARVER_BREAKOUT_TURNOVER,
)
from momo_bot.strategies.forecasts import calc_breakout_forecast
from scripts.execute_complete_crypto_study import (
    GROSS_CAPS,
    REFITS,
    RULES,
    SPECS,
    TARGETS,
    TICKER_CAPS,
    VOL_WINDOWS,
    capped_simplex,
    components as ewmac_components,
    digest_frame,
    evaluate,
    evaluate_fast,
    fast_risk_unit,
    json_safe,
    load_full_price_panel,
    metrics,
    refit_indices,
)
from scripts.execute_full_crypto_study import funding_coefficients


OUT = ROOT / "data_store/crypto_momentum_research/complete_results_carver5_v2"
INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"
V1 = ROOT / "data_store/crypto_momentum_research/complete_results"
HORIZONS = tuple(CARVER_BREAKOUT_HORIZONS)
FIXED_SCALARS = dict(CARVER_BREAKOUT_SCALARS)
EXPECTED_TURNOVER = dict(CARVER_BREAKOUT_TURNOVER)
METHODOLOGY_VERSION = "breakout-carver5-v2"
FIXED_FDM = 1.20
FDM_CAP = 2.5
COST_BUDGET_SR = 0.15


@dataclass
class ModelEntry:
    name: str
    config: dict
    forecast_builder: Callable[[float, float], pd.DataFrame]
    unit_builder: Callable[[float, float, float], pd.DataFrame] | None = None


def breakout_components(prices: pd.DataFrame) -> np.ndarray:
    """Canonical Carver components: complete channel followed by h/4 EWMA."""
    frames = [calc_breakout_forecast(prices, horizon=h, smoothing_divisor=4) for h in HORIZONS]
    return np.stack([frame.to_numpy(float) for frame in frames], axis=2)


def _step_path(steps: np.ndarray, index: pd.DatetimeIndex, default: np.ndarray) -> np.ndarray:
    shape = steps.shape[1:]
    flat = steps.reshape(len(index), -1)
    frame = pd.DataFrame(flat, index=index).ffill()
    fallback = np.broadcast_to(default, shape).reshape(-1)
    frame = frame.fillna(pd.Series(fallback, index=frame.columns))
    return frame.to_numpy().reshape((len(index),) + shape)


def parameter_path(
    raw: np.ndarray,
    returns: np.ndarray,
    index: pd.DatetimeIndex,
    schedule: str,
    *,
    fixed_scalars: bool,
    fit_weights: bool,
    shrink: float = 1.0,
    weight_cap: float = 0.20,
    smoothing: int = 1,
    history_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal scalar, weight and forecast-correlation paths."""
    t_count, _, rule_count = raw.shape
    scalar_steps = np.full((t_count, rule_count), np.nan)
    weight_steps = np.full((t_count, rule_count), np.nan)
    corr_steps = np.full((t_count, rule_count, rule_count), np.nan)
    fixed = np.asarray([FIXED_SCALARS[h] for h in HORIZONS], float)
    equal = np.ones(rule_count) / rule_count
    for location in refit_indices(index, schedule):
        start = 0 if history_days is None else int(index.searchsorted(index[location] - pd.Timedelta(days=history_days), side="left"))
        history = raw[start:location]
        scalars = fixed if fixed_scalars else np.asarray(
            [10.0 / np.nanmedian(np.abs(history[:, :, rule])) for rule in range(rule_count)]
        )
        scaled = np.clip(history * scalars[None, None, :], -20, 20)
        if fit_weights:
            pnl = scaled[:-2] / 10.0 * returns[start + 2:location, :, None]
            mean = np.nanmean(pnl, axis=0)
            std = np.nanstd(pnl, axis=0, ddof=1)
            scores = np.nanmedian(
                np.maximum(np.divide(mean, std, out=np.zeros_like(mean), where=std > 0), 0), axis=0
            )
            fitted = scores / scores.sum() if scores.sum() > 0 else equal
            weights = capped_simplex((1 - shrink) * fitted + shrink * equal, weight_cap)
        else:
            weights = equal
        flat = scaled.reshape(-1, rule_count)
        corr = pd.DataFrame(flat).corr().fillna(0).to_numpy()
        np.fill_diagonal(corr, 1.0)
        activation = min(location + 1, t_count - 1)
        scalar_steps[activation] = scalars
        weight_steps[activation] = weights
        corr_steps[activation] = corr
    scalars = _step_path(scalar_steps, index, fixed if fixed_scalars else np.ones(rule_count))
    weights = _step_path(weight_steps, index, equal)
    corr = _step_path(corr_steps, index, np.eye(rule_count))
    if smoothing > 1:
        scalars = pd.DataFrame(scalars, index=index).ewm(span=smoothing, adjust=False).mean().to_numpy()
        weights_frame = pd.DataFrame(weights, index=index).ewm(span=smoothing, adjust=False).mean()
        weights = weights_frame.div(weights_frame.sum(axis=1), axis=0).to_numpy()
    return scalars, weights, corr


def eligibility_path(
    prices: pd.DataFrame,
    maker: pd.Series,
    taker: pd.Series,
    *,
    taker_share: float,
    slippage_bps: float,
    risk_window: int,
    schedule: str,
) -> tuple[np.ndarray, list[dict]]:
    index = prices.index
    rule_turnover = np.asarray([EXPECTED_TURNOVER[h] for h in HORIZONS])
    annual_vol = prices.pct_change(fill_method=None).rolling(risk_window, min_periods=risk_window).std() * np.sqrt(365)
    blended = maker * (1 - taker_share) + taker * taker_share + slippage_bps / 10_000.0
    steps = np.full((len(index), prices.shape[1], len(HORIZONS)), np.nan)
    records: list[dict] = []
    for location in refit_indices(index, schedule):
        source = max(location - 1, 0)
        vol = annual_vol.iloc[source].to_numpy(float)
        cost_rate = blended.reindex(prices.columns).to_numpy(float)
        cost_sr = np.divide(cost_rate, vol, out=np.full_like(cost_rate, np.inf), where=np.isfinite(vol) & (vol > 0))
        maximum = np.divide(COST_BUDGET_SR, cost_sr, out=np.zeros_like(cost_sr), where=np.isfinite(cost_sr) & (cost_sr > 0))
        maximum = np.where(cost_rate <= 0, np.inf, maximum)
        eligible = rule_turnover[None, :] <= maximum[:, None]
        activation = min(location + 1, len(index) - 1)
        steps[activation] = eligible.astype(float)
        for ticker_number, symbol in enumerate(prices.columns):
            for rule_number, horizon in enumerate(HORIZONS):
                records.append({
                    "refit_timestamp": index[location], "activation_timestamp": index[activation],
                    "symbol": symbol, "horizon": horizon,
                    "expected_turnover": float(rule_turnover[rule_number]),
                    "cost_per_trade_sr": float(cost_sr[ticker_number]),
                    "maximum_turnover": float(maximum[ticker_number]),
                    "eligible": bool(eligible[ticker_number, rule_number]),
                    "reason": "within_cost_budget" if eligible[ticker_number, rule_number] else "exceeds_cost_budget",
                    "taker_share": taker_share, "slippage_bps": slippage_bps,
                    "risk_window": risk_window, "refit_schedule": schedule,
                })
    eligible = _step_path(steps, index, np.zeros((prices.shape[1], len(HORIZONS)))) > 0.5
    return eligible, records


def combine_components(
    raw: np.ndarray,
    scalars: np.ndarray,
    weights: np.ndarray,
    correlations: np.ndarray,
    eligible: np.ndarray,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    *,
    fixed_fdm: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    scaled = np.clip(raw * scalars[:, None, :], -20, 20)
    active = eligible & np.isfinite(scaled)
    effective = weights[:, None, :] * active
    total = effective.sum(axis=2, keepdims=True)
    effective = np.divide(effective, total, out=np.zeros_like(effective), where=total > 0)
    combined = np.nansum(np.where(active, scaled, 0) * effective, axis=2)
    if fixed_fdm is None:
        variance = np.einsum("tnh,thk,tnk->tn", effective, correlations, effective)
        fdm = np.divide(1.0, np.sqrt(variance), out=np.ones_like(variance), where=variance > 0)
        fdm = np.clip(fdm, 1.0, FDM_CAP)
    else:
        count = active.sum(axis=2)
        fdm = np.where(count > 1, fixed_fdm, 1.0)
    combined = np.where(active.any(axis=2), np.clip(combined * fdm, -20, 20), 0.0)
    return pd.DataFrame(combined / 20.0, index=index, columns=columns), fdm, effective


def normalize_existing_unit(unit: pd.DataFrame, returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    lagged = unit.shift(1).fillna(0)
    preliminary = (lagged * returns).sum(axis=1)
    predicted = preliminary.rolling(lookback, min_periods=lookback).std() * np.sqrt(365)
    return unit.div(predicted.replace(0, np.nan), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)


def build_ewmac_reference(fit_prices: pd.DataFrame, portfolio_prices: pd.DataFrame) -> pd.DataFrame:
    raw = ewmac_components(fit_prices, 60)
    returns = fit_prices.pct_change(fill_method=None).to_numpy(float)
    scalars, _, dfm = __import__("scripts.execute_complete_crypto_study", fromlist=["calibration_path"]).calibration_path(
        raw, returns, fit_prices.index, "quarterly", 1.0, 0.20, 1
    )
    locations = [fit_prices.columns.get_loc(column) for column in portfolio_prices]
    equal = np.ones((len(fit_prices), len(RULES))) / len(RULES)
    final = __import__("scripts.execute_complete_crypto_study", fromlist=["combine"]).combine(
        raw[:, locations], scalars, equal, dfm
    )
    return pd.DataFrame(final / 20.0, index=fit_prices.index, columns=portfolio_prices.columns)


def deterministic_select(rows: list[dict], family: str) -> dict | None:
    candidates = [
        row for row in rows
        if row["config"].get("family") == family
        and row["metrics"]["validation"]["net_sharpe"] is not None
        and row["config"].get("taker_share") == 1.0
        and row["config"].get("slippage_bps") == 5.0
        and row["config"].get("activation") == "next_open"
        and row["config"].get("selection_eligible", True)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (
        -row["metrics"]["validation"]["net_sharpe"],
        row["metrics"]["validation"]["annual_turnover"],
        row["config"].get("gross_cap", 0), row["config"].get("target_vol", 0), row["config_id"],
    ))[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_prices = load_full_price_panel()
    fit_prices = all_prices.loc[:, all_prices.notna().sum() >= 90]
    universe = pd.read_csv(INPUTS / "current_universe_snapshot.csv")
    members = universe.loc[universe.member.astype(str).str.lower().isin(["true", "1"]), "symbol"]
    portfolio_symbols = [symbol for symbol in members if symbol in fit_prices]
    portfolio_prices = fit_prices[portfolio_symbols]
    locations = [fit_prices.columns.get_loc(column) for column in portfolio_prices]
    open_prices = pd.read_parquet(INPUTS / "portfolio_open_prices.parquet").reindex(
        index=portfolio_prices.index, columns=portfolio_symbols
    )
    if open_prices.notna().sum().lt(90).any():
        raise ValueError("v2 next-open study requires at least 90 opens for every portfolio asset")
    events = pd.read_parquet(INPUTS / "funding_events.parquet")
    events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True)
    if events.duplicated(["symbol", "funding_time"]).any():
        raise ValueError("v2 headline study refuses duplicated funding events")
    funding_gaps = events.sort_values(["symbol", "funding_time"]).groupby("symbol")["funding_time"].diff().dt.total_seconds().div(3600)
    if (funding_gaps > 24).any():
        raise ValueError("v2 headline study refuses incomplete required funding history")
    missing_funding = sorted(set(portfolio_symbols) - set(events.symbol.unique()))
    if missing_funding:
        raise ValueError(f"v2 headline study is missing funding history for {missing_funding}")
    commission_path = INPUTS / "commission_rates.csv"
    if not commission_path.exists():
        raise FileNotFoundError("commission archive is required")
    commissions = pd.read_csv(commission_path).set_index("symbol")
    maker = commissions["maker"].reindex(portfolio_symbols).fillna(0.0002)
    taker = commissions["taker"].reindex(portfolio_symbols).fillna(0.0004)
    same_coeff, midnight_coeff = funding_coefficients(portfolio_prices, events)

    fit_returns = fit_prices.pct_change(fill_method=None).to_numpy(float)
    close_returns = portfolio_prices.pct_change(fill_method=None).fillna(0)
    open_forward_returns = open_prices.shift(-1).div(open_prices).sub(1).fillna(0)
    close_forward_returns = portfolio_prices.shift(-1).div(portfolio_prices).sub(1).fillna(0)
    volatility_cache = {
        window: close_returns.rolling(window, min_periods=window).std() * np.sqrt(365) for window in VOL_WINDOWS
    }
    masks = {
        "post_training": pd.Series(portfolio_prices.index >= portfolio_prices.index.min() + pd.Timedelta(days=365), index=portfolio_prices.index),
        "validation": pd.Series((portfolio_prices.index >= "2024-01-01") & (portfolio_prices.index < "2026-01-01"), index=portfolio_prices.index),
        "holdout_2026": pd.Series(portfolio_prices.index >= "2026-01-01", index=portfolio_prices.index),
    }
    raw_fit = breakout_components(fit_prices)
    raw = raw_fit[:, locations]
    pd.DataFrame(
        np.concatenate([raw[:, :, rule] for rule in range(len(HORIZONS))], axis=1),
        index=fit_prices.index,
        columns=pd.MultiIndex.from_product([[f"breakout_{h}" for h in HORIZONS], portfolio_symbols], names=["rule", "symbol"]),
    ).to_parquet(OUT / "raw_smoothed_breakout_components.parquet")

    eligibility_cache: dict[tuple, tuple[np.ndarray, list[dict]]] = {}
    def eligibility(taker_share: float, slippage: float, risk_window: int, schedule: str):
        key = (taker_share, slippage, risk_window, schedule)
        if key not in eligibility_cache:
            eligibility_cache[key] = eligibility_path(
                portfolio_prices, maker, taker, taker_share=taker_share, slippage_bps=slippage,
                risk_window=risk_window, schedule=schedule,
            )
        return eligibility_cache[key]

    models: list[ModelEntry] = []
    parameter_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    diagnostic_paths: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    def parameters(schedule: str, fixed: bool, fitted: bool, shrink: float, cap: float, smoothing: int):
        key = (schedule, fixed, fitted, shrink, cap, smoothing)
        if key not in parameter_cache:
            parameter_cache[key] = parameter_path(
                raw_fit, fit_returns, fit_prices.index, schedule, fixed_scalars=fixed,
                fit_weights=fitted, shrink=shrink, weight_cap=cap, smoothing=smoothing,
            )
        scalars, weights, corr = parameter_cache[key]
        return scalars, weights, corr

    def add_breakout_model(name: str, config: dict, param_key: tuple, fixed_fdm: float | None = None):
        schedule, fixed, fitted, shrink, cap, smoothing = param_key
        scalars, weights, corr = parameters(*param_key)
        risk_window = int(config["volatility_window"])
        cache: dict[tuple[float, float], pd.DataFrame] = {}
        def builder(taker_share: float, slippage: float) -> pd.DataFrame:
            key = (taker_share, slippage)
            if key not in cache:
                active, _ = eligibility(taker_share, slippage, risk_window, schedule)
                forecast, fdm, effective = combine_components(
                    raw, scalars, weights, corr, active, fit_prices.index, portfolio_prices.columns,
                    fixed_fdm=fixed_fdm,
                )
                cache[key] = forecast
                if taker_share == 1.0 and slippage == 5.0:
                    diagnostic_paths[name] = (fdm, effective)
            return cache[key]
        models.append(ModelEntry(name, config, builder))

    for risk_window in VOL_WINDOWS:
        add_breakout_model(
            f"breakout_carver5_equal_causal_fdm_vol{risk_window}",
            {"family": "breakout", "method": "carver_fixed_scalars_equal", "fdm_mode": "causal_expanding",
             "refit": "quarterly", "volatility_window": risk_window, "selection_eligible": True},
            ("quarterly", True, False, 1.0, 0.20, 1),
        )
        add_breakout_model(
            f"breakout_carver5_equal_fixed_fdm_vol{risk_window}",
            {"family": "breakout", "method": "carver_fixed_scalars_equal", "fdm_mode": "fixed_1.20",
             "refit": "quarterly", "volatility_window": risk_window, "selection_eligible": True},
            ("quarterly", True, False, 1.0, 0.20, 1), fixed_fdm=FIXED_FDM,
        )
        for schedule in REFITS:
            add_breakout_model(
                f"breakout_crypto_equal_{schedule}_vol{risk_window}",
                {"family": "breakout", "method": "crypto_expanding_scalars_equal", "fdm_mode": "causal_expanding",
                 "refit": schedule, "volatility_window": risk_window, "selection_eligible": True},
                (schedule, False, False, 1.0, 0.20, 1),
            )
            for spec_name, (shrink, cap, smoothing) in SPECS.items():
                add_breakout_model(
                    f"breakout_crypto_{spec_name}_{schedule}_vol{risk_window}",
                    {"family": "breakout", "method": "crypto_expanding_scalars_shrunk_weights",
                     "weight_spec": spec_name, "fdm_mode": "causal_expanding", "refit": schedule,
                     "volatility_window": risk_window, "selection_eligible": True},
                    (schedule, False, True, shrink, cap, smoothing),
                )

    fixed = np.asarray([FIXED_SCALARS[h] for h in HORIZONS])
    scaled_fixed = np.clip(raw * fixed[None, None, :], -20, 20)
    pd.DataFrame(
        np.concatenate([scaled_fixed[:, :, rule] for rule in range(len(HORIZONS))], axis=1),
        index=fit_prices.index,
        columns=pd.MultiIndex.from_product([[f"breakout_{h}" for h in HORIZONS], portfolio_symbols], names=["rule", "symbol"]),
    ).to_parquet(OUT / "carver_scaled_component_forecasts.parquet")
    for risk_window in VOL_WINDOWS:
        for rule, horizon in enumerate(HORIZONS):
            frame = pd.DataFrame(scaled_fixed[:, :, rule] / 20, index=fit_prices.index, columns=portfolio_symbols)
            models.append(ModelEntry(
                f"breakout_control_h{horizon}_vol{risk_window}",
                {"family": "breakout_control", "method": "single_horizon_smoothed_fixed_scalar",
                 "horizon": horizon, "volatility_window": risk_window, "selection_eligible": False},
                lambda _t, _s, frame=frame: frame,
            ))

    # Static Optuna is reconstructed on corrected smoothed components but cannot be selected.
    legacy = pd.read_pickle(ROOT / "data_store/optimized_breakout_params.pkl")
    legacy_frame = np.full(raw.shape[:2], np.nan)
    valid_legacy = 0
    for ticker_number, symbol in enumerate(portfolio_symbols):
        payload = legacy.get(symbol, {})
        weights = np.asarray(payload.get("weights", []), float)
        horizons = tuple(payload.get("horizons", HORIZONS))
        valid = payload.get("status", "success") == "success" and horizons == HORIZONS and len(weights) == len(HORIZONS) and np.isfinite(weights).all() and weights.sum() > 0
        if valid:
            weights = weights / weights.sum(); valid_legacy += 1
        else:
            weights = np.ones(len(HORIZONS)) / len(HORIZONS)
        legacy_frame[:, ticker_number] = np.nansum(scaled_fixed[:, ticker_number] * weights, axis=1)
    legacy_forecast = pd.DataFrame(np.clip(legacy_frame * FIXED_FDM, -20, 20) / 20, index=fit_prices.index, columns=portfolio_symbols)
    models.append(ModelEntry(
        "breakout_legacy_optuna_smoothed",
        {"family": "breakout_legacy", "method": "static_per_ticker_optuna_corrected_components",
         "optimization_scope": "legacy_static_full_sample_reference", "valid_optuna_tickers": valid_legacy,
         "volatility_window": 90, "selection_eligible": False},
        lambda _t, _s: legacy_forecast,
    ))

    ewmac = build_ewmac_reference(fit_prices, portfolio_prices)
    primary_breakout = next(model for model in models if model.name == "breakout_carver5_equal_causal_fdm_vol90")
    forecast_blend_cache: dict[tuple, pd.DataFrame] = {}
    def forecast_blend(taker_share: float, slippage: float) -> pd.DataFrame:
        key = (taker_share, slippage)
        if key not in forecast_blend_cache:
            bo = primary_breakout.forecast_builder(taker_share, slippage)
            pair = np.stack([bo.to_numpy(float) * 20, ewmac.to_numpy(float) * 20], axis=2)
            corr = np.full((len(pair), 2, 2), np.nan)
            steps = np.full_like(corr, np.nan)
            for location in refit_indices(fit_prices.index, "quarterly"):
                flat = pair[:location].reshape(-1, 2)
                value = pd.DataFrame(flat).corr().fillna(0).to_numpy(); np.fill_diagonal(value, 1)
                steps[min(location + 1, len(pair) - 1)] = value
            corr = _step_path(steps, fit_prices.index, np.eye(2))
            weight = np.array([0.5, 0.5])
            variance = np.einsum("h,thk,k->t", weight, corr, weight)
            fdm = np.clip(np.divide(1, np.sqrt(variance), out=np.ones_like(variance), where=variance > 0), 1, FDM_CAP)
            blended = np.clip(pair.mean(axis=2) * fdm[:, None], -20, 20) / 20
            forecast_blend_cache[key] = pd.DataFrame(blended, index=fit_prices.index, columns=portfolio_symbols)
        return forecast_blend_cache[key]

    combo_base = {"family": "divergent_combination", "ewmac_model": "ts_equal_vol60", "breakout_model": primary_breakout.name,
                  "volatility_window": 60, "selection_eligible": True}
    models.append(ModelEntry("divergent_forecast_blend", {**combo_base, "construction": "forecast_50_50"}, forecast_blend))

    def combo_unit(kind: str, taker_share: float, slippage: float, cap: float) -> pd.DataFrame:
        vol = volatility_cache[60]
        bo = primary_breakout.forecast_builder(taker_share, slippage)
        bo_unit = fast_risk_unit(bo, close_returns, vol, cap, 60)
        ts_unit = fast_risk_unit(ewmac, close_returns, vol, cap, 60)
        sleeve = normalize_existing_unit((bo_unit + ts_unit) / 2, close_returns, 60)
        if kind == "sleeve":
            return sleeve
        forecast_unit = fast_risk_unit(forecast_blend(taker_share, slippage), close_returns, vol, cap, 60)
        return normalize_existing_unit((forecast_unit + sleeve) / 2, close_returns, 60)

    models.append(ModelEntry(
        "divergent_sleeve_blend", {**combo_base, "construction": "equal_risk_sleeves_50_50"}, forecast_blend,
        lambda t, s, cap: combo_unit("sleeve", t, s, cap),
    ))
    models.append(ModelEntry(
        "divergent_meta_blend", {**combo_base, "construction": "forecast_and_sleeve_meta_50_50"}, forecast_blend,
        lambda t, s, cap: combo_unit("meta", t, s, cap),
    ))

    rows: list[dict] = []
    daily_store: dict[str, pd.Series] = {}
    funding_store: dict[str, pd.Series] = {}
    held_store: dict[str, pd.DataFrame] = {}
    maker_array, taker_array = maker.to_numpy(float), taker.to_numpy(float)
    returns_array = open_forward_returns.to_numpy(float)
    same_array, midnight_array = same_coeff.to_numpy(float), midnight_coeff.to_numpy(float)
    signal_contract = {
        "horizons": list(HORIZONS),
        "forecast_scalars": FIXED_SCALARS,
        "expected_annual_turnover": EXPECTED_TURNOVER,
        "smoothing": "ewma_h_over_4",
        "smoothing_divisor": 4,
        "component_cap": 20.0,
        "final_forecast_cap": 20.0,
        "cost_budget_sr": COST_BUDGET_SR,
        "speed_filter_formula": "maximum_turnover=0.15/(one_way_cost_rate/annualized_volatility)",
        "perpetual_roll_cost": 0.0,
        "fdm_cap": FDM_CAP,
        "320_day_rule_excluded": True,
    }
    for model_number, model in enumerate(models, 1):
        print(f"model {model_number}/{len(models)} {model.name}", flush=True)
        risk_window = int(model.config.get("volatility_window", 90))
        raw_headline = model.forecast_builder(1.0, 5.0)
        units = {
            cap: model.unit_builder(1.0, 5.0, cap) if model.unit_builder else fast_risk_unit(
                raw_headline, close_returns, volatility_cache[risk_window], cap, risk_window
            ) for cap in TICKER_CAPS
        }
        for cap, unit in units.items():
            unit_array = unit.to_numpy(float)
            for target in TARGETS:
                for gross in GROSS_CAPS:
                    for frequency in ("daily", "weekly", "monthly"):
                        split = evaluate_fast(
                            returns_array, unit_array, portfolio_prices.index, target, gross, frequency, 1.0, 5.0,
                            same_array, midnight_array, masks, maker_array, taker_array,
                        )
                        cfg = {**model.config, **signal_contract, "model": model.name, "methodology_version": METHODOLOGY_VERSION,
                               "target_vol": target,
                               "gross_cap": gross, "ticker_risk_cap": cap, "rebalance": frequency,
                               "activation": "next_open", "taker_share": 1.0, "slippage_bps": 5.0,
                               "funding_mode": "event_level_actual_history", "return_convention": "next_open_open_to_open"}
                        config_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20]
                        rows.append({"config_id": config_id, "config": cfg, "metrics": split})
        for taker_share in (0.0, 0.5, 1.0):
            for slippage in (0.0, 2.0, 5.0, 10.0):
                forecast = model.forecast_builder(taker_share, slippage)
                unit = model.unit_builder(taker_share, slippage, 0.10) if model.unit_builder else fast_risk_unit(
                    forecast, close_returns, volatility_cache[risk_window], 0.10, risk_window
                )
                split, net, held, funding = evaluate(
                    portfolio_prices, open_forward_returns, unit, 0.15, 1.0, "daily", "next_open",
                    taker_share, slippage, same_coeff, midnight_coeff, masks, maker, taker,
                )
                cfg = {**model.config, **signal_contract, "model": model.name, "methodology_version": METHODOLOGY_VERSION,
                       "target_vol": 0.15,
                       "gross_cap": 1.0, "ticker_risk_cap": 0.10, "rebalance": "daily",
                       "activation": "next_open", "taker_share": taker_share, "slippage_bps": slippage,
                       "funding_mode": "event_level_actual_history", "return_convention": "next_open_open_to_open"}
                config_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20]
                rows.append({"config_id": config_id, "config": cfg, "metrics": split})
                if taker_share == 1.0 and slippage == 5.0:
                    daily_store[model.name] = net; funding_store[model.name] = funding; held_store[model.name] = held
        forecast = model.forecast_builder(1.0, 5.0)
        unit = model.unit_builder(1.0, 5.0, 0.10) if model.unit_builder else fast_risk_unit(
            forecast, close_returns, volatility_cache[risk_window], 0.10, risk_window
        )
        split, *_ = evaluate(
            portfolio_prices, close_forward_returns, unit, 0.15, 1.0, "daily", "next_close", 1.0, 5.0,
            same_coeff, midnight_coeff, masks, maker, taker,
        )
        cfg = {**model.config, **signal_contract, "model": model.name, "methodology_version": METHODOLOGY_VERSION,
               "target_vol": 0.15,
               "gross_cap": 1.0, "ticker_risk_cap": 0.10, "rebalance": "daily", "activation": "next_close",
               "taker_share": 1.0, "slippage_bps": 5.0, "funding_mode": "event_level_actual_history",
               "return_convention": "next_close_close_to_close"}
        rows.append({"config_id": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20],
                     "config": cfg, "metrics": split})

    rows = list({row["config_id"]: row for row in rows}.values())
    selected = [item for item in (deterministic_select(rows, "breakout"), deterministic_select(rows, "divergent_combination")) if item]
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    input_hashes = {
        "fit_prices": digest_frame(fit_prices), "portfolio_prices": digest_frame(portfolio_prices),
        "portfolio_open_prices": digest_frame(open_prices), "funding": digest_frame(events),
        "v1_manifest": hashlib.sha256((V1 / "study_manifest.json").read_bytes()).hexdigest(),
    }
    manifest = {
        "schema_version": 4, "methodology_version": METHODOLOGY_VERSION, "generated_at": generated_at,
        "input_hashes": input_hashes, "horizons": list(HORIZONS), "smoothing": "ewma_h_over_4",
        "fixed_scalars": FIXED_SCALARS, "expected_turnover": EXPECTED_TURNOVER,
        "cost_budget_sr": COST_BUDGET_SR, "primary_fdm": "causal_expanding", "fixed_fdm_sensitivity": FIXED_FDM,
        "320_day_rule_excluded": True, "fit_universe_assets": fit_prices.shape[1],
        "portfolio_assets": portfolio_prices.shape[1], "configuration_count": len(rows),
        "selection_period": ["2024-01-01", "2025-12-31"],
        "holdout_period": ["2026-01-01", str(portfolio_prices.index.max())],
        "selection_rule": "combined_breakout_only_validation_then_turnover_and_risk_tiebreak",
        "universe_limitation": "current_universe_historical_fallback", "prospective_oos": False,
        "shadow_only": True, "production_signals_changed": False,
    }
    (OUT / "tested_configurations.json").write_text(json.dumps(json_safe(rows), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    (OUT / "selected_configurations.json").write_text(json.dumps(json_safe(selected), indent=2, allow_nan=False), encoding="utf-8")
    deployable = {row["config"]["family"]: {"config_id": row["config_id"], **row["config"]} for row in selected}
    (OUT / "deployable_configurations.json").write_text(json.dumps(json_safe(deployable), indent=2), encoding="utf-8")
    (OUT / "study_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    # Persist expensive simulation checkpoints before presentation-oriented flattening.
    pd.DataFrame(daily_store).to_parquet(OUT / "default_daily_returns.parquet")
    pd.DataFrame(funding_store).to_parquet(OUT / "default_daily_funding.parquet")
    flat = []
    for row in rows:
        record = {"config_id": row["config_id"], **row["config"]}
        for split, values in row["metrics"].items():
            record.update({f"{split}_{key}": value for key, value in values.items()})
        for key, value in list(record.items()):
            if isinstance(value, (dict, list, tuple)):
                record[key] = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
        flat.append(record)
    pd.DataFrame(flat).to_parquet(OUT / "tested_configurations.parquet", index=False)
    pd.DataFrame({name: frame.iloc[-1] for name, frame in held_store.items()}).T.to_csv(OUT / "latest_positions.csv")
    headline_active, headline_eligibility = eligibility(1.0, 5.0, 90, "quarterly")
    all_eligibility = [record for _, records in eligibility_cache.values() for record in records]
    pd.DataFrame(all_eligibility).drop_duplicates().to_parquet(OUT / "breakout_eligibility.parquet", index=False)
    pd.DataFrame(headline_eligibility).to_csv(OUT / "breakout_eligibility_headline.csv", index=False)
    primary_name = "breakout_carver5_equal_causal_fdm_vol90"
    primary_fdm, primary_weights = diagnostic_paths[primary_name]
    pd.DataFrame(primary_fdm, index=fit_prices.index, columns=portfolio_symbols).to_parquet(OUT / "primary_fdm_history.parquet")
    pd.DataFrame(
        np.concatenate([primary_weights[:, :, rule] for rule in range(len(HORIZONS))], axis=1),
        index=fit_prices.index,
        columns=pd.MultiIndex.from_product([[f"breakout_{h}" for h in HORIZONS], portfolio_symbols], names=["rule", "symbol"]),
    ).to_parquet(OUT / "primary_effective_rule_weights.parquet")

    signal_parts = []
    model_map = {model.name: model for model in models}
    for selected_item in selected:
        selected_name = selected_item["config"]["model"]
        selected_model = model_map[selected_name]
        selected_forecast = selected_model.forecast_builder(1.0, 5.0) * 20
        selected_held = held_store[selected_name]
        if selected_name in diagnostic_paths:
            selected_fdm = diagnostic_paths[selected_name][0]
            risk_window = int(selected_model.config.get("volatility_window", 90))
            schedule = selected_model.config.get("refit", "quarterly")
            selected_active = eligibility(1.0, 5.0, risk_window, schedule)[0]
        else:
            selected_fdm = np.full((len(fit_prices), len(portfolio_symbols)), np.nan)
            selected_active = headline_active
        for ticker_number, symbol in enumerate(portfolio_symbols):
            frame = pd.DataFrame({
                "timestamp": fit_prices.index, "symbol": symbol, "model": selected_name,
                "forecast": selected_forecast[symbol].to_numpy(),
                "target_position": selected_held[symbol].to_numpy(),
                "fdm": selected_fdm[:, ticker_number],
                "eligible_horizons": [json.dumps([h for rule, h in enumerate(HORIZONS) if selected_active[row, ticker_number, rule]]) for row in range(len(fit_prices))],
            })
            frame["quality_flag"] = np.where(frame.eligible_horizons.eq("[]"), "no_cost_eligible_rule",
                                              np.where(frame.forecast.notna(), "valid", "warmup"))
            signal_parts.append(frame)
    signal_records_frame = pd.concat(signal_parts, ignore_index=True)
    signal_records_frame.to_parquet(OUT / "daily_signal_records.parquet", index=False)

    # Reconciled common-risk ticker attribution for the selected standalone and combinations.
    attribution_rows = []
    for name in [row["config"]["model"] for row in selected]:
        held = held_store[name]
        trades = held.diff().abs().fillna(held.abs())
        funding = held * same_coeff + held.shift(1).fillna(0) * midnight_coeff
        contribution = held * open_forward_returns - trades.mul(taker, axis=1) - trades * 0.0005 + funding
        for symbol in portfolio_symbols:
            attribution_rows.append({"model": name, "symbol": symbol,
                                     "net_return_contribution": float(contribution.loc[masks["post_training"], symbol].sum()),
                                     "funding_return": float(funding.loc[masks["post_training"], symbol].sum()),
                                     "turnover": float(trades.loc[masks["post_training"], symbol].sum())})
    attribution_frame = pd.DataFrame(attribution_rows)
    attribution_frame.to_parquet(OUT / "ticker_attribution.parquet", index=False)

    metadata_path = INPUTS / "universe_metadata.parquet"
    metadata = pd.read_parquet(metadata_path).set_index("symbol") if metadata_path.exists() else pd.DataFrame(index=portfolio_symbols)
    volume = metadata.get("median_quote_volume_30d", pd.Series(index=portfolio_symbols, dtype=float)).reindex(portfolio_symbols)
    capacity_rows = []
    group_rows = []
    analysis_meta = pd.DataFrame(index=portfolio_symbols)
    analysis_meta["sector"] = metadata.get("sector", pd.Series("Unclassified", index=portfolio_symbols)).reindex(portfolio_symbols).fillna("Unclassified")
    analysis_meta["liquidity"] = volume
    analysis_meta["volatility"] = close_returns.rolling(90, min_periods=90).std().iloc[-1] * np.sqrt(365)
    for column in ("liquidity", "volatility"):
        try:
            analysis_meta[f"{column}_group"] = pd.qcut(analysis_meta[column], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
        except ValueError:
            analysis_meta[f"{column}_group"] = "Unavailable"
    for name in [row["config"]["model"] for row in selected]:
        held = held_store[name]
        trades = held.diff().abs().fillna(held.abs()).loc[masks["post_training"]]
        for participation in (.01, .05, .10):
            limits = np.divide(participation * volume.to_numpy(float)[None, :], trades.to_numpy(float),
                               out=np.full(trades.shape, np.nan), where=trades.to_numpy(float) > 0)
            daily_capacity = np.nanmin(limits, axis=1)
            finite = daily_capacity[np.isfinite(daily_capacity)]
            capacity_rows.append({"model": name, "participation": participation,
                                  "median_capacity_usd": float(np.nanmedian(finite)) if len(finite) else None,
                                  "fifth_percentile_capacity_usd": float(np.nanpercentile(finite, 5)) if len(finite) else None})
        model_attr = attribution_frame[attribution_frame.model.eq(name)].set_index("symbol")
        for grouping in ("sector", "liquidity_group", "volatility_group"):
            for label, symbols in analysis_meta.groupby(grouping, dropna=False).groups.items():
                block = model_attr.reindex(list(symbols))
                group_rows.append({"model": name, "grouping": grouping, "group": str(label),
                                   "asset_count": len(symbols),
                                   "net_return_contribution": float(block.net_return_contribution.sum()),
                                   "funding_return": float(block.funding_return.sum()),
                                   "turnover": float(block.turnover.sum())})
    pd.DataFrame(capacity_rows).to_csv(OUT / "capacity_analysis.csv", index=False)
    pd.DataFrame(group_rows).to_parquet(OUT / "group_attribution.parquet", index=False)

    run_id = hashlib.sha256(json.dumps({"generated_at": generated_at, "inputs": input_hashes}, sort_keys=True).encode()).hexdigest()[:24]
    with sqlite3.connect(OUT / "crypto_breakout_research.sqlite") as connection:
        pd.DataFrame([{"run_id": run_id, "generated_at": generated_at, "manifest_json": json.dumps(json_safe(manifest), sort_keys=True)}]).to_sql("study_runs", connection, if_exists="append", index=False)
        pd.DataFrame([{"run_id": run_id, "config_id": row["config_id"], "config_json": json.dumps(json_safe(row["config"]), sort_keys=True), "metrics_json": json.dumps(json_safe(row["metrics"]), sort_keys=True)} for row in rows]).to_sql("tested_configurations", connection, if_exists="append", index=False)
        eligibility_frame = pd.DataFrame(all_eligibility).drop_duplicates(); eligibility_frame.insert(0, "run_id", run_id)
        eligibility_frame.to_sql("breakout_eligibility", connection, if_exists="append", index=False)
        ledger_signals = signal_records_frame.copy(); ledger_signals.insert(0, "run_id", run_id)
        ledger_signals.to_sql("signal_records", connection, if_exists="append", index=False, chunksize=10000)
        ledger_ticker = attribution_frame.copy(); ledger_ticker.insert(0, "run_id", run_id)
        ledger_ticker.to_sql("ticker_attribution", connection, if_exists="append", index=False)
        ledger_groups = pd.DataFrame(group_rows); ledger_groups.insert(0, "run_id", run_id)
        ledger_groups.to_sql("group_attribution", connection, if_exists="append", index=False)
        ledger_returns = pd.DataFrame(daily_store).rename_axis("timestamp").stack().rename("net_return").reset_index()
        ledger_returns.insert(0, "run_id", run_id); ledger_returns.to_sql("daily_returns", connection, if_exists="append", index=False)
        ledger_funding = pd.DataFrame(funding_store).rename_axis("timestamp").stack().rename("funding_return").reset_index()
        ledger_funding.insert(0, "run_id", run_id); ledger_funding.to_sql("daily_funding", connection, if_exists="append", index=False)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
