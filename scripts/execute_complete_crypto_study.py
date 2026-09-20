from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.research.signals import (
    basket_rank_weights, beta_constrained_weights, breakout_forecast, btc_hedged_weights,
    btc_residualized_forecasts, continuous_rank_weights, cross_sectional_percentiles,
    long_only_rank_weights, rolling_betas,
)
from momo_bot.research.universe import STABLE_BASES
from scripts.execute_full_crypto_study import funding_coefficients

OUT = ROOT / "data_store/crypto_momentum_research/complete_results"
INPUTS = ROOT / "data_store/crypto_momentum_research/inputs"
RULES = ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128))
VOL_WINDOWS = (60, 90, 180, 360)
REFITS = ("quarterly", "semiannual", "annual", "frozen")
SPECS = {
    "shrink_75": (.75, .30, 90),
    "shrink_80_primary": (.80, .25, 125),
    "shrink_90": (.90, .25, 125),
}
TARGETS = (.15, .20, .25, .30, .35, .40)
GROSS_CAPS = (1., 1.5, 2., 2.5, 3.)
TICKER_CAPS = tuple(x / 100 for x in range(5, 41, 5))
REBALANCES = ("daily", "weekly", "monthly")
EXCLUDED_NON_STANDARD_CONTRACTS = {"DEFIUSDT"}


def digest_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def load_full_price_panel() -> pd.DataFrame:
    refreshed = pd.read_pickle(ROOT / "data_store/crypto_research_prices_1d.pkl").sort_index()
    legacy_path = ROOT / "data_store/crypto_tickers_1d.pkl"
    if not legacy_path.exists():
        return refreshed
    try:
        legacy = pd.read_pickle(legacy_path)
    except TypeError:
        # Compatibility for a pandas-3 StringDtype pickle opened by pandas 2.2.
        import pandas.core.arrays.string_ as string_module
        original = string_module.StringDtype

        class CompatibleStringDtype(original):
            def __init__(self, storage=None, na_value=pd.NA):
                super().__init__(storage)

        string_module.StringDtype = CompatibleStringDtype
        pd.StringDtype = CompatibleStringDtype
        legacy = pd.read_pickle(legacy_path)
    legacy.index = pd.to_datetime(legacy.index, utc=True).tz_localize(None)
    refreshed.index = pd.to_datetime(refreshed.index, utc=True).tz_localize(None)
    panel = refreshed.combine_first(legacy)
    usable = [column for column in panel.columns
              if str(column).endswith("USDT") and column not in EXCLUDED_NON_STANDARD_CONTRACTS
              and str(column)[:-4].upper() not in STABLE_BASES]
    return panel[usable].sort_index()


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


def components(prices: pd.DataFrame, vol_window: int) -> np.ndarray:
    daily_price_vol = prices.pct_change(fill_method=None).rolling(vol_window, min_periods=vol_window).std() * prices.shift(1)
    result = []
    for fast, slow in RULES:
        fast_ma = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
        slow_ma = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()
        result.append(((fast_ma - slow_ma) / daily_price_vol.replace(0, np.nan)).to_numpy(float))
    return np.stack(result, axis=2)


def refit_indices(index: pd.DatetimeIndex, schedule: str) -> list[int]:
    freq = {"quarterly": "QS", "semiannual": "2QS", "annual": "YS", "frozen": "YS"}[schedule]
    output = []
    for stamp in pd.date_range(index.min(), index.max(), freq=freq):
        location = int(index.searchsorted(stamp, side="left"))
        if location >= 365:
            output.append(location)
            if schedule == "frozen": break
    return output


def capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    values = np.maximum(np.nan_to_num(values, nan=0), 0)
    values = values / values.sum() if values.sum() > 0 else np.ones_like(values) / len(values)
    result = np.zeros_like(values); free = np.ones(len(values), dtype=bool); remaining = 1.
    while free.any():
        proposal = values[free] / values[free].sum() * remaining
        over = proposal > cap + 1e-12
        if not over.any(): result[free] = proposal; break
        names = np.where(free)[0][over]; result[names] = cap; free[names] = False; remaining = 1 - result[~free].sum()
    return result


def calibration_path(raw: np.ndarray, returns: np.ndarray, index: pd.DatetimeIndex, schedule: str,
                     shrink: float, weight_cap: float, smoothing: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_count, _, rule_count = raw.shape
    scalar_steps = np.full((t_count, rule_count), np.nan)
    weight_steps = np.full((t_count, rule_count), np.nan)
    dfm_steps = np.full(t_count, np.nan)
    for location in refit_indices(index, schedule):
        history = raw[:location]
        scalars = np.array([10 / np.nanmedian(np.abs(history[:, :, r])) for r in range(rule_count)])
        scaled = np.clip(history * scalars, -20, 20)
        # Forecasts are known only after the daily close. Use the following full
        # close-to-close interval for pooled fitting, never the contemporaneous overnight.
        pnl = scaled[:-2] / 10 * returns[2:location, :, None]
        mean = np.nanmean(pnl, axis=0); std = np.nanstd(pnl, axis=0, ddof=1)
        scores = np.nanmedian(np.maximum(np.divide(mean, std, out=np.zeros_like(mean), where=std > 0), 0), axis=0)
        raw_weights = scores / scores.sum() if scores.sum() > 0 else np.ones(rule_count) / rule_count
        equal = np.ones(rule_count) / rule_count
        weights = capped_simplex((1 - shrink) * raw_weights + shrink * equal, weight_cap)
        flat = scaled.reshape(-1, rule_count)
        corr = pd.DataFrame(flat).corr().fillna(0).to_numpy(); np.fill_diagonal(corr, 1)
        variance = float(weights @ corr @ weights)
        dfm = min(1 / math.sqrt(variance), 2.5) if variance > 0 else 1.
        activation = min(location + 1, t_count - 1)
        scalar_steps[activation] = scalars; weight_steps[activation] = weights; dfm_steps[activation] = dfm
    equal = np.ones(rule_count) / rule_count
    scalars = pd.DataFrame(scalar_steps, index=index).ffill().fillna(1.0)
    weights = pd.DataFrame(weight_steps, index=index).ffill().fillna(pd.Series(equal))
    dfm = pd.Series(dfm_steps, index=index).ffill().fillna(1.0)
    if smoothing > 1:
        scalars = scalars.ewm(span=smoothing, adjust=False).mean()
        weights = weights.ewm(span=smoothing, adjust=False).mean(); weights = weights.div(weights.sum(axis=1), axis=0)
        dfm = dfm.ewm(span=smoothing, adjust=False).mean()
    return scalars.to_numpy(), weights.to_numpy(), dfm.to_numpy()


def combine(raw: np.ndarray, scalars: np.ndarray, weights: np.ndarray, dfm: np.ndarray) -> np.ndarray:
    scaled = np.clip(raw * scalars[:, None, :], -20, 20)
    valid = np.isfinite(scaled)
    numerator = np.nansum(scaled * weights[:, None, :], axis=2)
    denominator = np.sum(valid * weights[:, None, :], axis=2)
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return np.clip(result * dfm[:, None], -20, 20)


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=1)
    l = np.where(valid, left, np.nan); r = np.where(valid, right, np.nan)
    l = l - np.nanmean(l, axis=1, keepdims=True); r = r - np.nanmean(r, axis=1, keepdims=True)
    numerator = np.nansum(l * r, axis=1); denominator = np.sqrt(np.nansum(l * l, axis=1) * np.nansum(r * r, axis=1))
    return np.divide(numerator, denominator, out=np.full(len(left), np.nan), where=(denominator > 0) & (count >= 5))


def ic_weighted_forecast(scaled_components: np.ndarray, prices: pd.DataFrame, outcome_prices: pd.DataFrame,
                         horizons: tuple[int, ...]) -> pd.DataFrame:
    index, columns = prices.index, prices.columns
    component_ranks = np.stack([pd.DataFrame(scaled_components[:, :, r], index=index, columns=columns).rank(axis=1, pct=True).to_numpy()
                                for r in range(scaled_components.shape[2])], axis=2)
    future_ranks = {h: outcome_prices.shift(-(h + 1)).div(outcome_prices.shift(-1)).sub(1).rank(axis=1, pct=True).to_numpy()
                    for h in horizons}
    steps = np.full((len(index), scaled_components.shape[2]), np.nan)
    for location in refit_indices(index, "quarterly"):
        scores = []
        for rule in range(scaled_components.shape[2]):
            horizon_ics = []
            for horizon in horizons:
                usable = max(location - horizon - 1, 0)
                horizon_ics.append(np.nanmean(row_correlation(component_ranks[:usable, :, rule], future_ranks[horizon][:usable])))
            scores.append(max(float(np.nanmean(horizon_ics)), 0.0))
        raw_weights = np.asarray(scores); raw_weights = raw_weights / raw_weights.sum() if raw_weights.sum() > 0 else np.ones(len(scores)) / len(scores)
        weights = capped_simplex(.20 * raw_weights + .80 * np.ones(len(scores)) / len(scores), .25)
        steps[min(location + 1, len(index) - 1)] = weights
    weights = pd.DataFrame(steps, index=index).ffill().fillna(pd.Series(np.ones(5) / 5)).ewm(span=125, adjust=False).mean()
    weights = weights.div(weights.sum(axis=1), axis=0).to_numpy()
    valid = np.isfinite(scaled_components)
    numerator = np.nansum(scaled_components * weights[:, None, :], axis=2)
    denominator = np.sum(valid * weights[:, None, :], axis=2)
    final = np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)
    return pd.DataFrame(np.clip(final, -20, 20), index=index, columns=columns)


def signed_normalized(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.div(frame.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def buffered_basket(percentiles: pd.DataFrame, fraction: float, buffer: float) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=percentiles.index, columns=percentiles.columns)
    previous_long: set[str] = set(); previous_short: set[str] = set()
    for timestamp, row in percentiles.iterrows():
        entrants_long = set(row[row >= 1 - fraction].index); entrants_short = set(row[row <= fraction].index)
        keep_long = {s for s in previous_long if pd.notna(row[s]) and row[s] >= 1 - fraction - buffer}
        keep_short = {s for s in previous_short if pd.notna(row[s]) and row[s] <= fraction + buffer}
        longs, shorts = entrants_long | keep_long, entrants_short | keep_short
        if longs: result.loc[timestamp, list(longs)] = .5 / len(longs)
        if shorts: result.loc[timestamp, list(shorts)] = -.5 / len(shorts)
        previous_long, previous_short = longs, shorts
    return result


def fast_risk_unit(raw: pd.DataFrame, returns: pd.DataFrame, vol: pd.DataFrame, cap: float, lookback: int) -> pd.DataFrame:
    x = np.nan_to_num(raw.to_numpy(float), nan=0); absolute = np.abs(x)
    totals = absolute.sum(axis=1, keepdims=True); budgets = np.divide(absolute, totals, out=np.zeros_like(absolute), where=totals > 0)
    active_count = (absolute > 0).sum(axis=1); effective = np.maximum(cap, np.divide(1., active_count, out=np.ones_like(active_count, float), where=active_count > 0))
    result = np.zeros_like(budgets); free = absolute > 0; remaining = np.ones(len(raw))
    for _ in range(raw.shape[1]):
        denominator = (budgets * free).sum(axis=1)
        proposal = np.divide(budgets, denominator[:, None], out=np.zeros_like(budgets), where=denominator[:, None] > 0) * remaining[:, None]
        over = free & (proposal > effective[:, None] + 1e-12)
        if not over.any(): result[free] = proposal[free]; break
        result[over] = np.broadcast_to(effective[:, None], result.shape)[over]
        free[over] = False; remaining = 1 - result.sum(axis=1)
    risk_units = np.sign(x) * result
    inverse = np.divide(risk_units, vol.to_numpy(float), out=np.zeros_like(risk_units), where=np.isfinite(vol.to_numpy(float)) & (vol.to_numpy(float) > 0))
    preliminary = np.nansum(np.roll(inverse, 1, axis=0) * np.nan_to_num(returns.to_numpy(float)), axis=1); preliminary[0] = 0
    predicted = pd.Series(preliminary, index=raw.index).rolling(lookback, min_periods=lookback).std().to_numpy() * np.sqrt(365)
    unit = np.divide(inverse, predicted[:, None], out=np.zeros_like(inverse), where=np.isfinite(predicted[:, None]) & (predicted[:, None] > 0))
    return pd.DataFrame(unit, index=raw.index, columns=raw.columns)


def held_weights(targets: pd.DataFrame, frequency: str, activation: str) -> pd.DataFrame:
    index = targets.index
    if frequency == "daily": mask = np.ones(len(index), bool)
    elif frequency == "weekly": mask = index.weekday == 0
    else: mask = ~index.to_period("M").duplicated()
    if activation not in {"next_open", "next_close"}:
        raise ValueError("activation must be next_open or next_close")
    return targets.where(pd.Series(mask, index=index), np.nan, axis=0).ffill().shift(1).fillna(0)


def metrics(series: pd.Series, turnover: pd.Series, funding: pd.Series, mask: pd.Series) -> dict:
    x = series[mask].dropna(); eq = (1 + x).cumprod(); vol = x.std() * np.sqrt(365); ret = x.mean() * 365
    return {"annual_return": float(ret), "annual_volatility": float(vol), "net_sharpe": float(ret / vol) if vol > 0 else None,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "cumulative_return": float(eq.iloc[-1] - 1),
            "annual_turnover": float(turnover[mask].mean() * 365), "funding_return": float(funding[mask].sum()), "observations": int(len(x))}


def evaluate(prices, returns, unit, target, gross_cap, frequency, activation, taker_share, slippage,
             same_coeff, midnight_coeff, masks, maker_rates, taker_rates):
    desired = unit * target; gross = desired.abs().sum(axis=1)
    desired = desired.mul((gross_cap / gross.replace(0, np.nan)).clip(upper=1).fillna(0), axis=0)
    held = held_weights(desired, frequency, activation)
    gross_return = (held * returns).sum(axis=1)
    trades = held.diff().abs().fillna(held.abs())
    turnover = trades.sum(axis=1)
    blended = taker_rates * taker_share + maker_rates * (1 - taker_share)
    fee = trades.mul(blended, axis=1).sum(axis=1)
    slip = turnover * slippage / 10000
    funding_position = held if activation == "next_open" else held.shift(1).fillna(0)
    funding = (funding_position * same_coeff).sum(axis=1) + (funding_position.shift(1).fillna(0) * midnight_coeff).sum(axis=1)
    net = gross_return - fee - slip + funding
    return {name: metrics(net, turnover, funding, mask) for name, mask in masks.items()}, net, held, funding


def evaluate_fast(returns_array: np.ndarray, unit_array: np.ndarray, index: pd.DatetimeIndex, target: float,
                  gross_cap: float, frequency: str, taker_share: float, slippage: float,
                  same_array: np.ndarray, midnight_array: np.ndarray, masks: dict[str, pd.Series],
                  maker_array: np.ndarray, taker_array: np.ndarray) -> dict:
    gross = np.abs(unit_array).sum(axis=1)
    multiplier = np.minimum(target, np.divide(gross_cap, gross, out=np.full_like(gross, target), where=gross > 0))
    desired = unit_array * multiplier[:, None]
    if frequency == "daily":
        rebalance = np.ones(len(index), dtype=bool)
    elif frequency == "weekly":
        rebalance = index.weekday == 0
    else:
        rebalance = np.asarray(~index.to_period("M").duplicated())
    source = np.maximum.accumulate(np.where(rebalance, np.arange(len(index)), -1))
    held = np.zeros_like(desired)
    delayed_source = np.full(len(index), -1, dtype=int)
    delayed_source[1:] = source[:-1]
    valid = delayed_source >= 0
    held[valid] = desired[delayed_source[valid]]
    trades = np.abs(np.diff(held, axis=0, prepend=np.zeros((1, held.shape[1]))))
    turnover = trades.sum(axis=1)
    blended = taker_array * taker_share + maker_array * (1 - taker_share)
    fee = (trades * blended[None, :]).sum(axis=1)
    funding_matrix = held * same_array + np.vstack([np.zeros((1, held.shape[1])), held[:-1]]) * midnight_array
    funding = funding_matrix.sum(axis=1)
    net = (held * returns_array).sum(axis=1) - fee - turnover * slippage / 10000 + funding
    net_series = pd.Series(net, index=index); turnover_series = pd.Series(turnover, index=index); funding_series = pd.Series(funding, index=index)
    return {name: metrics(net_series, turnover_series, funding_series, mask) for name, mask in masks.items()}


def base_models(fit_prices: pd.DataFrame, portfolio_prices: pd.DataFrame,
                portfolio_open_prices: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict]]:
    fit_returns = fit_prices.pct_change(fill_method=None).to_numpy(float)
    portfolio_locations = [fit_prices.columns.get_loc(c) for c in portfolio_prices]
    models = {}
    primary_components = None
    for vol_window in VOL_WINDOWS:
        raw = components(fit_prices, vol_window)
        for schedule in REFITS:
            for spec_name, (shrink, cap, smoothing) in SPECS.items():
                scalars, weights, dfm = calibration_path(raw, fit_returns, fit_prices.index, schedule, shrink, cap, smoothing)
                final = combine(raw[:, portfolio_locations], scalars, weights, dfm)
                name = f"ts_{spec_name}_{schedule}_vol{vol_window}"
                models[name] = (pd.DataFrame(final, index=fit_prices.index, columns=portfolio_prices.columns) / 20,
                                {"family": "time_series", "rule_weights": spec_name, "refit": schedule, "volatility_window": vol_window})
                if vol_window == 90 and schedule == "quarterly" and spec_name == "shrink_80_primary": primary_components = (raw[:, portfolio_locations], scalars, weights, dfm)
        scalars, _, dfm = calibration_path(raw, fit_returns, fit_prices.index, "quarterly", 1., .20, 1)
        equal = np.ones((len(fit_prices), len(RULES))) / len(RULES)
        final = combine(raw[:, portfolio_locations], scalars, equal, dfm)
        models[f"ts_equal_vol{vol_window}"] = (pd.DataFrame(final, index=fit_prices.index, columns=portfolio_prices.columns) / 20,
                                               {"family": "time_series", "rule_weights": "equal", "refit": "quarterly_scalars", "volatility_window": vol_window})
    # Legacy per-ticker Optuna reference, with explicit equal fallback.
    legacy = pd.read_pickle(ROOT / "data_store/optimized_crypto_weights_carver.pkl")
    raw, scalars, _, _ = primary_components; scaled = np.clip(raw * scalars[:, None, :], -20, 20)
    legacy_final = np.full(scaled.shape[:2], np.nan); fallback = 0
    for j, symbol in enumerate(portfolio_prices):
        payload = legacy.get(symbol, {}); w = np.asarray(payload.get("weights", []), float)
        if len(w) != len(RULES) or not np.isfinite(w).all() or w.sum() <= 0: w = np.ones(len(RULES)) / len(RULES); fallback += 1
        else: w = w / w.sum()
        valid = np.isfinite(scaled[:, j]); legacy_final[:, j] = np.where(valid.any(axis=1), np.nansum(scaled[:, j] * w, axis=1), np.nan)
    models["ts_legacy_optuna"] = (pd.DataFrame(np.clip(legacy_final, -20, 20), index=fit_prices.index, columns=portfolio_prices.columns) / 20,
                                  {"family": "time_series", "rule_weights": "legacy_per_ticker_optuna", "fallback_equal_tickers": fallback, "volatility_window": 90})
    asset_returns = portfolio_prices.pct_change(fill_method=None); betas = rolling_betas(asset_returns, lookback=90)
    raw_primary, scalars, _, _ = primary_components
    scaled_primary = np.clip(raw_primary * scalars[:, None, :], -20, 20)
    component_columns = pd.MultiIndex.from_product([[f"ewmac_{fast}_{slow}" for fast, slow in RULES], portfolio_prices.columns],
                                                   names=["rule", "symbol"])
    component_matrix = np.concatenate([scaled_primary[:, :, rule] for rule in range(len(RULES))], axis=1)
    pd.DataFrame(component_matrix, index=fit_prices.index, columns=component_columns).to_parquet(OUT / "primary_ts_component_forecasts.parquet")
    for ic_name, horizons in {"ic1": (1,), "ic5": (5,), "ic20": (20,), "ic5_20_primary": (5, 20)}.items():
        primary = ic_weighted_forecast(scaled_primary, portfolio_prices, portfolio_open_prices, horizons)
        ranks = cross_sectional_percentiles(primary); continuous = continuous_rank_weights(ranks)
        xs = {
            f"xs_{ic_name}_signed_unhedged": signed_normalized(primary),
            f"xs_{ic_name}_dollar_neutral": continuous,
            f"xs_{ic_name}_basket_10": basket_rank_weights(ranks, .10),
            f"xs_{ic_name}_basket_20": basket_rank_weights(ranks, .20),
            f"xs_{ic_name}_basket_30": basket_rank_weights(ranks, .30),
            f"xs_{ic_name}_long_only_20": long_only_rank_weights(ranks, primary, .20),
            f"xs_{ic_name}_btc_hedged": btc_hedged_weights(continuous, betas),
            f"xs_{ic_name}_beta_constrained": beta_constrained_weights(continuous, betas),
        }
        residual = btc_residualized_forecasts(primary, asset_returns, lookback=90)
        xs[f"xs_{ic_name}_btc_residualized"] = continuous_rank_weights(cross_sectional_percentiles(residual))
        if ic_name == "ic5_20_primary":
            xs["xs_ic5_20_basket20_buffer5"] = buffered_basket(ranks, .20, .05)
            xs["xs_ic5_20_basket20_buffer10"] = buffered_basket(ranks, .20, .10)
        for name, frame in xs.items():
            models[name] = (frame, {"family": "cross_sectional", "construction": name,
                                         "ic_horizons": list(horizons), "rank_buffer": .05 if "buffer5" in name else .10 if "buffer10" in name else 0,
                                         "volatility_window": 90})
    breakouts = {}
    for horizon in (16, 32, 64, 128, 256):
        frame = pd.DataFrame({c: breakout_forecast(portfolio_prices[c], horizon) for c in portfolio_prices}) / 20
        models[f"breakout_{horizon}"] = (frame, {"family": "breakout", "horizon": horizon, "volatility_window": 90})
        breakouts[horizon] = frame
    models["breakout_equal"] = (sum(breakouts.values()) / len(breakouts), {"family": "breakout", "horizons": list(breakouts), "weights": "equal", "volatility_window": 90})
    ts_primary = models["ts_shrink_80_primary_quarterly_vol90"][0]
    xs_primary = models["xs_ic5_20_primary_dollar_neutral"][0]
    models["diagnostic_ts_xs_equal_risk"] = ((signed_normalized(ts_primary) + signed_normalized(xs_primary)) / 2,
                                              {"family": "diagnostic_combination", "sleeves": ["time_series", "cross_sectional"], "weights": "equal_risk_proxy", "volatility_window": 90})
    return models


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_prices = load_full_price_panel()
    fit_prices = all_prices.loc[:, all_prices.notna().sum() >= 90]
    selected = pd.read_csv(INPUTS / "current_universe_snapshot.csv"); members = selected.loc[selected.member.astype(str).str.lower().isin(["true", "1"]), "symbol"]
    portfolio_symbols = [s for s in members if s in fit_prices]
    portfolio_prices = fit_prices[portfolio_symbols]
    events = pd.read_parquet(INPUTS / "funding_events.parquet")
    events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True)
    if events.duplicated(["symbol", "funding_time"]).any():
        raise ValueError("Headline study refuses duplicated funding events")
    funding_gaps = events.sort_values(["symbol", "funding_time"]).groupby("symbol")["funding_time"].diff().dt.total_seconds().div(3600)
    if (funding_gaps > 24).any():
        raise ValueError("Headline study refuses incomplete required funding history")
    missing_funding_symbols = sorted(set(portfolio_symbols) - set(events.symbol.unique()))
    if missing_funding_symbols:
        raise ValueError(f"Headline study is missing funding history for {missing_funding_symbols}")
    commission_path = INPUTS / "commission_rates.csv"
    if not commission_path.exists():
        raise FileNotFoundError("Run commission-rate archival before the complete study")
    commissions = pd.read_csv(commission_path).set_index("symbol")
    maker_rates = commissions["maker"].reindex(portfolio_symbols).fillna(.0002)
    taker_rates = commissions["taker"].reindex(portfolio_symbols).fillna(.0004)
    same_coeff, midnight_coeff = funding_coefficients(portfolio_prices, events)
    open_prices = pd.read_parquet(INPUTS / "portfolio_open_prices.parquet").reindex(index=portfolio_prices.index, columns=portfolio_symbols)
    if open_prices.notna().sum().lt(90).any():
        raise ValueError("Headline next-open study requires at least 90 daily open prices for every portfolio asset")
    models = base_models(fit_prices, portfolio_prices, open_prices)
    close_returns = portfolio_prices.pct_change(fill_method=None).fillna(0)
    open_forward_returns = open_prices.shift(-1).div(open_prices).sub(1).fillna(0)
    close_forward_returns = portfolio_prices.shift(-1).div(portfolio_prices).sub(1).fillna(0)
    volatility_cache = {window: close_returns.rolling(window, min_periods=window).std() * np.sqrt(365) for window in VOL_WINDOWS}
    masks = {"post_training": pd.Series(portfolio_prices.index >= portfolio_prices.index.min() + pd.Timedelta(days=365), index=portfolio_prices.index),
             "validation": pd.Series((portfolio_prices.index >= "2024-01-01") & (portfolio_prices.index < "2026-01-01"), index=portfolio_prices.index),
             "holdout_2026": pd.Series(portfolio_prices.index >= "2026-01-01", index=portfolio_prices.index)}
    rows = []; daily_store = {}; funding_store = {}; latest_store = {}; attribution_store = {}
    attribution_models = {
        "ts_equal_vol90", "ts_shrink_80_primary_quarterly_vol90", "ts_legacy_optuna",
        "xs_ic5_20_primary_dollar_neutral", "xs_ic5_20_basket20_buffer5", "breakout_equal",
        "diagnostic_ts_xs_equal_risk",
    }
    input_hashes = {"fit_prices": digest_frame(fit_prices), "portfolio_prices": digest_frame(portfolio_prices),
                    "portfolio_open_prices": digest_frame(open_prices), "funding": digest_frame(events)}
    returns_array = open_forward_returns.to_numpy(float); same_array = same_coeff.to_numpy(float); midnight_array = midnight_coeff.to_numpy(float)
    maker_array = maker_rates.to_numpy(float); taker_array = taker_rates.to_numpy(float)
    for model_number, (model_name, (raw, model_config)) in enumerate(models.items(), 1):
        print(f"model {model_number}/{len(models)} {model_name}", flush=True)
        risk_window = int(model_config.get("volatility_window", 90))
        model_volatility = volatility_cache[risk_window]
        units = {cap: fast_risk_unit(raw, close_returns, model_volatility, cap, risk_window) for cap in TICKER_CAPS}
        for cap, unit in units.items():
            unit_array = unit.to_numpy(float)
            for target in TARGETS:
                for gross in GROSS_CAPS:
                    for frequency in REBALANCES:
                        if model_config.get("rank_buffer", 0) and frequency != "weekly":
                            continue
                        split_metrics = evaluate_fast(returns_array, unit_array, portfolio_prices.index, target, gross, frequency, 1., 5.,
                                                      same_array, midnight_array, masks, maker_array, taker_array)
                        cfg = {**model_config, "model": model_name, "target_vol": target, "gross_cap": gross, "ticker_risk_cap": cap,
                               "rebalance": frequency, "activation": "next_open", "taker_share": 1., "slippage_bps": 5.,
                               "return_convention": "next_open_open_to_open",
                               "funding_mode": "event_level_actual_history", "commission_source": "binance_symbol_commission_api_with_documented_fallback",
                               "universe": "top100_current_fdv_fallback", "fit_universe_assets": fit_prices.shape[1], "portfolio_assets": portfolio_prices.shape[1]}
                        config_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20]
                        rows.append({"config_id": config_id, "config": cfg, "metrics": split_metrics})
        # Complete cost and delay sensitivities at the default portfolio risk setting.
        unit = units[.10]
        for taker in (0., .5, 1.):
            for slippage in (0., 2., 5., 10.):
                default_frequency = "weekly" if model_config.get("rank_buffer", 0) else "daily"
                split_metrics, net, held, funding = evaluate(portfolio_prices, open_forward_returns, unit, .15, 1., default_frequency, "next_open", taker, slippage, same_coeff, midnight_coeff, masks, maker_rates, taker_rates)
                cfg = {**model_config, "model": model_name, "target_vol": .15, "gross_cap": 1., "ticker_risk_cap": .10,
                       "rebalance": default_frequency, "activation": "next_open", "taker_share": taker, "slippage_bps": slippage,
                       "return_convention": "next_open_open_to_open",
                       "funding_mode": "event_level_actual_history", "commission_source": "binance_symbol_commission_api_with_documented_fallback",
                       "universe": "top100_current_fdv_fallback", "fit_universe_assets": fit_prices.shape[1], "portfolio_assets": portfolio_prices.shape[1]}
                rows.append({"config_id": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20], "config": cfg, "metrics": split_metrics})
                if taker == 1 and slippage == 5:
                    daily_store[model_name] = net; funding_store[model_name] = funding; latest_store[model_name] = held.iloc[-1]
                    if model_name in attribution_models:
                        attribution_store[model_name] = held
        default_frequency = "weekly" if model_config.get("rank_buffer", 0) else "daily"
        split_metrics, *_ = evaluate(portfolio_prices, close_forward_returns, unit, .15, 1., default_frequency, "next_close", 1., 5., same_coeff, midnight_coeff, masks, maker_rates, taker_rates)
        cfg = {**model_config, "model": model_name, "target_vol": .15, "gross_cap": 1., "ticker_risk_cap": .10,
               "rebalance": default_frequency, "activation": "next_close", "taker_share": 1., "slippage_bps": 5.,
               "return_convention": "next_close_close_to_close",
               "funding_mode": "event_level_actual_history", "commission_source": "binance_symbol_commission_api_with_documented_fallback",
               "universe": "top100_current_fdv_fallback", "fit_universe_assets": fit_prices.shape[1], "portfolio_assets": portfolio_prices.shape[1]}
        rows.append({"config_id": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20], "config": cfg, "metrics": split_metrics})
    # Deterministic baselines, evaluated separately without parameter fitting.
    available = portfolio_prices.notna().astype(float); equal = available.div(available.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fdv_snapshot = selected.set_index("symbol")["fully_diluted_valuation"].reindex(portfolio_symbols)
    fdv_weight = available.mul(fdv_snapshot, axis=1); fdv_weight = fdv_weight.div(fdv_weight.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    baselines = {"baseline_equal_basket": equal, "baseline_fdv_weighted": fdv_weight,
                 "baseline_btc_buy_hold": pd.DataFrame({c: (1. if c == "BTCUSDT" else 0.) for c in portfolio_prices}, index=portfolio_prices.index)}
    rng = np.random.default_rng(20260920); random_sign = pd.DataFrame(rng.choice([-1., 1.], size=portfolio_prices.shape), index=portfolio_prices.index, columns=portfolio_prices.columns) * available
    baselines["baseline_no_skill"] = random_sign.div(random_sign.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    for name, held in baselines.items():
        gross_ret = (held * open_forward_returns).sum(axis=1); trades = held.diff().abs().fillna(held.abs()); turnover = trades.sum(axis=1)
        funding = (held * same_coeff).sum(axis=1) + (held.shift(1).fillna(0) * midnight_coeff).sum(axis=1)
        net = gross_ret - trades.mul(taker_rates, axis=1).sum(axis=1) - turnover * .0005 + funding
        split = {key: metrics(net, turnover, funding, mask) for key, mask in masks.items()}
        cfg = {"family": "baseline", "model": name, "activation": "next_open", "return_convention": "next_open_open_to_open",
               "taker_share": 1., "slippage_bps": 5.,
               "funding_mode": "event_level_actual_history", "commission_source": "binance_symbol_commission_api_with_documented_fallback"}
        rows.append({"config_id": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20], "config": cfg, "metrics": split})
        daily_store[name] = net; funding_store[name] = funding; latest_store[name] = held.iloc[-1]
    # Recovery checkpoint: the expensive simulation matrix is complete at this point.
    (OUT / "tested_configurations.checkpoint.json").write_text(
        json.dumps(json_safe(rows), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    pd.DataFrame(daily_store).to_parquet(OUT / "default_daily_returns.checkpoint.parquet")
    pd.DataFrame(funding_store).to_parquet(OUT / "default_daily_funding.checkpoint.parquet")
    # Ticker and cohort attribution for the default risk/cost setting of each primary methodology.
    metadata_path = INPUTS / "universe_metadata.parquet"
    metadata = pd.read_parquet(metadata_path).set_index("symbol") if metadata_path.exists() else pd.DataFrame(index=portfolio_symbols)
    trailing_vol = close_returns.rolling(90, min_periods=90).std().iloc[-1] * np.sqrt(365)
    btc_beta = close_returns.rolling(90, min_periods=90).cov(close_returns.get("BTCUSDT")).iloc[-1].div(
        close_returns.get("BTCUSDT").rolling(90, min_periods=90).var().iloc[-1]
    ) if "BTCUSDT" in close_returns else pd.Series(np.nan, index=portfolio_symbols)
    analysis_meta = pd.DataFrame(index=portfolio_symbols)
    analysis_meta["liquidity"] = metadata.get("median_quote_volume_30d", pd.Series(index=portfolio_symbols, dtype=float)).reindex(portfolio_symbols)
    analysis_meta["sector"] = metadata.get("sector", pd.Series("Unclassified", index=portfolio_symbols)).reindex(portfolio_symbols).fillna("Unclassified")
    listing = pd.to_datetime(metadata.get("listing_time", pd.Series(index=portfolio_symbols, dtype="datetime64[ns, UTC]")), utc=True).reindex(portfolio_symbols).dt.tz_localize(None)
    analysis_meta["listing_age_days"] = (portfolio_prices.index.max() - listing).dt.days
    analysis_meta["btc_beta"] = btc_beta.reindex(portfolio_symbols)
    analysis_meta["volatility"] = trailing_vol.reindex(portfolio_symbols)
    for column in ("liquidity", "listing_age_days", "btc_beta", "volatility"):
        try: analysis_meta[f"{column}_group"] = pd.qcut(analysis_meta[column], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
        except ValueError: analysis_meta[f"{column}_group"] = "Unavailable"
    ticker_rows = []; group_rows = []; capacity_rows = []
    post_training = masks["post_training"]
    for model_name, held in attribution_store.items():
        trades = held.diff().abs().fillna(held.abs())
        fee_matrix = trades.mul(taker_rates, axis=1)
        slip_matrix = trades * .0005
        funding_matrix = held * same_coeff + held.shift(1).fillna(0) * midnight_coeff
        contribution = held * open_forward_returns - fee_matrix - slip_matrix + funding_matrix
        for symbol in portfolio_symbols:
            ticker_rows.append({"model": model_name, "symbol": symbol,
                                "net_return_contribution": float(contribution.loc[post_training, symbol].sum()),
                                "funding_return": float(funding_matrix.loc[post_training, symbol].sum()),
                                "turnover": float(trades.loc[post_training, symbol].sum()),
                                **analysis_meta.loc[symbol].to_dict()})
        for grouping in ("listing_age_days_group", "liquidity_group", "btc_beta_group", "sector", "volatility_group"):
            for label, symbols in analysis_meta.groupby(grouping, dropna=False).groups.items():
                symbols = list(symbols)
                group_rows.append({"model": model_name, "grouping": grouping, "group": str(label), "asset_count": len(symbols),
                                   "net_return_contribution": float(contribution.loc[post_training, symbols].sum().sum()),
                                   "funding_return": float(funding_matrix.loc[post_training, symbols].sum().sum()),
                                   "turnover": float(trades.loc[post_training, symbols].sum().sum())})
        volume = analysis_meta["liquidity"].replace(0, np.nan).to_numpy(float)
        trade_array = trades.loc[post_training].to_numpy(float)
        for participation in (.01, .05, .10):
            limits = np.divide(participation * volume[None, :], trade_array,
                               out=np.full_like(trade_array, np.nan), where=trade_array > 0)
            daily_capacity = np.nanmin(limits, axis=1)
            finite = daily_capacity[np.isfinite(daily_capacity)]
            capacity_rows.append({"model": model_name, "participation_of_median_daily_quote_volume": participation,
                                  "median_capacity_usd": float(np.nanmedian(finite)) if len(finite) else None,
                                  "fifth_percentile_capacity_usd": float(np.nanpercentile(finite, 5)) if len(finite) else None})
    pd.DataFrame(ticker_rows).to_parquet(OUT / "ticker_attribution.parquet", index=False)
    pd.DataFrame(group_rows).to_parquet(OUT / "group_attribution.parquet", index=False)
    pd.DataFrame(capacity_rows).to_csv(OUT / "capacity_analysis.csv", index=False)

    # Reconcile every archived account funding record; absent historical exposure is kept explicitly unmatched.
    actual_path = INPUTS / "actual_funding_income_recent.parquet"
    if actual_path.exists():
        actual = pd.read_parquet(actual_path).copy()
        actual["day"] = pd.to_datetime(actual["timestamp"], utc=True).dt.normalize()
        reference_name = "ts_shrink_80_primary_quarterly_vol90"
        reference_held = attribution_store.get(reference_name)
        if reference_held is not None:
            modeled_daily = (reference_held * same_coeff + reference_held.shift(1).fillna(0) * midnight_coeff)
            modeled_long = modeled_daily.stack().rename("modeled_funding_return").reset_index()
            modeled_long.columns = ["day", "symbol", "modeled_funding_return"]
            modeled_long["day"] = pd.to_datetime(modeled_long["day"], utc=True).dt.normalize()
            actual = actual.merge(modeled_long, on=["day", "symbol"], how="left")
        else:
            actual["modeled_funding_return"] = np.nan
        actual["match_status"] = np.where(actual["modeled_funding_return"].fillna(0).eq(0), "unmatched_no_model_exposure",
                                            np.where(np.sign(actual["income_usd"]) == np.sign(actual["modeled_funding_return"]),
                                                     "direction_agrees_notional_unavailable", "unmatched_direction_disagrees"))
        actual.to_parquet(OUT / "actual_funding_reconciliation.parquet", index=False)
        actual.groupby(["symbol", "match_status"], dropna=False)["income_usd"].agg(["count", "sum"]).reset_index().to_csv(
            OUT / "actual_funding_summary.csv", index=False)
    # Daily signal ledger for the primary methods, including individual ticker strength and ranks.
    signal_frames = []
    for model_name in attribution_models:
        if model_name not in models or model_name not in attribution_store:
            continue
        strength = models[model_name][0] * 20
        rank = strength.rank(axis=1, pct=True)
        position = attribution_store[model_name]
        frame = pd.concat({"forecast": strength, "cross_sectional_percentile": rank, "target_position": position}, axis=1)
        long = frame.stack(level=1, future_stack=True).reset_index().rename(columns={"level_1": "symbol"})
        long.insert(1, "model", model_name)
        long["model_hash"] = hashlib.sha256(json.dumps(models[model_name][1], sort_keys=True).encode()).hexdigest()
        long["data_hash"] = input_hashes["portfolio_prices"]
        long["config_hash"] = hashlib.sha256(json.dumps({"target_vol": .15, "gross_cap": 1., "ticker_risk_cap": .10,
                                                          "rebalance": "weekly" if models[model_name][1].get("rank_buffer", 0) else "daily",
                                                          "taker_share": 1., "slippage_bps": 5.}, sort_keys=True).encode()).hexdigest()
        long["direction"] = np.sign(long["forecast"]).fillna(0).astype(int)
        long["data_cutoff"] = long["level_0"]
        long["quality_flag"] = np.where(long["forecast"].notna(), "valid", "warmup")
        long = long.rename(columns={"level_0": "timestamp"})
        signal_frames.append(long)
    if signal_frames:
        signal_records = pd.concat(signal_frames, ignore_index=True)
        signal_records.to_parquet(OUT / "daily_signal_records.parquet", index=False)
        signal_records.groupby("model", sort=False).tail(len(portfolio_symbols)).to_csv(OUT / "latest_signal_records.csv", index=False)
    latest_rates = events.sort_values("funding_time").groupby("symbol").tail(1).set_index("symbol")["funding_rate"].reindex(portfolio_symbols)
    expected_rows = []
    for model_name, positions in latest_store.items():
        for symbol, position in positions.items():
            rate = latest_rates.get(symbol)
            expected_rows.append({"model": model_name, "symbol": symbol, "position_weight": float(position),
                                  "reference_funding_rate": float(rate) if pd.notna(rate) else None,
                                  "next_settlement_expected_return": float(-position * rate) if pd.notna(rate) else None,
                                  "next_24h_expected_return": float(-position * rate * 3) if pd.notna(rate) else None,
                                  "estimation_method": "latest_observed_rate_not_realized"})
    pd.DataFrame(expected_rows).to_csv(OUT / "expected_funding_snapshot.csv", index=False)
    # Remove exact duplicates between the headline grid and sensitivity matrix.
    rows = list({row["config_id"]: row for row in rows}.values())
    # Validation-only selection; holdout metrics are disclosed only after selection.
    selected_configs = []
    families = sorted({row["config"].get("family") for row in rows})
    for family in families:
        candidates = [row for row in rows if row["config"].get("family") == family
                      and row["metrics"]["validation"]["net_sharpe"] is not None
                      and row["config"].get("taker_share", 1.) == 1.
                      and row["config"].get("slippage_bps", 5.) == 5.
                      and row["config"].get("activation", "next_open") == "next_open"]
        if candidates:
            selected_configs.append(max(candidates, key=lambda row: row["metrics"]["validation"]["net_sharpe"]))
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    manifest = {"schema_version": 3, "generated_at": generated_at, "input_hashes": input_hashes,
                "fit_universe_assets": fit_prices.shape[1], "portfolio_assets": portfolio_prices.shape[1], "configuration_count": len(rows),
                "selection_period": ["2024-01-01", "2025-12-31"], "holdout_period": ["2026-01-01", str(portfolio_prices.index.max())],
                "universe_limitation": "current_universe_historical_fallback", "prospective_oos": False,
                "selection_rule": "validation_2024_2025_only_then_disclose_2026_holdout",
                "funding_coverage": [str(pd.to_datetime(events.funding_time, utc=True).min()), str(pd.to_datetime(events.funding_time, utc=True).max())]}
    (OUT / "tested_configurations.json").write_text(json.dumps(json_safe(rows), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    (OUT / "selected_configurations.json").write_text(json.dumps(json_safe(selected_configs), indent=2, allow_nan=False), encoding="utf-8")
    deployable = {row["config"]["family"]: {"config_id": row["config_id"], **row["config"]} for row in selected_configs}
    (OUT / "deployable_configurations.json").write_text(json.dumps(json_safe(deployable), indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    flat = []
    for row in rows:
        record = {"config_id": row["config_id"], **row["config"]}
        for split, vals in row["metrics"].items():
            record.update({f"{split}_{key}": value for key, value in vals.items()})
        flat.append(record)
    pd.DataFrame(flat).to_parquet(OUT / "tested_configurations.parquet", index=False)
    pd.DataFrame(daily_store).to_parquet(OUT / "default_daily_returns.parquet")
    pd.DataFrame(funding_store).to_parquet(OUT / "default_daily_funding.parquet")
    pd.DataFrame(latest_store).T.to_csv(OUT / "latest_positions.csv")
    # Canonical append-only SQLite ledger. Every invocation gets a content-addressed run id.
    run_id = hashlib.sha256(json.dumps({"generated_at": generated_at, "inputs": input_hashes}, sort_keys=True).encode()).hexdigest()[:24]
    with sqlite3.connect(OUT / "crypto_momentum_research.sqlite") as connection:
        pd.DataFrame([{"run_id": run_id, "generated_at": generated_at, "manifest_json": json.dumps(json_safe(manifest), sort_keys=True)}]).to_sql(
            "study_runs", connection, if_exists="append", index=False)
        pd.DataFrame([{"run_id": run_id, "config_id": row["config_id"], "config_json": json.dumps(json_safe(row["config"]), sort_keys=True),
                       "metrics_json": json.dumps(json_safe(row["metrics"]), sort_keys=True)} for row in rows]).to_sql(
            "tested_configurations", connection, if_exists="append", index=False)
        daily_long = pd.DataFrame(daily_store).rename_axis("timestamp").stack().rename("net_return").reset_index()
        daily_long.insert(0, "run_id", run_id); daily_long.to_sql("daily_returns", connection, if_exists="append", index=False)
        funding_long = pd.DataFrame(funding_store).rename_axis("timestamp").stack().rename("funding_return").reset_index()
        funding_long.insert(0, "run_id", run_id); funding_long.to_sql("daily_funding", connection, if_exists="append", index=False)
        if signal_frames:
            ledger_signals = signal_records.copy(); ledger_signals.insert(0, "run_id", run_id)
            ledger_signals.to_sql("signal_records", connection, if_exists="append", index=False, chunksize=10000)
        ledger_ticker = pd.DataFrame(ticker_rows); ledger_ticker.insert(0, "run_id", run_id)
        ledger_ticker.to_sql("ticker_attribution", connection, if_exists="append", index=False)
        ledger_groups = pd.DataFrame(group_rows); ledger_groups.insert(0, "run_id", run_id)
        ledger_groups.to_sql("group_attribution", connection, if_exists="append", index=False)
        if actual_path.exists():
            ledger_actual = actual.copy(); ledger_actual.insert(0, "run_id", run_id)
            ledger_actual.to_sql("actual_funding", connection, if_exists="append", index=False)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
