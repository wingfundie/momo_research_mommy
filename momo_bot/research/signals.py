from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .config import EWMAC_RULES, SignalConfig


def causal_volatility(price: pd.Series, lookback: int, *, annualization: int = 365) -> pd.Series:
    """Trailing return volatility with no backward fill or future leakage."""
    numeric = pd.to_numeric(price, errors="coerce").astype(float)
    returns = numeric.pct_change(fill_method=None)
    return returns.rolling(lookback, min_periods=lookback).std() * np.sqrt(annualization)


def ewmac_forecast(
    price: pd.Series,
    fast: int,
    slow: int,
    *,
    volatility_lookback: int,
    scalar: float = 1.0,
    cap: float = 20.0,
) -> pd.Series:
    """Causal, dimensionless EWMAC forecast with preserved warm-up NaNs."""
    if fast <= 0 or slow <= fast:
        raise ValueError("EWMAC requires 0 < fast < slow")
    numeric = pd.to_numeric(price, errors="coerce").astype(float)
    fast_ma = numeric.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ma = numeric.ewm(span=slow, adjust=False, min_periods=slow).mean()
    daily_price_vol = numeric.pct_change(fill_method=None).rolling(
        volatility_lookback, min_periods=volatility_lookback
    ).std() * numeric.shift(1)
    raw = (fast_ma - slow_ma) / daily_price_vol.replace(0.0, np.nan)
    return (raw * float(scalar)).clip(-cap, cap)


def rule_forecasts(
    price: pd.Series,
    config: SignalConfig,
    *,
    scalars: Mapping[tuple[int, int], float] | None = None,
) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for fast, slow in config.rules:
        scalar = 1.0 if scalars is None else float(scalars.get((fast, slow), 1.0))
        values[f"ewmac_{fast}_{slow}"] = ewmac_forecast(
            price,
            fast,
            slow,
            volatility_lookback=config.volatility_lookback,
            scalar=scalar,
            cap=config.forecast_cap,
        )
    return pd.DataFrame(values, index=price.index)


def equal_rule_weights(columns: list[str] | pd.Index) -> pd.Series:
    columns = list(columns)
    if not columns:
        raise ValueError("At least one rule is required")
    return pd.Series(1.0 / len(columns), index=columns, dtype=float)


def constrained_shrunk_weights(
    raw_weights: pd.Series,
    *,
    shrinkage_to_equal: float,
    max_rule_weight: float,
) -> pd.Series:
    """Shrink non-negative weights to equal and project onto a capped simplex."""
    if raw_weights.empty:
        raise ValueError("raw_weights cannot be empty")
    if not 0.0 <= shrinkage_to_equal <= 1.0:
        raise ValueError("shrinkage_to_equal must be in [0, 1]")
    if max_rule_weight * len(raw_weights) < 1.0 - 1e-12:
        raise ValueError("max_rule_weight is infeasible for this rule count")
    clean = raw_weights.astype(float).clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clean = equal_rule_weights(clean.index) if clean.sum() <= 0 else clean / clean.sum()
    equal = equal_rule_weights(clean.index)
    shrunk = (1.0 - shrinkage_to_equal) * clean + shrinkage_to_equal * equal
    return _capped_simplex(shrunk, max_rule_weight)


def _capped_simplex(weights: pd.Series, cap: float) -> pd.Series:
    result = weights.copy().astype(float)
    free = pd.Series(True, index=result.index)
    remaining = 1.0
    while free.any():
        free_values = result[free]
        allocation = free_values / free_values.sum() * remaining
        over = allocation > cap + 1e-12
        if not over.any():
            result.loc[free] = allocation
            break
        capped_names = allocation[over].index
        result.loc[capped_names] = cap
        free.loc[capped_names] = False
        remaining = 1.0 - float(result.loc[~free].sum())
    return result / result.sum()


def combine_forecasts(
    forecasts: pd.DataFrame,
    weights: pd.Series | None = None,
    *,
    diversification_multiplier: float = 1.0,
    cap: float = 20.0,
) -> pd.Series:
    if weights is None:
        weights = equal_rule_weights(forecasts.columns)
    aligned = weights.reindex(forecasts.columns)
    if aligned.isna().any():
        raise ValueError("Missing rule weights")
    valid_weight = forecasts.notna().mul(aligned, axis=1).sum(axis=1)
    weighted = forecasts.mul(aligned, axis=1).sum(axis=1, min_count=1)
    combined = weighted.div(valid_weight.replace(0.0, np.nan)) * float(diversification_multiplier)
    return combined.clip(-cap, cap)


def pooled_rule_scores(forecasts: Mapping[str, pd.DataFrame], prices: pd.DataFrame) -> pd.Series:
    """Estimate non-negative pooled rule scores from causal one-day rule P&L."""
    samples: list[pd.Series] = []
    for symbol, frame in forecasts.items():
        if symbol not in prices:
            continue
        returns = prices[symbol].pct_change(fill_method=None)
        for column in frame:
            pnl = frame[column].shift(1).div(10.0) * returns
            vol = pnl.std()
            score = 0.0 if not np.isfinite(vol) or vol <= 0 else max(float(pnl.mean() / vol), 0.0)
            samples.append(pd.Series({"rule": column, "score": score, "symbol": symbol}))
    if not samples:
        first = next(iter(forecasts.values()), pd.DataFrame(columns=[f"ewmac_{a}_{b}" for a, b in EWMAC_RULES]))
        return equal_rule_weights(first.columns)
    table = pd.DataFrame(samples)
    return table.groupby("rule")["score"].median()


def pooled_forecast_scalars(
    forecasts: Mapping[str, pd.DataFrame], *, target_absolute_forecast: float = 10.0
) -> pd.Series:
    """Robust pooled scalars estimated from all available historical assets."""
    stacked = []
    for symbol, frame in forecasts.items():
        part = frame.abs().copy()
        part["symbol"] = symbol
        stacked.append(part)
    if not stacked:
        return pd.Series(dtype=float)
    sample = pd.concat(stacked)
    medians = sample.drop(columns="symbol").median(axis=0, skipna=True).replace(0.0, np.nan)
    return (float(target_absolute_forecast) / medians).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def forecast_diversification_multiplier(forecasts: pd.DataFrame, weights: pd.Series, cap: float = 2.5) -> float:
    correlation = forecasts.corr().fillna(0.0)
    np.fill_diagonal(correlation.values, 1.0)
    w = weights.reindex(correlation.columns).fillna(0.0).to_numpy(float)
    variance = float(w @ correlation.to_numpy(float) @ w)
    return float(min(1.0 / np.sqrt(variance), cap)) if variance > 0 else 1.0


def pooled_rank_ic_scores(
    component_forecasts: Mapping[str, pd.DataFrame], prices: pd.DataFrame, horizons: tuple[int, ...] = (5, 20)
) -> pd.Series:
    """Score rules by historical cross-sectional rank IC over selected horizons."""
    rule_names = next(iter(component_forecasts.values())).columns
    scores = {}
    for rule in rule_names:
        forecast = pd.DataFrame({symbol: frame[rule] for symbol, frame in component_forecasts.items()})
        horizon_scores = []
        for horizon in horizons:
            future = prices.pct_change(horizon, fill_method=None).shift(-horizon)
            horizon_scores.append(forecast.corrwith(future, axis=1, method="spearman").mean())
        scores[rule] = max(float(np.nanmean(horizon_scores)), 0.0)
    return pd.Series(scores)


def expanding_refit_weights(
    prices: pd.DataFrame,
    component_forecasts: Mapping[str, pd.DataFrame],
    config: SignalConfig,
) -> pd.DataFrame:
    """Quarterly/semiannual/annual expanding refits, activated next candle."""
    index = prices.index.sort_values()
    if index.empty:
        return pd.DataFrame()
    schedule = {"quarterly": "QS", "semiannual": "2QS", "annual": "YS", "frozen": "YS"}
    if config.refit_schedule not in schedule:
        raise ValueError(f"Unsupported refit schedule: {config.refit_schedule}")
    candidate_dates = pd.date_range(index.min(), index.max(), freq=schedule[config.refit_schedule])
    rows: dict[pd.Timestamp, pd.Series] = {}
    for refit_date in candidate_dates:
        train_end = index[index < refit_date]
        if len(train_end) < config.minimum_training_days:
            continue
        cutoff = train_end[-1]
        sliced = {symbol: frame.loc[:cutoff] for symbol, frame in component_forecasts.items()}
        scores = pooled_rule_scores(sliced, prices.loc[:cutoff])
        rows[pd.Timestamp(refit_date)] = constrained_shrunk_weights(
            scores,
            shrinkage_to_equal=config.shrinkage_to_equal,
            max_rule_weight=config.max_rule_weight,
        )
        if config.refit_schedule == "frozen":
            break
    if not rows:
        columns = next(iter(component_forecasts.values())).columns
        return pd.DataFrame([equal_rule_weights(columns)], index=[index.min()])
    weights = pd.DataFrame(rows).T.sort_index()
    # Refit becomes active strictly after the refit timestamp and transitions causally.
    weights = weights.reindex(index).ffill().shift(1)
    equal = equal_rule_weights(weights.columns)
    weights = weights.fillna(equal)
    if config.smoothing_days > 1:
        weights = weights.ewm(span=config.smoothing_days, adjust=False).mean()
        weights = weights.div(weights.sum(axis=1), axis=0)
    return weights


def cross_sectional_percentiles(forecasts: pd.DataFrame, membership: pd.DataFrame | None = None) -> pd.DataFrame:
    eligible = forecasts if membership is None else forecasts.where(membership.reindex_like(forecasts).fillna(False))
    return eligible.rank(axis=1, pct=True, method="average")


def continuous_rank_weights(percentiles: pd.DataFrame) -> pd.DataFrame:
    centered = percentiles - 0.5
    gross = centered.abs().sum(axis=1).replace(0.0, np.nan)
    return centered.div(gross, axis=0).fillna(0.0)


def basket_rank_weights(percentiles: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must be between zero and one half")
    long_mask = percentiles >= 1.0 - fraction
    short_mask = percentiles <= fraction
    long_count = long_mask.sum(axis=1).replace(0, np.nan)
    short_count = short_mask.sum(axis=1).replace(0, np.nan)
    return long_mask.div(long_count, axis=0).fillna(0.0) * 0.5 - short_mask.div(short_count, axis=0).fillna(0.0) * 0.5


def long_only_rank_weights(percentiles: pd.DataFrame, absolute_forecasts: pd.DataFrame, fraction: float) -> pd.DataFrame:
    selected = (percentiles >= 1.0 - fraction) & (absolute_forecasts > 0.0)
    count = selected.sum(axis=1).replace(0, np.nan)
    return selected.div(count, axis=0).fillna(0.0)


def breakout_forecast(price: pd.Series, horizon: int = 64, cap: float = 20.0) -> pd.Series:
    """Causal channel breakout reference with a complete-window warm-up."""
    high = price.rolling(horizon, min_periods=horizon).max()
    low = price.rolling(horizon, min_periods=horizon).min()
    midpoint = (high + low) / 2.0
    spread = (high - low).replace(0.0, np.nan)
    return (40.0 * (price - midpoint) / spread).clip(-cap, cap)


def rank_information_coefficient(
    forecasts: pd.DataFrame, prices: pd.DataFrame, horizons: tuple[int, ...] = (5, 20)
) -> pd.Series:
    """Cross-sectional Spearman IC, using only subsequently observed returns."""
    values = []
    for horizon in horizons:
        future = prices.pct_change(horizon, fill_method=None).shift(-horizon)
        daily = forecasts.corrwith(future, axis=1, method="spearman")
        values.append(daily.rename(f"ic_{horizon}"))
    return pd.concat(values, axis=1).mean(axis=1, skipna=True)


def btc_residualized_forecasts(
    forecasts: pd.DataFrame, returns: pd.DataFrame, benchmark: str = "BTCUSDT", lookback: int = 90
) -> pd.DataFrame:
    betas = rolling_betas(returns, benchmark=benchmark, lookback=lookback)
    benchmark_forecast = forecasts[benchmark] if benchmark in forecasts else pd.Series(0.0, index=forecasts.index)
    return forecasts - betas.mul(benchmark_forecast, axis=0)


def beta_constrained_weights(
    weights: pd.DataFrame, betas: pd.DataFrame, *, gross_cap: float = 1.0
) -> pd.DataFrame:
    """Project each row away from the beta vector, then restore requested gross."""
    result = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    aligned = betas.reindex_like(weights)
    for timestamp in weights.index:
        w = weights.loc[timestamp].fillna(0.0).to_numpy(float)
        beta = aligned.loc[timestamp].fillna(0.0).to_numpy(float)
        denominator = float(beta @ beta)
        projected = w if denominator <= 0 else w - beta * float(beta @ w) / denominator
        gross = float(np.abs(projected).sum())
        if gross > 0:
            projected *= min(float(gross_cap), gross) / gross
        result.loc[timestamp] = projected
    return result


def rolling_betas(returns: pd.DataFrame, benchmark: str = "BTCUSDT", lookback: int = 90) -> pd.DataFrame:
    if benchmark not in returns:
        raise KeyError(f"Missing benchmark {benchmark}")
    benchmark_returns = returns[benchmark]
    variance = benchmark_returns.rolling(lookback, min_periods=lookback).var().replace(0.0, np.nan)
    return pd.DataFrame(
        {column: returns[column].rolling(lookback, min_periods=lookback).cov(benchmark_returns) / variance for column in returns},
        index=returns.index,
    )


def btc_hedged_weights(weights: pd.DataFrame, betas: pd.DataFrame, benchmark: str = "BTCUSDT") -> pd.DataFrame:
    aligned_betas = betas.reindex_like(weights)
    result = weights.copy().fillna(0.0)
    portfolio_beta = (result * aligned_betas).sum(axis=1)
    if benchmark not in result:
        result[benchmark] = 0.0
    benchmark_beta = betas.get(benchmark, pd.Series(1.0, index=weights.index)).replace(0.0, np.nan).fillna(1.0)
    result[benchmark] = result[benchmark] - portfolio_beta / benchmark_beta
    gross = result.abs().sum(axis=1).replace(0.0, np.nan)
    return result.div(gross, axis=0).fillna(0.0)


@dataclass(frozen=True)
class WeightSpecification:
    name: str
    shrinkage: float
    cap: float
    smoothing_days: int


WEIGHT_SPECIFICATIONS = (
    WeightSpecification("shrink_75", 0.75, 0.30, 90),
    WeightSpecification("shrink_80_primary", 0.80, 0.25, 125),
    WeightSpecification("shrink_90", 0.90, 0.25, 125),
)
