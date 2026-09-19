from __future__ import annotations

"""Optimize forecast weights and run backtests for momentum and breakout systems.

This script calibrates or loads forecast scalars, computes optional diversification
multipliers (DM), optimizes per-ticker weights with Optuna, and writes a backtest
bundle used by the Plotly Dash dashboard.

Calibration JSON (optional):
  Use `calibrate_forecasts.py` to generate a JSON payload with scalars and DM.
  Pass it here with `--calibration-json` to reuse those values.

Examples:
  python reoptimize_all.py --mode both --oos-fraction 0
  python reoptimize_all.py --mode breakout --calibration-json data_store/calibration.json
"""

import argparse
import ast
import datetime as dt
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy.stats import norm

import portfolio_strategy as cta
from momo_bot.costs import BacktestCostConfig, CommissionRate, calculate_costs, config_from_settings
from momo_bot.config import settings
from momo_bot.data_validation import validate_price_frame
from momo_bot.exchanges import BinanceFuturesExchange

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


def _periods_per_year(freq: str) -> int:
    """Return the number of periods per year for a data frequency string."""
    return cta._periods_per_year(freq)


def _bars_per_day(freq: str) -> int:
    """Estimate the number of bars per day for an intraday frequency string."""
    if freq.endswith("h"):
        try:
            hours = int(freq[:-1])
        except ValueError:
            hours = 24
        return max(1, int(24 / hours))
    return 1


def _resolve_ewmac_factors(
    base_factors: List[Tuple[int, int]],
    *,
    span_mode: str,
    data_frequency: str,
) -> List[Tuple[int, int]]:
    """Resolve EWMAC factors based on data frequency and span mode.

    Args:
        base_factors: List of (fast, slow) spans defined for daily data.
        span_mode: "native" or "daily-equivalent".
        data_frequency: Frequency string (e.g., "1d", "4h").

    Returns:
        List of (fast, slow) spans appropriate for the data frequency.
    """
    if span_mode == "daily-equivalent" and data_frequency != "1d":
        multiplier = _bars_per_day(data_frequency)
        return [(max(2, f * multiplier), max(4, s * multiplier)) for f, s in base_factors]
    return base_factors


def _calibrate_ewmac_scalars(
    price_frame: pd.DataFrame,
    ewmac_factors: List[Tuple[int, int]],
    *,
    vol_lookback: int,
    fallback_scalars: Dict[Tuple[int, int], float],
    base_factors: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """Calibrate EWMAC scalars so the median absolute forecast is ~10.

    Args:
        price_frame: Price data by ticker (columns).
        ewmac_factors: List of (fast, slow) spans to calibrate.
        vol_lookback: Volatility lookback window for the forecast.
        fallback_scalars: Carver defaults to use when calibration fails.
        base_factors: Base daily factors aligned to fallback scalars.

    Returns:
        Mapping of (fast, slow) span to scalar.
    """
    scalars: Dict[Tuple[int, int], float] = {}
    for factor in ewmac_factors:
        f, s = factor
        abs_means: List[float] = []
        for ticker in price_frame.columns:
            price = price_frame[ticker].dropna()
            if price.empty:
                continue
            raw_forecast = cta.calc_ewma_forecast(price, Lfast=f, Lslow=s, vol_lookback=vol_lookback)
            abs_mean = raw_forecast.abs().replace([np.inf, -np.inf], np.nan).dropna().mean()
            if pd.notnull(abs_mean) and abs_mean > 0:
                abs_means.append(float(abs_mean))
        median_abs = float(np.nanmedian(abs_means)) if abs_means else float("nan")
        scalar = 10.0 / median_abs if pd.notnull(median_abs) and median_abs > 0 else float("nan")
        scalars[factor] = scalar

    for base_factor, actual_factor in zip(base_factors, ewmac_factors):
        if actual_factor not in scalars or not pd.notnull(scalars[actual_factor]):
            fallback = fallback_scalars.get(base_factor, float("nan"))
            scalars[actual_factor] = float(fallback) if pd.notnull(fallback) else float("nan")

    return scalars


def _calibrate_breakout_scalars(
    price_frame: pd.DataFrame,
    breakout_horizons: List[int],
    *,
    fallback_scalars: Dict[int, float],
) -> Dict[int, float]:
    """Calibrate breakout scalars so the median absolute forecast is ~10.

    Args:
        price_frame: Price data by ticker (columns).
        breakout_horizons: Lookback windows for breakout forecasts.
        fallback_scalars: Carver defaults to use when calibration fails.

    Returns:
        Mapping of breakout horizon to scalar.
    """
    scalars: Dict[int, float] = {}
    for horizon in breakout_horizons:
        abs_means: List[float] = []
        for ticker in price_frame.columns:
            price = price_frame[ticker].dropna()
            if price.empty or len(price) < horizon:
                continue
            raw_forecast = cta.calc_breakout_forecast(price, horizon=horizon)
            abs_mean = raw_forecast.abs().replace([np.inf, -np.inf], np.nan).dropna().mean()
            if pd.notnull(abs_mean) and abs_mean > 0:
                abs_means.append(float(abs_mean))
        median_abs = float(np.nanmedian(abs_means)) if abs_means else float("nan")
        scalar = 10.0 / median_abs if pd.notnull(median_abs) and median_abs > 0 else float("nan")
        scalars[horizon] = scalar

    for horizon in breakout_horizons:
        if horizon not in scalars or not pd.notnull(scalars[horizon]):
            fallback = fallback_scalars.get(horizon, float("nan"))
            scalars[horizon] = float(fallback) if pd.notnull(fallback) else float("nan")

    return scalars


def _compute_breakout_corr_matrix(
    price_frame: pd.DataFrame,
    breakout_horizons: List[int],
    *,
    scalars: Dict[int, float],
) -> pd.DataFrame:
    """Compute a pooled correlation matrix for breakout forecasts.

    The matrix is averaged across tickers and negative correlations are floored at
    zero to avoid unstable multipliers.
    """
    n = len(breakout_horizons)
    corr_sum = np.zeros((n, n), dtype=float)
    count = 0

    for ticker in price_frame.columns:
        price = price_frame[ticker].dropna()
        if price.empty or len(price) < max(breakout_horizons):
            continue
        signal_df = pd.DataFrame(index=price.index)
        for horizon in breakout_horizons:
            scalar = scalars.get(horizon)
            if scalar is None or not pd.notnull(scalar):
                continue
            raw_forecast = cta.calc_breakout_forecast(price, horizon=horizon)
            signal = cta.get_signal(price, raw_forecast, None, scalar)
            signal_df[str(horizon)] = signal
        signal_df = signal_df.dropna()
        if signal_df.shape[0] < 20 or signal_df.shape[1] != n:
            continue
        corr = signal_df.corr()
        if corr.isnull().all().all():
            continue
        corr_sum += corr.values
        count += 1

    if count == 0:
        corr = np.eye(n)
    else:
        corr = corr_sum / count

    corr = np.where(np.isfinite(corr), corr, 0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=breakout_horizons, columns=breakout_horizons)


def _compute_ewmac_corr_matrix(
    price_frame: pd.DataFrame,
    ewmac_factors: List[Tuple[int, int]],
    *,
    scalars: Dict[Tuple[int, int], float],
    vol_lookback: int,
) -> pd.DataFrame:
    """Compute a pooled correlation matrix for EWMAC forecasts.

    The matrix is averaged across tickers and negative correlations are floored at
    zero to avoid unstable multipliers.
    """
    n = len(ewmac_factors)
    corr_sum = np.zeros((n, n), dtype=float)
    count = 0

    for ticker in price_frame.columns:
        price = price_frame[ticker].dropna()
        if price.empty or len(price) < max(s for _, s in ewmac_factors):
            continue
        signal_df = pd.DataFrame(index=price.index)
        for factor in ewmac_factors:
            scalar = scalars.get(factor)
            if scalar is None or not pd.notnull(scalar):
                continue
            raw_forecast = cta.calc_ewma_forecast(
                price, Lfast=factor[0], Lslow=factor[1], vol_lookback=vol_lookback
            )
            signal = cta.get_signal(price, raw_forecast, None, scalar)
            signal_df[str(factor)] = signal
        signal_df = signal_df.dropna()
        if signal_df.shape[0] < 20 or signal_df.shape[1] != n:
            continue
        corr = signal_df.corr()
        if corr.isnull().all().all():
            continue
        corr_sum += corr.values
        count += 1

    if count == 0:
        corr = np.eye(n)
    else:
        corr = corr_sum / count

    corr = np.where(np.isfinite(corr), corr, 0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=[str(f) for f in ewmac_factors], columns=[str(f) for f in ewmac_factors])


def _forecast_diversification_multiplier(
    weights: np.ndarray,
    corr_matrix: pd.DataFrame,
    *,
    dm_cap: float | None = None,
) -> float:
    """Calculate the forecast diversification multiplier (DM).

    Args:
        weights: Forecast weights (will be normalized).
        corr_matrix: Correlation matrix of forecast variations.
        dm_cap: Optional cap for the multiplier.

    Returns:
        Diversification multiplier (>= 1 when variance < 1).
    """
    if corr_matrix.empty:
        return 1.0
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(corr_matrix), dtype=float)
    w = w / w.sum()
    try:
        variance = float(w @ corr_matrix.values @ w)
    except Exception:
        return 1.0
    if not np.isfinite(variance) or variance <= 0:
        return 1.0
    dm = 1.0 / np.sqrt(variance)
    if dm_cap is not None and np.isfinite(dm_cap):
        dm = min(dm, dm_cap)
    return float(dm)


def _resolve_path(path: Path) -> Path:
    """Resolve a potentially relative path using the configured base directory."""
    if path.is_absolute():
        return path
    return (settings.base_dir / path).resolve()


def _load_calibration(path: Path) -> Dict[str, Any]:
    """Load a calibration JSON payload from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration JSON must be a dictionary.")
    return payload


def _parse_ewmac_factors(raw: Iterable[Any]) -> List[Tuple[int, int]]:
    """Parse a list of EWMAC factors from JSON or other iterables."""
    factors: List[Tuple[int, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            f, s = item
        else:
            try:
                f, s = ast.literal_eval(str(item))
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid EWMAC factor: {item}") from exc
        factors.append((int(f), int(s)))
    return factors


def _parse_ewmac_scalars(raw: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
    """Parse scalar keys like '(2, 8)' into tuple keys for EWMAC scalars."""
    scalars: Dict[Tuple[int, int], float] = {}
    for key, value in raw.items():
        if isinstance(key, (list, tuple)) and len(key) == 2:
            f, s = key
        else:
            try:
                f, s = ast.literal_eval(str(key))
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid EWMAC scalar key: {key}") from exc
        scalars[(int(f), int(s))] = float(value)
    return scalars


def _parse_breakout_horizons(raw: Iterable[Any]) -> List[int]:
    """Parse breakout horizons from an iterable."""
    return [int(h) for h in raw]


def _parse_breakout_scalars(raw: Dict[str, Any]) -> Dict[int, float]:
    """Parse breakout scalar keys from JSON into integer horizons."""
    scalars: Dict[int, float] = {}
    for key, value in raw.items():
        scalars[int(key)] = float(value)
    return scalars


@dataclass(frozen=True)
class SplitResult:
    train_index: pd.Index
    oos_index: pd.Index
    first_oos_ts: pd.Timestamp | None
    oos_n: int


def split_train_oos(
    index: pd.Index,
    *,
    oos_fraction: float | None,
    oos_periods: int | None,
) -> SplitResult:
    """Return disjoint train/OOS indices."""
    if index.empty:
        return SplitResult(index, index[:0], None, 0)
    n = len(index)
    if oos_periods is not None and oos_periods > 0:
        oos_n = min(oos_periods, n - 1)
    elif oos_fraction is not None and oos_fraction > 0:
        oos_n = max(1, int(n * oos_fraction))
        oos_n = min(oos_n, n - 1)
    else:
        return SplitResult(index, index[:0], None, 0)
    first_oos_pos = n - oos_n
    train_index = index[:first_oos_pos]
    oos_index = index[first_oos_pos:]
    first_oos_ts = oos_index[0] if len(oos_index) else None
    return SplitResult(train_index, oos_index, first_oos_ts, oos_n)


def _split_train_test_index(
    index: pd.Index,
    *,
    oos_fraction: float | None,
    oos_periods: int | None,
) -> Tuple[pd.Timestamp | None, int]:
    """Backward-compatible wrapper returning first OOS timestamp and OOS length."""
    split = split_train_oos(index, oos_fraction=oos_fraction, oos_periods=oos_periods)
    return split.first_oos_ts, split.oos_n


def _deflated_sharpe(sr: float, n_trials: int, n_obs: int) -> float:
    """Compute a deflated Sharpe adjustment for multiple testing."""
    if not np.isfinite(sr) or n_trials <= 1 or n_obs <= 1:
        return float("nan")
    sr_std = np.sqrt((1.0 + 0.5 * sr ** 2) / max(1, n_obs - 1))
    if sr_std <= 0 or not np.isfinite(sr_std):
        return float("nan")
    threshold = norm.ppf(1.0 - (1.0 / n_trials))
    if not np.isfinite(threshold):
        return float("nan")
    return float(sr - threshold * sr_std)


def _compute_metrics(returns: pd.Series, freq: str, *, n_trials: int | None = None) -> Dict[str, float]:
    """Compute performance metrics for a return series."""
    returns = returns.dropna()
    if returns.empty:
        return {
            "sharpe": float("nan"),
            "cagr": float("nan"),
            "max_drawdown": float("nan"),
            "total_return": float("nan"),
            "skew": float("nan"),
            "upper_tail": float("nan"),
            "lower_tail": float("nan"),
            "t_stat": float("nan"),
            "deflated_sharpe": float("nan"),
        }
    periods_per_year = _periods_per_year(freq)
    sharpe = cta._annualized_sharpe(returns, freq)
    equity = (1.0 + returns).cumprod()
    total_return = equity.iloc[-1] - 1.0
    n_periods = len(returns)
    cagr = (1.0 + total_return) ** (periods_per_year / max(1, n_periods)) - 1.0
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = drawdown.min()
    std = returns.std()
    t_stat = returns.mean() / (std / np.sqrt(n_periods)) if std and not np.isnan(std) else np.nan
    upper_tail = returns.quantile(0.95)
    lower_tail = returns.quantile(0.05)
    skew = returns.skew()
    deflated_sharpe = float("nan")
    if n_trials is not None and pd.notnull(sharpe):
        deflated_sharpe = _deflated_sharpe(float(sharpe), int(n_trials), len(returns))
    return {
        "sharpe": float(sharpe) if pd.notnull(sharpe) else float("nan"),
        "cagr": float(cagr),
        "max_drawdown": float(max_drawdown),
        "total_return": float(total_return),
        "skew": float(skew) if pd.notnull(skew) else float("nan"),
        "upper_tail": float(upper_tail),
        "lower_tail": float(lower_tail),
        "t_stat": float(t_stat) if pd.notnull(t_stat) else float("nan"),
        "deflated_sharpe": float(deflated_sharpe) if pd.notnull(deflated_sharpe) else float("nan"),
    }


def _build_ewmac_signal(
    price: pd.Series,
    weights: List[float],
    ewmac_factors: List[Tuple[int, int]],
    *,
    scalars: Dict[Tuple[int, int], float],
    diversification_multiplier: float,
    vol_lookback: int,
    cap_final_forecast: bool = True,
) -> pd.Series:
    """Build a combined EWMAC forecast series with scaling and DM applied."""
    if len(weights) != len(ewmac_factors):
        raise ValueError("Weights length does not match ewmac_factors length.")
    signal_df = pd.DataFrame(index=price.index)
    for (i, j), w in zip(ewmac_factors, weights):
        scalar = scalars.get((i, j))
        if scalar is None:
            raise ValueError(f"Missing scalar for EWMAC pair {(i, j)}.")
        raw_forecast = cta.calc_ewma_forecast(price, Lfast=i, Lslow=j, vol_lookback=vol_lookback)
        signal = cta.get_signal(price, raw_forecast, None, scalar)
        signal_df[f"({i},{j})"] = signal
    weighted_signal = np.dot(signal_df.fillna(0.0), weights)
    weighted_signal = pd.Series(weighted_signal, index=signal_df.index) * diversification_multiplier
    return weighted_signal.clip(-20, 20) if cap_final_forecast else weighted_signal


def _build_breakout_signal(
    price: pd.Series,
    weights: List[float],
    breakout_horizons: List[int],
    *,
    scalars: Dict[int, float],
    diversification_multiplier: float,
    cap_final_forecast: bool = True,
) -> pd.Series:
    """Build a combined breakout forecast series with scaling and DM applied."""
    if len(weights) != len(breakout_horizons):
        raise ValueError("Weights length does not match breakout_horizons length.")
    signal_df = pd.DataFrame(index=price.index)
    for horizon, w in zip(breakout_horizons, weights):
        scalar = scalars.get(horizon)
        if scalar is None:
            raise ValueError(f"Missing scalar for breakout horizon {horizon}.")
        raw_forecast = cta.calc_breakout_forecast(price, horizon=horizon)
        signal = cta.get_signal(price, raw_forecast, None, scalar)
        signal_df[f"Breakout_{horizon}"] = signal
    weighted_signal = np.dot(signal_df.fillna(0.0), weights)
    weighted_signal = pd.Series(weighted_signal, index=signal_df.index) * diversification_multiplier
    return weighted_signal.clip(-20, 20) if cap_final_forecast else weighted_signal


def _run_backtest(
    price: pd.Series,
    signal: pd.Series,
    *,
    ticker: str | None = None,
    portfolio_value: float,
    target_vol_annual: float,
    data_frequency: str,
    vol_lookback: int,
    max_leverage: float | None,
    cost_config: BacktestCostConfig | None = None,
    exchange: BinanceFuturesExchange | None = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Run a single-ticker backtest and return gross/net returns, equity, positions, costs."""
    positions, positions_usd = cta._position_size_from_forecast(
        price,
        signal,
        portfolio_value=portfolio_value,
        target_vol_annual=target_vol_annual,
        data_frequency=data_frequency,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
    )
    pnl = cta.get_pnl(price, trades=None, positions=positions)
    gross_pnl = pnl[3].fillna(0.0)
    gross_returns = gross_pnl / portfolio_value

    costs_df = pd.DataFrame(index=price.index)
    costs_df["fees"] = 0.0
    costs_df["slippage"] = 0.0
    costs_df["funding"] = 0.0
    costs_df["total_cost"] = 0.0
    costs_df["turnover_notional"] = 0.0

    if cost_config is not None and cost_config.enabled:
        if exchange is not None and ticker is not None:
            commission = exchange.get_commission_rate(ticker)
            funding_rates = exchange.get_funding_rates(ticker, price.index.min(), price.index.max())
        else:
            symbol = ticker or "UNKNOWN"
            commission = CommissionRate(
                symbol=symbol,
                maker=cost_config.maker_fee,
                taker=cost_config.taker_fee,
                source=cost_config.source,
            )
            funding_rates = pd.Series(dtype=float)

        costs = calculate_costs(
            positions_usd,
            commission=commission,
            config=cost_config,
            funding_rates=funding_rates,
        )
        costs_df = pd.DataFrame(
            {
                "fees": costs.fees,
                "slippage": costs.slippage,
                "funding": costs.funding,
                "total_cost": costs.total,
                "turnover_notional": costs.turnover_notional,
            },
            index=price.index,
        )

    net_pnl = gross_pnl - costs_df["total_cost"].fillna(0.0)
    net_returns = net_pnl / portfolio_value
    equity = (1.0 + net_returns.fillna(0.0)).cumprod()
    return gross_returns, net_returns, equity, positions_usd, costs_df


def _optimize_momentum_weights(
    price_frame: pd.DataFrame,
    tickers: Iterable[str],
    *,
    ewmac_factors: List[Tuple[int, int]],
    forecast_scalars: Dict[Tuple[int, int], float],
    vol_lookback: int,
    n_trials: int,
    data_frequency: str,
    portfolio_value: float,
    target_vol_annual: float,
    max_leverage: float | None,
    diversification_multiplier: float,
    cap_final_forecast: bool,
) -> Dict[str, Dict[str, Any]]:
    """Optimize EWMAC weights per ticker using Optuna.

    Returns:
        Mapping keyed by ticker, with weights, Sharpe estimate, and status.
    """
    results: Dict[str, Dict[str, Any]] = {}
    scalars = [forecast_scalars.get(pair) for pair in ewmac_factors]
    if any(s is None or not pd.notnull(s) for s in scalars):
        missing = [ewmac_factors[i] for i, s in enumerate(scalars) if s is None or not pd.notnull(s)]
        raise ValueError(f"Missing EWMAC scalars for: {missing}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for ticker in tqdm(list(tickers), desc="Momentum opt"):
        price = price_frame[ticker].dropna()
        min_required = max(j for _, j in ewmac_factors) + vol_lookback
        if len(price) < min_required:
            results[ticker] = {"weights": [np.nan] * len(ewmac_factors), "sr": np.nan, "status": "skipped_insufficient_data"}
            continue

        def _objective(trial: optuna.Trial) -> float:
            return cta._objective_optimize_weights(
                trial,
                price,
                ewmac_factors,
                scalars,
                vol_lookback,
                portfolio_value,
                diversification_multiplier,
                data_frequency,
                target_vol_annual,
                max_leverage,
                cap_final_forecast=cap_final_forecast,
            )

        study = optuna.create_study(study_name=f"ewmac_weights_{ticker}", direction="minimize")
        study.optimize(_objective, n_trials=n_trials)

        if study.best_trial and study.best_trial.value != np.inf:
            raw_weights = [study.best_trial.params[f"w{i+1}"] for i in range(len(ewmac_factors))]
            total_raw = sum(raw_weights)
            weights = [w / total_raw for w in raw_weights] if total_raw > 1e-8 else [1.0 / len(ewmac_factors)] * len(ewmac_factors)
            results[ticker] = {"weights": weights, "sr": -study.best_trial.value, "status": "success"}
        else:
            results[ticker] = {"weights": [np.nan] * len(ewmac_factors), "sr": np.nan, "status": "failed_no_valid_solution"}

    return results


def _optimize_breakout_weights_single(
    price: pd.Series,
    *,
    breakout_horizons: List[int],
    forecast_scalars: Dict[int, float],
    vol_lookback: int,
    n_trials: int,
    data_frequency: str,
    portfolio_value: float,
    target_vol_annual: float,
    max_leverage: float | None,
    diversification_multiplier: float,
    cap_final_forecast: bool,
    opt_mode: str,
    top_percentile: float,
) -> Dict[str, Any]:
    """Optimize breakout weights for a single price series."""
    min_required = max(breakout_horizons) + vol_lookback
    price = price.dropna()
    if price.empty or len(price) < min_required:
        return {"weights": [np.nan] * len(breakout_horizons), "sr": np.nan, "status": "skipped_insufficient_data", "n_valid": 0}

    scalars = [forecast_scalars.get(h) for h in breakout_horizons]
    if any(s is None or not pd.notnull(s) for s in scalars):
        missing = [breakout_horizons[i] for i, s in enumerate(scalars) if s is None or not pd.notnull(s)]
        raise ValueError(f"Missing breakout scalars for horizons: {missing}")

    def _objective(trial: optuna.Trial) -> float:
        return cta._objective_optimize_breakout_weights(
            trial,
            price,
            breakout_horizons,
            scalars,
            vol_lookback,
            portfolio_value,
            diversification_multiplier,
            data_frequency,
            target_vol_annual,
            max_leverage,
            cap_final_forecast=cap_final_forecast,
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials)

    valid_trials = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    if not valid_trials:
        return {"weights": [np.nan] * len(breakout_horizons), "sr": np.nan, "status": "failed_no_valid_solution", "n_valid": 0}

    if opt_mode == "ensemble":
        pct = min(max(top_percentile, 0.0), 1.0)
        valid_trials.sort(key=lambda t: t.value)
        n_select = max(1, int(len(valid_trials) * pct))
        top_trials = valid_trials[:n_select]
        avg_weights = np.zeros(len(breakout_horizons), dtype=float)
        for trial in top_trials:
            weights_vec = np.array([trial.params[f"w{i+1}"] for i in range(len(breakout_horizons))], dtype=float)
            total = weights_vec.sum()
            if total <= 0:
                weights_vec = np.full(len(breakout_horizons), 1.0 / len(breakout_horizons))
            else:
                weights_vec = weights_vec / total
            avg_weights += weights_vec
        final_weights = avg_weights / len(top_trials)
        final_weights = final_weights / final_weights.sum()
        avg_sr = -float(np.mean([t.value for t in top_trials]))
        return {"weights": final_weights.tolist(), "sr": avg_sr, "status": "success", "n_valid": len(valid_trials)}

    best_trial = study.best_trial
    if best_trial is None or best_trial.value is None or not np.isfinite(best_trial.value):
        return {"weights": [np.nan] * len(breakout_horizons), "sr": np.nan, "status": "failed_no_valid_solution", "n_valid": len(valid_trials)}
    raw_weights = [best_trial.params[f"w{i+1}"] for i in range(len(breakout_horizons))]
    total_raw = sum(raw_weights)
    weights = [w / total_raw for w in raw_weights] if total_raw > 1e-8 else [1.0 / len(breakout_horizons)] * len(breakout_horizons)
    return {"weights": weights, "sr": -best_trial.value, "status": "success", "n_valid": len(valid_trials)}


def _optimize_breakout_weights(
    price_frame: pd.DataFrame,
    tickers: Iterable[str],
    *,
    breakout_horizons: List[int],
    vol_lookback: int,
    n_trials: int,
    data_frequency: str,
    portfolio_value: float,
    target_vol_annual: float,
    max_leverage: float | None,
    diversification_multiplier: float,
    forecast_scalars: Dict[int, float],
    cap_final_forecast: bool,
    opt_mode: str,
    top_percentile: float,
) -> Dict[str, Dict[str, Any]]:
    """Optimize breakout weights per ticker using Optuna."""
    results: Dict[str, Dict[str, Any]] = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for ticker in tqdm(list(tickers), desc="Breakout opt"):
        price = price_frame[ticker].dropna()
        result = _optimize_breakout_weights_single(
            price,
            breakout_horizons=breakout_horizons,
            forecast_scalars=forecast_scalars,
            vol_lookback=vol_lookback,
            n_trials=n_trials,
            data_frequency=data_frequency,
            portfolio_value=portfolio_value,
            target_vol_annual=target_vol_annual,
            max_leverage=max_leverage,
            diversification_multiplier=diversification_multiplier,
            cap_final_forecast=cap_final_forecast,
            opt_mode=opt_mode,
            top_percentile=top_percentile,
        )
        results[ticker] = result
    return results


def _run_walkforward_breakout(
    price: pd.Series,
    *,
    ticker: str | None = None,
    breakout_horizons: List[int],
    forecast_scalars: Dict[int, float],
    vol_lookback: int,
    n_trials: int,
    data_frequency: str,
    portfolio_value: float,
    target_vol_annual: float,
    max_leverage: float | None,
    diversification_multiplier: float,
    cap_final_forecast: bool,
    opt_mode: str,
    top_percentile: float,
    train_window: int,
    test_window: int,
    step_window: int,
    train_mode: str,
    cost_config: BacktestCostConfig | None = None,
) -> Dict[str, Any]:
    """Run a walk-forward optimization for a single ticker."""
    price = price.dropna()
    min_required = max(breakout_horizons) + vol_lookback
    if price.empty or len(price) < train_window + test_window or len(price) < min_required:
        return {"status": "skipped_insufficient_data", "n_windows": 0}

    step = max(1, step_window)
    train_end = train_window
    oos_series_chunks: List[pd.DataFrame] = []
    weight_rows: List[pd.Series] = []

    while train_end + test_window <= len(price):
        if train_mode == "rolling":
            train_start = max(0, train_end - train_window)
        else:
            train_start = 0
        train_slice = price.iloc[train_start:train_end]
        if len(train_slice) < min_required:
            train_end += step
            continue

        opt_result = _optimize_breakout_weights_single(
            train_slice,
            breakout_horizons=breakout_horizons,
            forecast_scalars=forecast_scalars,
            vol_lookback=vol_lookback,
            n_trials=n_trials,
            data_frequency=data_frequency,
            portfolio_value=portfolio_value,
            target_vol_annual=target_vol_annual,
            max_leverage=max_leverage,
            diversification_multiplier=diversification_multiplier,
            cap_final_forecast=cap_final_forecast,
            opt_mode=opt_mode,
            top_percentile=top_percentile,
        )
        if opt_result.get("status") != "success":
            train_end += step
            continue

        weights = opt_result["weights"]
        test_end = train_end + test_window
        combined = price.iloc[:test_end]
        signal = _build_breakout_signal(
            combined,
            weights,
            breakout_horizons,
            scalars=forecast_scalars,
            diversification_multiplier=diversification_multiplier,
            cap_final_forecast=cap_final_forecast,
        )
        gross_full, returns_full, _equity_full, positions_usd_full, costs_full = _run_backtest(
            combined,
            signal,
            ticker=ticker,
            portfolio_value=portfolio_value,
            target_vol_annual=target_vol_annual,
            data_frequency=data_frequency,
            vol_lookback=vol_lookback,
            max_leverage=max_leverage,
            cost_config=cost_config,
        )
        returns_test = returns_full.iloc[train_end:test_end]
        if returns_test.empty:
            train_end += step
            continue

        test_index = returns_test.index
        test_series = pd.DataFrame(
            {
                "price": combined.reindex(test_index),
                "signal": signal.reindex(test_index),
                "returns": returns_test,
                "gross_returns": gross_full.reindex(test_index),
                "net_returns": returns_test,
                "positions_usd": positions_usd_full.reindex(test_index),
            },
            index=test_index,
        )
        for col in costs_full.columns:
            test_series[col] = costs_full[col].reindex(test_index)
        oos_series_chunks.append(test_series)
        weight_rows.append(pd.Series(weights, index=[str(h) for h in breakout_horizons], name=combined.index[train_end]))
        train_end += step

    if not oos_series_chunks:
        return {"status": "failed_no_valid_windows", "n_windows": 0}

    oos_series = pd.concat(oos_series_chunks).sort_index()
    oos_series = oos_series[~oos_series.index.duplicated(keep="last")]
    oos_returns = oos_series["returns"]
    oos_equity = (1.0 + oos_returns.fillna(0.0)).cumprod()
    oos_series["equity"] = oos_equity
    weights_df = pd.DataFrame(weight_rows)
    return {
        "status": "success",
        "series": oos_series,
        "weights": weights_df,
        "n_windows": int(len(weights_df)),
        "first_oos_ts": oos_series.index.min(),
        "last_oos_ts": oos_series.index.max(),
    }


def _run_walkforward_breakout_worker(ticker: str, price: pd.Series, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Process-pool entrypoint for one ticker's breakout WFO."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        result = _run_walkforward_breakout(price, ticker=ticker, **kwargs)
    except Exception as exc:
        result = {
            "status": f"error_{type(exc).__name__}",
            "error": str(exc),
            "n_windows": 0,
        }
    return ticker, result


def _empty_breakout_series(price: pd.Series) -> pd.DataFrame:
    """Create a dashboard-compatible static series for WFO-only/skipped-static tickers."""
    price = price.dropna()
    series = pd.DataFrame(index=price.index)
    series["price"] = price
    series["signal"] = 0.0
    series["returns"] = 0.0
    series["gross_returns"] = 0.0
    series["net_returns"] = 0.0
    series["equity"] = 1.0
    series["positions_usd"] = 0.0
    series["fees"] = 0.0
    series["slippage"] = 0.0
    series["funding"] = 0.0
    series["total_cost"] = 0.0
    series["turnover_notional"] = 0.0
    return series


def _ensure_breakout_payload(
    breakout_results: Dict[str, Any],
    ticker: str,
    *,
    price: pd.Series,
    params: Dict[str, Any],
    dm_value: float,
) -> Dict[str, Any]:
    """Ensure every WFO ticker has a bundle payload, even if static optimization skipped it."""
    payload = breakout_results.get(ticker)
    if payload is not None:
        return payload

    params_with_dm = dict(params) if params else {"status": "missing_static_params"}
    params_with_dm["dm"] = dm_value
    payload = {
        "metrics": None,
        "metrics_gross": None,
        "metrics_train": None,
        "metrics_oos": None,
        "params": params_with_dm,
        "series": _empty_breakout_series(price),
    }
    breakout_results[ticker] = payload
    return payload


def _normalise_walkforward_payload(wf_result: Dict[str, Any], data_frequency: str) -> Dict[str, Any]:
    """Convert a raw WFO result into the bundle shape."""
    status = wf_result.get("status", "unknown")
    payload: Dict[str, Any] = {
        "status": status,
        "n_windows": int(wf_result.get("n_windows", 0) or 0),
    }
    if wf_result.get("error"):
        payload["error"] = wf_result["error"]
    if status == "success":
        series = wf_result["series"]
        payload.update(
            {
                "metrics_oos": _compute_metrics(series["returns"], data_frequency),
                "series": series,
                "weights": wf_result["weights"],
                "first_oos_ts": wf_result.get("first_oos_ts"),
                "last_oos_ts": wf_result.get("last_oos_ts"),
            }
        )
    return payload


def _metric(metrics: Dict[str, Any] | None, key: str) -> float:
    if not metrics:
        return float("nan")
    value = metrics.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_walkforward_summary_csv(
    results: Dict[str, Any],
    breakout_params: Dict[str, Any],
    *,
    output_dir: Path,
    ts_tag: str,
) -> Path:
    """Write a static-vs-WFO comparison CSV next to the bundle."""
    rows: List[Dict[str, Any]] = []
    for ticker, payload in sorted(results.items()):
        static_metrics = payload.get("metrics_oos")
        wf_payload = payload.get("walkforward", {})
        wf_metrics = wf_payload.get("metrics_oos")
        wf_series = wf_payload.get("series")
        first_wfo = wf_payload.get("first_oos_ts")
        last_wfo = wf_payload.get("last_oos_ts")
        if isinstance(wf_series, pd.DataFrame) and not wf_series.empty:
            first_wfo = first_wfo or wf_series.index.min()
            last_wfo = last_wfo or wf_series.index.max()

        rows.append(
            {
                "ticker": ticker,
                "static_status": (breakout_params.get(ticker) or {}).get("status", "missing"),
                "wfo_status": wf_payload.get("status", "missing"),
                "static_oos_sharpe": _metric(static_metrics, "sharpe"),
                "wfo_oos_sharpe": _metric(wf_metrics, "sharpe"),
                "static_oos_return": _metric(static_metrics, "total_return"),
                "wfo_oos_return": _metric(wf_metrics, "total_return"),
                "static_mdd": _metric(static_metrics, "max_drawdown"),
                "wfo_mdd": _metric(wf_metrics, "max_drawdown"),
                "n_wfo_windows": int(wf_payload.get("n_windows", 0) or 0),
                "first_wfo_oos_date": first_wfo,
                "last_wfo_oos_date": last_wfo,
            }
        )

    summary_path = output_dir / f"walkforward_summary_{ts_tag}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    """CLI entrypoint for optimization and backtest bundle generation."""
    parser = argparse.ArgumentParser(description="Re-optimize momentum/breakout weights and run backtests.")
    parser.add_argument("--mode", choices=["momentum", "breakout", "both"], default="both")
    parser.add_argument("--n-trials-momentum", type=int, default=200)
    parser.add_argument("--n-trials-breakout", type=int, default=200)
    parser.add_argument(
        "--breakout-params-input",
        type=str,
        default="",
        help="Optional existing breakout params pickle to reuse instead of running static optimization.",
    )
    parser.add_argument("--oos-fraction", type=float, default=0.2)
    parser.add_argument("--oos-periods", type=int, default=0)
    parser.add_argument("--vol-lookback-momentum", type=int, default=360)
    parser.add_argument("--vol-lookback-breakout", type=int, default=180)
    parser.add_argument(
        "--ewmac-span-mode",
        choices=["native", "daily-equivalent"],
        default="native",
        help="Use native spans or daily-equivalent spans for intraday data.",
    )
    parser.add_argument(
        "--ewmac-scalar-mode",
        choices=["carver", "calibrated"],
        default="carver",
        help="Use Carver scalars or calibrate scalars from the dataset.",
    )
    parser.add_argument(
        "--ewmac-dm-mode",
        choices=["fixed", "correlation"],
        default="fixed",
        help="Use a fixed EWMAC diversification multiplier or compute from forecast correlations.",
    )
    parser.add_argument(
        "--ewmac-dm-weighting",
        choices=["equal", "optuna"],
        default="equal",
        help="When using correlation DM, use equal weights or optimized weights to compute the multiplier.",
    )
    parser.add_argument("--ewmac-dm-fixed", type=float, default=1.12)
    parser.add_argument("--ewmac-dm-cap", type=float, default=2.5)
    parser.add_argument("--ewmac-final-cap", action="store_true")
    parser.add_argument(
        "--breakout-scalar-mode",
        choices=["carver", "calibrated"],
        default="carver",
        help="Use Carver breakout scalars or calibrate scalars from the dataset.",
    )
    parser.add_argument(
        "--breakout-dm-mode",
        choices=["fixed", "correlation"],
        default="fixed",
        help="Use a fixed breakout diversification multiplier or compute from forecast correlations.",
    )
    parser.add_argument(
        "--breakout-dm-weighting",
        choices=["equal", "optuna"],
        default="equal",
        help="When using correlation DM, use equal weights or optimized weights to compute the multiplier.",
    )
    parser.add_argument("--breakout-dm-fixed", type=float, default=1.24)
    parser.add_argument("--breakout-dm-cap", type=float, default=2.5)
    parser.add_argument("--breakout-final-cap", action="store_true")
    parser.add_argument(
        "--breakout-opt-mode",
        choices=["best", "ensemble"],
        default="best",
        help="Choose single best trial or an ensemble average for breakout weights.",
    )
    parser.add_argument("--breakout-ensemble-percentile", type=float, default=0.2)
    parser.add_argument("--breakout-wfo", action="store_true", help="Enable walk-forward optimization for breakout.")
    parser.add_argument("--breakout-wfo-train-window", type=int, default=365)
    parser.add_argument("--breakout-wfo-test-window", type=int, default=90)
    parser.add_argument("--breakout-wfo-step-window", type=int, default=0)
    parser.add_argument(
        "--breakout-wfo-train-mode",
        choices=["expanding", "rolling"],
        default="expanding",
    )
    parser.add_argument("--breakout-wfo-n-trials", type=int, default=0)
    parser.add_argument("--breakout-wfo-workers", type=int, default=1)
    parser.add_argument(
        "--breakout-wfo-checkpoint-path",
        type=str,
        default="",
        help="Optional pickle path for resumable per-ticker breakout WFO results.",
    )
    parser.add_argument("--breakout-wfo-checkpoint-every", type=int, default=1)
    parser.add_argument("--breakout-benchmark-ticker", type=str, default="BTCUSDT")
    parser.add_argument("--calibration-json", type=str, default="")
    parser.add_argument(
        "--calibration-use",
        choices=["scalars", "dm", "both"],
        default="both",
        help="When a calibration JSON is supplied, choose which parts to use.",
    )
    parser.add_argument("--output-dir", type=str, default=str(settings.data_dir))
    parser.add_argument("--costs", choices=["off", "default", "binance"], default=settings.cost_source)
    parser.add_argument("--maker-fee", type=float, default=settings.futures_maker_fee)
    parser.add_argument("--taker-fee", type=float, default=settings.futures_taker_fee)
    parser.add_argument("--taker-share", type=float, default=settings.taker_share)
    parser.add_argument("--slippage-bps", type=float, default=settings.slippage_bps)
    parser.add_argument("--funding-mode", choices=["zero", "historical"], default=settings.funding_mode)
    args = parser.parse_args()

    calibration_payload: Dict[str, Any] | None = None
    calibration_path: Path | None = None
    if args.calibration_json:
        calibration_path = _resolve_path(Path(args.calibration_json))
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration JSON not found: {calibration_path}")
        calibration_payload = _load_calibration(calibration_path)

    if args.breakout_wfo_workers < 1:
        raise ValueError("--breakout-wfo-workers must be >= 1")
    if args.breakout_wfo_checkpoint_every < 1:
        raise ValueError("--breakout-wfo-checkpoint-every must be >= 1")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    portfolio_value = settings.portfolio_value
    target_vol_annual = settings.target_vol_annual
    max_leverage = settings.max_leverage
    cost_config = BacktestCostConfig(
        enabled=args.costs != "off",
        source=args.costs,
        taker_share=args.taker_share,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        slippage_bps=args.slippage_bps,
        include_funding=args.funding_mode != "zero",
        funding_mode=args.funding_mode,
    )
    exchange = BinanceFuturesExchange(cost_config=cost_config) if cost_config.enabled and args.costs == "binance" else None

    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    results_bundle: Dict[str, Any] = {
        "meta": {
            "timestamp": ts_tag,
            "portfolio_value": portfolio_value,
            "target_vol_annual": target_vol_annual,
            "max_leverage": max_leverage,
            "oos_fraction": args.oos_fraction,
            "oos_periods": args.oos_periods,
            "ewmac_span_mode": args.ewmac_span_mode,
            "ewmac_scalar_mode": args.ewmac_scalar_mode,
            "ewmac_dm_mode": args.ewmac_dm_mode,
            "ewmac_dm_weighting": args.ewmac_dm_weighting,
            "ewmac_dm_fixed": args.ewmac_dm_fixed,
            "ewmac_dm_cap": args.ewmac_dm_cap,
            "ewmac_final_cap": args.ewmac_final_cap,
            "breakout_scalar_mode": args.breakout_scalar_mode,
            "breakout_dm_mode": args.breakout_dm_mode,
            "breakout_dm_fixed": args.breakout_dm_fixed,
            "breakout_dm_cap": args.breakout_dm_cap,
            "breakout_dm_weighting": args.breakout_dm_weighting,
            "breakout_final_cap": args.breakout_final_cap,
            "breakout_opt_mode": args.breakout_opt_mode,
            "breakout_ensemble_percentile": args.breakout_ensemble_percentile,
            "breakout_wfo": args.breakout_wfo,
            "breakout_wfo_train_window": args.breakout_wfo_train_window,
            "breakout_wfo_test_window": args.breakout_wfo_test_window,
            "breakout_wfo_step_window": args.breakout_wfo_step_window,
            "breakout_wfo_train_mode": args.breakout_wfo_train_mode,
            "breakout_wfo_n_trials": args.breakout_wfo_n_trials,
            "breakout_wfo_workers": args.breakout_wfo_workers,
            "breakout_benchmark_ticker": args.breakout_benchmark_ticker,
            "calibration_json": str(calibration_path) if calibration_path else "",
            "calibration_use": args.calibration_use,
            "cost_config": asdict(cost_config),
        }
    }

    if args.mode in ("momentum", "both"):
        momentum_path = _resolve_path(settings.momentum_price_data_path)
        if not momentum_path.exists():
            raise FileNotFoundError(f"Momentum price data not found: {momentum_path}")
        price_frame = pd.read_pickle(momentum_path).drop_duplicates()
        momentum_validation = validate_price_frame(
            price_frame,
            path=momentum_path,
            configured_frequency=settings.momentum_data_frequency,
            strict=settings.strict_data_frequency,
        )
        for warning in momentum_validation.warnings:
            print(f"[momentum:data] {warning}")
        tickers = list(price_frame.columns)
        base_ewmac_factors = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128)]
        data_frequency = momentum_validation.effective_frequency
        ewmac_factors = _resolve_ewmac_factors(
            base_ewmac_factors,
            span_mode=args.ewmac_span_mode,
            data_frequency=data_frequency,
        )

        split = split_train_oos(price_frame.index, oos_fraction=args.oos_fraction, oos_periods=args.oos_periods or None)
        train_frame = price_frame if split.first_oos_ts is None else price_frame.reindex(split.train_index).copy()

        print(f"[momentum] Optimizing {len(tickers)} tickers on {len(train_frame)} rows.")
        calibration_momentum = calibration_payload.get("momentum") if calibration_payload else None
        if args.calibration_json and calibration_momentum is None:
            raise ValueError("Calibration JSON does not include a momentum section.")
        use_calibration_scalars = calibration_momentum is not None and args.calibration_use in ("scalars", "both")
        use_calibration_dm = calibration_momentum is not None and args.calibration_use in ("dm", "both")

        if use_calibration_scalars:
            if "ewmac_scalars" not in calibration_momentum:
                raise ValueError("Calibration JSON momentum section missing ewmac_scalars.")
            if "ewmac_factors" in calibration_momentum:
                calib_factors = _parse_ewmac_factors(calibration_momentum["ewmac_factors"])
                if calib_factors != ewmac_factors:
                    raise ValueError("Calibration EWMAC factors do not match current run factors.")
            ewmac_scalars = _parse_ewmac_scalars(calibration_momentum["ewmac_scalars"])
        elif args.ewmac_scalar_mode == "calibrated":
            calibration_frame = train_frame
            ewmac_scalars = _calibrate_ewmac_scalars(
                calibration_frame,
                ewmac_factors,
                vol_lookback=args.vol_lookback_momentum,
                fallback_scalars=cta.CARVER_EWMAC_FORECAST_SCALARS,
                base_factors=base_ewmac_factors,
            )
        else:
            ewmac_scalars = {
                actual: cta.CARVER_EWMAC_FORECAST_SCALARS.get(base, float("nan"))
                for base, actual in zip(base_ewmac_factors, ewmac_factors)
            }
        missing_scalars = [k for k, v in ewmac_scalars.items() if not pd.notnull(v)]
        if missing_scalars:
            raise ValueError(f"Missing EWMAC scalars for factors: {missing_scalars}")

        if use_calibration_dm:
            if "ewmac_dm_value" not in calibration_momentum:
                raise ValueError("Calibration JSON momentum section missing ewmac_dm_value.")
            ewmac_dm = float(calibration_momentum["ewmac_dm_value"])
        elif args.ewmac_dm_mode == "correlation":
            corr_matrix = _compute_ewmac_corr_matrix(
                train_frame,
                ewmac_factors,
                scalars=ewmac_scalars,
                vol_lookback=args.vol_lookback_momentum,
            )
            equal_weights = np.full(len(ewmac_factors), 1.0 / len(ewmac_factors))
            ewmac_dm = _forecast_diversification_multiplier(
                equal_weights,
                corr_matrix,
                dm_cap=args.ewmac_dm_cap,
            )
        else:
            ewmac_dm = args.ewmac_dm_fixed

        momentum_params = _optimize_momentum_weights(
            train_frame,
            tickers,
            ewmac_factors=ewmac_factors,
            forecast_scalars=ewmac_scalars,
            vol_lookback=args.vol_lookback_momentum,
            n_trials=args.n_trials_momentum,
            data_frequency=data_frequency,
            portfolio_value=portfolio_value,
            target_vol_annual=target_vol_annual,
            max_leverage=max_leverage,
            diversification_multiplier=ewmac_dm,
            cap_final_forecast=args.ewmac_final_cap,
        )

        momentum_params_path = output_dir / f"optimized_momentum_weights_{ts_tag}.pkl"
        pd.to_pickle(momentum_params, momentum_params_path)
        print(f"[momentum] Saved params to {momentum_params_path}")

        momentum_results: Dict[str, Any] = {}
        ewmac_dm_by_ticker: Dict[str, float] | None = None
        if not use_calibration_dm and args.ewmac_dm_mode == "correlation" and args.ewmac_dm_weighting == "optuna":
            ewmac_dm_by_ticker = {}
            for ticker in tickers:
                params = momentum_params.get(ticker, {})
                weights = params.get("weights")
                if not weights:
                    continue
                corr_matrix = _compute_ewmac_corr_matrix(
                    train_frame[[ticker]].dropna(),
                    ewmac_factors,
                    scalars=ewmac_scalars,
                    vol_lookback=args.vol_lookback_momentum,
                )
                dm = _forecast_diversification_multiplier(
                    np.asarray(weights, dtype=float),
                    corr_matrix,
                    dm_cap=args.ewmac_dm_cap,
                )
                ewmac_dm_by_ticker[ticker] = dm

        for ticker in tqdm(tickers, desc="Momentum backtest"):
            params = momentum_params.get(ticker, {})
            if params.get("status") != "success":
                continue
            weights = params["weights"]
            price = price_frame[ticker].dropna()
            if price.empty:
                continue
            dm_for_ticker = ewmac_dm_by_ticker.get(ticker, ewmac_dm) if ewmac_dm_by_ticker else ewmac_dm
            signal = _build_ewmac_signal(
                price,
                weights,
                ewmac_factors,
                scalars=ewmac_scalars,
                diversification_multiplier=dm_for_ticker,
                vol_lookback=args.vol_lookback_momentum,
                cap_final_forecast=args.ewmac_final_cap,
            )
            gross_returns, returns, equity, positions_usd, costs_df = _run_backtest(
                price,
                signal,
                ticker=ticker,
                portfolio_value=portfolio_value,
                target_vol_annual=target_vol_annual,
                data_frequency=data_frequency,
                vol_lookback=args.vol_lookback_momentum,
                max_leverage=max_leverage,
                cost_config=cost_config,
                exchange=exchange,
            )
            metrics_full = _compute_metrics(returns, data_frequency, n_trials=args.n_trials_momentum)
            metrics_gross = _compute_metrics(gross_returns, data_frequency, n_trials=args.n_trials_momentum)

            metrics_train: Dict[str, float] | None = None
            metrics_oos: Dict[str, float] | None = None
            if split.first_oos_ts is not None and split.oos_n > 0:
                returns_train = returns.reindex(split.train_index)
                returns_oos = returns.reindex(split.oos_index)
                metrics_train = _compute_metrics(returns_train, data_frequency, n_trials=args.n_trials_momentum)
                metrics_oos = _compute_metrics(returns_oos, data_frequency)

            params_with_dm = dict(params)
            params_with_dm["dm"] = dm_for_ticker
            momentum_results[ticker] = {
                "metrics": metrics_full,
                "metrics_gross": metrics_gross,
                "metrics_train": metrics_train,
                "metrics_oos": metrics_oos,
                "params": params_with_dm,
                "series": pd.DataFrame(
                    {
                        "price": price,
                        "signal": signal,
                        "returns": returns,
                        "gross_returns": gross_returns,
                        "net_returns": returns,
                        "equity": equity,
                        "positions_usd": positions_usd,
                    }
                ),
            }
            for col in costs_df.columns:
                momentum_results[ticker]["series"][col] = costs_df[col]

        results_bundle["momentum"] = {
            "params_path": str(momentum_params_path),
            "data_frequency": data_frequency,
            "configured_frequency": momentum_validation.configured_frequency,
            "inferred_frequency": momentum_validation.inferred_frequency,
            "effective_frequency": momentum_validation.effective_frequency,
            "data_validation": asdict(momentum_validation),
            "first_oos_ts": split.first_oos_ts,
            "train_rows": len(split.train_index),
            "oos_rows": len(split.oos_index),
            "ewmac_factors": ewmac_factors,
            "ewmac_span_mode": args.ewmac_span_mode,
            "ewmac_scalar_mode": args.ewmac_scalar_mode,
            "ewmac_dm_mode": args.ewmac_dm_mode,
            "ewmac_scalar_source": "calibration-json" if use_calibration_scalars else args.ewmac_scalar_mode,
            "ewmac_dm_source": "calibration-json" if use_calibration_dm else args.ewmac_dm_mode,
            "ewmac_dm_value": ewmac_dm,
            "ewmac_dm_cap": args.ewmac_dm_cap,
            "ewmac_dm_weighting": args.ewmac_dm_weighting,
            "ewmac_final_cap": args.ewmac_final_cap,
            "ewmac_scalars": {str(k): float(v) for k, v in ewmac_scalars.items()},
            "vol_lookback": args.vol_lookback_momentum,
            "results": momentum_results,
        }

    if args.mode in ("breakout", "both"):
        if not settings.breakout_price_data_path:
            raise FileNotFoundError("Breakout price data not found; set MOMO_BREAKOUT_PRICE_DATA_PATH.")
        breakout_path = _resolve_path(settings.breakout_price_data_path)
        if not breakout_path.exists():
            raise FileNotFoundError(f"Breakout price data not found: {breakout_path}")
        price_frame = pd.read_pickle(breakout_path).drop_duplicates()
        breakout_validation = validate_price_frame(
            price_frame,
            path=breakout_path,
            configured_frequency=settings.breakout_data_frequency,
            strict=settings.strict_data_frequency,
        )
        for warning in breakout_validation.warnings:
            print(f"[breakout:data] {warning}")
        tickers = list(price_frame.columns)
        breakout_horizons = [10, 20, 40, 80, 160]
        data_frequency = breakout_validation.effective_frequency
        wfo_step = args.breakout_wfo_step_window or args.breakout_wfo_test_window
        wfo_trials = args.breakout_wfo_n_trials or args.n_trials_breakout
        benchmark_payload: Dict[str, Any] | None = None
        if args.breakout_benchmark_ticker in price_frame.columns:
            bench_price = price_frame[args.breakout_benchmark_ticker].dropna()
            bench_returns = bench_price.pct_change()
            benchmark_payload = pd.DataFrame({"price": bench_price, "returns": bench_returns})

        split = split_train_oos(price_frame.index, oos_fraction=args.oos_fraction, oos_periods=args.oos_periods or None)
        train_frame = price_frame if split.first_oos_ts is None else price_frame.reindex(split.train_index).copy()

        calibration_breakout = calibration_payload.get("breakout") if calibration_payload else None
        if args.calibration_json and calibration_breakout is None:
            raise ValueError("Calibration JSON does not include a breakout section.")
        use_breakout_calibration_scalars = calibration_breakout is not None and args.calibration_use in ("scalars", "both")
        use_breakout_calibration_dm = calibration_breakout is not None and args.calibration_use in ("dm", "both")

        if use_breakout_calibration_scalars:
            if "breakout_scalars" not in calibration_breakout:
                raise ValueError("Calibration JSON breakout section missing breakout_scalars.")
            if "breakout_horizons" in calibration_breakout:
                calib_horizons = _parse_breakout_horizons(calibration_breakout["breakout_horizons"])
                if calib_horizons != breakout_horizons:
                    raise ValueError("Calibration breakout horizons do not match current run horizons.")
            breakout_scalars = _parse_breakout_scalars(calibration_breakout["breakout_scalars"])
        elif args.breakout_scalar_mode == "calibrated":
            breakout_scalars = _calibrate_breakout_scalars(
                train_frame,
                breakout_horizons,
                fallback_scalars=cta.CARVER_BREAKOUT_FORECAST_SCALARS,
            )
        else:
            breakout_scalars = {
                h: cta.CARVER_BREAKOUT_FORECAST_SCALARS.get(h, float("nan")) for h in breakout_horizons
            }
        missing_scalars = [h for h, v in breakout_scalars.items() if not pd.notnull(v)]
        if missing_scalars:
            raise ValueError(f"Missing breakout scalars for horizons: {missing_scalars}")

        if use_breakout_calibration_dm:
            if "breakout_dm_value" not in calibration_breakout:
                raise ValueError("Calibration JSON breakout section missing breakout_dm_value.")
            breakout_dm = float(calibration_breakout["breakout_dm_value"])
        elif args.breakout_dm_mode == "correlation":
            corr_matrix = _compute_breakout_corr_matrix(
                train_frame,
                breakout_horizons,
                scalars=breakout_scalars,
            )
            equal_weights = np.full(len(breakout_horizons), 1.0 / len(breakout_horizons))
            breakout_dm = _forecast_diversification_multiplier(
                equal_weights,
                corr_matrix,
                dm_cap=args.breakout_dm_cap,
            )
        else:
            breakout_dm = args.breakout_dm_fixed

        breakout_params_input_path: Path | None = None
        breakout_params_path = output_dir / f"optimized_breakout_params_{ts_tag}.pkl"
        if args.breakout_params_input:
            breakout_params_input_path = Path(args.breakout_params_input)
            if not breakout_params_input_path.is_absolute():
                breakout_params_input_path = _resolve_path(breakout_params_input_path)
            if not breakout_params_input_path.exists():
                raise FileNotFoundError(f"Breakout params input not found: {breakout_params_input_path}")
            breakout_params = pd.read_pickle(breakout_params_input_path)
            if not isinstance(breakout_params, dict):
                raise ValueError(f"Breakout params input must contain a dict: {breakout_params_input_path}")
            print(f"[breakout] Loaded params from {breakout_params_input_path}")
        else:
            print(f"[breakout] Optimizing {len(tickers)} tickers on {len(train_frame)} rows.")
            breakout_params = _optimize_breakout_weights(
                train_frame,
                tickers,
                breakout_horizons=breakout_horizons,
                vol_lookback=args.vol_lookback_breakout,
                n_trials=args.n_trials_breakout,
                data_frequency=data_frequency,
                portfolio_value=portfolio_value,
                target_vol_annual=target_vol_annual,
                max_leverage=max_leverage,
                diversification_multiplier=breakout_dm,
                forecast_scalars=breakout_scalars,
                cap_final_forecast=args.breakout_final_cap,
                opt_mode=args.breakout_opt_mode,
                top_percentile=args.breakout_ensemble_percentile,
            )
        pd.to_pickle(breakout_params, breakout_params_path)
        print(f"[breakout] Saved params to {breakout_params_path}")

        breakout_results: Dict[str, Any] = {}
        breakout_dm_by_ticker: Dict[str, float] | None = None
        if not use_breakout_calibration_dm and args.breakout_dm_mode == "correlation" and args.breakout_dm_weighting == "optuna":
            breakout_dm_by_ticker = {}
            for ticker in tickers:
                params = breakout_params.get(ticker, {})
                weights = params.get("weights")
                if not weights:
                    continue
                corr_matrix = _compute_breakout_corr_matrix(
                    train_frame[[ticker]].dropna(),
                    breakout_horizons,
                    scalars=breakout_scalars,
                )
                dm = _forecast_diversification_multiplier(
                    np.asarray(weights, dtype=float),
                    corr_matrix,
                    dm_cap=args.breakout_dm_cap,
                )
                breakout_dm_by_ticker[ticker] = dm
        for ticker in tqdm(tickers, desc="Breakout backtest"):
            params = breakout_params.get(ticker, {})
            if params.get("status") != "success":
                continue
            weights = params["weights"]
            price = price_frame[ticker].dropna()
            if price.empty:
                continue
            dm_for_ticker = breakout_dm_by_ticker.get(ticker, breakout_dm) if breakout_dm_by_ticker else breakout_dm
            signal = _build_breakout_signal(
                price,
                weights,
                breakout_horizons,
                scalars=breakout_scalars,
                diversification_multiplier=dm_for_ticker,
                cap_final_forecast=args.breakout_final_cap,
            )
            gross_returns, returns, equity, positions_usd, costs_df = _run_backtest(
                price,
                signal,
                ticker=ticker,
                portfolio_value=portfolio_value,
                target_vol_annual=target_vol_annual,
                data_frequency=data_frequency,
                vol_lookback=args.vol_lookback_breakout,
                max_leverage=max_leverage,
                cost_config=cost_config,
                exchange=exchange,
            )
            metrics_full = _compute_metrics(returns, data_frequency, n_trials=args.n_trials_breakout)
            metrics_gross = _compute_metrics(gross_returns, data_frequency, n_trials=args.n_trials_breakout)

            metrics_train: Dict[str, float] | None = None
            metrics_oos: Dict[str, float] | None = None
            if split.first_oos_ts is not None and split.oos_n > 0:
                returns_train = returns.reindex(split.train_index)
                returns_oos = returns.reindex(split.oos_index)
                metrics_train = _compute_metrics(returns_train, data_frequency, n_trials=args.n_trials_breakout)
                metrics_oos = _compute_metrics(returns_oos, data_frequency)

            params_with_dm = dict(params)
            params_with_dm["dm"] = dm_for_ticker
            breakout_results[ticker] = {
                "metrics": metrics_full,
                "metrics_gross": metrics_gross,
                "metrics_train": metrics_train,
                "metrics_oos": metrics_oos,
                "params": params_with_dm,
                "series": pd.DataFrame(
                    {
                        "price": price,
                        "signal": signal,
                        "returns": returns,
                        "gross_returns": gross_returns,
                        "net_returns": returns,
                        "equity": equity,
                        "positions_usd": positions_usd,
                    }
                ),
            }
            for col in costs_df.columns:
                breakout_results[ticker]["series"][col] = costs_df[col]

        wfo_checkpoint_path: Path | None = None
        if args.breakout_wfo:
            print(
                "[breakout] Walk-forward: "
                f"train={args.breakout_wfo_train_window}, "
                f"test={args.breakout_wfo_test_window}, "
                f"step={wfo_step}, workers={args.breakout_wfo_workers}."
            )
            wfo_kwargs = {
                "breakout_horizons": breakout_horizons,
                "forecast_scalars": breakout_scalars,
                "vol_lookback": args.vol_lookback_breakout,
                "n_trials": wfo_trials,
                "data_frequency": data_frequency,
                "portfolio_value": portfolio_value,
                "target_vol_annual": target_vol_annual,
                "max_leverage": max_leverage,
                "diversification_multiplier": breakout_dm,
                "cap_final_forecast": args.breakout_final_cap,
                "opt_mode": args.breakout_opt_mode,
                "top_percentile": args.breakout_ensemble_percentile,
                "train_window": args.breakout_wfo_train_window,
                "test_window": args.breakout_wfo_test_window,
                "step_window": wfo_step,
                "train_mode": args.breakout_wfo_train_mode,
                "cost_config": cost_config,
            }

            def _record_wfo(ticker: str, wf_result: Dict[str, Any]) -> None:
                price = price_frame[ticker].dropna()
                params = breakout_params.get(ticker, {})
                dm_for_ticker = breakout_dm_by_ticker.get(ticker, breakout_dm) if breakout_dm_by_ticker else breakout_dm
                payload = _ensure_breakout_payload(
                    breakout_results,
                    ticker,
                    price=price,
                    params=params,
                    dm_value=dm_for_ticker,
                )
                payload["walkforward"] = _normalise_walkforward_payload(wf_result, data_frequency)

            if args.breakout_wfo_checkpoint_path:
                wfo_checkpoint_path = Path(args.breakout_wfo_checkpoint_path)
                if not wfo_checkpoint_path.is_absolute():
                    wfo_checkpoint_path = _resolve_path(wfo_checkpoint_path)
            else:
                wfo_checkpoint_path = output_dir / f"breakout_wfo_checkpoint_{ts_tag}.pkl"
            wfo_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            wfo_completed: Dict[str, Dict[str, Any]] = {}
            if wfo_checkpoint_path.exists():
                loaded_checkpoint = pd.read_pickle(wfo_checkpoint_path)
                if not isinstance(loaded_checkpoint, dict):
                    raise ValueError(f"WFO checkpoint must contain a dict: {wfo_checkpoint_path}")
                wfo_completed = {
                    ticker: result
                    for ticker, result in loaded_checkpoint.items()
                    if ticker in price_frame.columns and isinstance(result, dict)
                }
                for ticker, wf_result in sorted(wfo_completed.items()):
                    _record_wfo(ticker, wf_result)
                print(f"[breakout] Loaded {len(wfo_completed)} WFO checkpoint records from {wfo_checkpoint_path}")

            def _save_wfo_checkpoint() -> None:
                tmp_path = wfo_checkpoint_path.with_name(f"{wfo_checkpoint_path.name}.tmp")
                pd.to_pickle(wfo_completed, tmp_path)
                tmp_path.replace(wfo_checkpoint_path)

            remaining_wfo_tickers = [ticker for ticker in tickers if ticker not in wfo_completed]
            print(
                "[breakout] Walk-forward checkpoint: "
                f"{len(wfo_completed)} done, {len(remaining_wfo_tickers)} remaining, path={wfo_checkpoint_path}"
            )

            if args.breakout_wfo_workers == 1:
                for ticker in tqdm(remaining_wfo_tickers, desc="Breakout WFO"):
                    price = price_frame[ticker].dropna()
                    wf_result = _run_walkforward_breakout(
                        price,
                        ticker=ticker,
                        **wfo_kwargs,
                    )
                    wfo_completed[ticker] = wf_result
                    _record_wfo(ticker, wf_result)
                    if len(wfo_completed) % args.breakout_wfo_checkpoint_every == 0:
                        _save_wfo_checkpoint()
            else:
                with ProcessPoolExecutor(
                    max_workers=args.breakout_wfo_workers,
                    mp_context=mp.get_context("spawn"),
                ) as executor:
                    futures = [
                        executor.submit(
                            _run_walkforward_breakout_worker,
                            ticker,
                            price_frame[ticker].dropna(),
                            wfo_kwargs,
                        )
                        for ticker in remaining_wfo_tickers
                    ]
                    with tqdm(total=len(futures), desc="Breakout WFO") as pbar:
                        for future in as_completed(futures):
                            ticker, wf_result = future.result()
                            wfo_completed[ticker] = wf_result
                            _record_wfo(ticker, wf_result)
                            if len(wfo_completed) % args.breakout_wfo_checkpoint_every == 0:
                                _save_wfo_checkpoint()
                            pbar.update(1)
            _save_wfo_checkpoint()

            wfo_summary_path = _write_walkforward_summary_csv(
                breakout_results,
                breakout_params,
                output_dir=output_dir,
                ts_tag=ts_tag,
            )
            print(f"[breakout] Saved walk-forward summary to {wfo_summary_path}")

        results_bundle["breakout"] = {
            "params_path": str(breakout_params_path),
            "data_frequency": data_frequency,
            "configured_frequency": breakout_validation.configured_frequency,
            "inferred_frequency": breakout_validation.inferred_frequency,
            "effective_frequency": breakout_validation.effective_frequency,
            "data_validation": asdict(breakout_validation),
            "first_oos_ts": split.first_oos_ts,
            "train_rows": len(split.train_index),
            "oos_rows": len(split.oos_index),
            "breakout_horizons": breakout_horizons,
            "breakout_scalar_mode": args.breakout_scalar_mode,
            "breakout_dm_mode": args.breakout_dm_mode,
            "breakout_scalar_source": "calibration-json" if use_breakout_calibration_scalars else args.breakout_scalar_mode,
            "breakout_dm_source": "calibration-json" if use_breakout_calibration_dm else args.breakout_dm_mode,
            "breakout_dm_value": breakout_dm,
            "breakout_dm_cap": args.breakout_dm_cap,
            "breakout_dm_weighting": args.breakout_dm_weighting,
            "breakout_final_cap": args.breakout_final_cap,
            "breakout_opt_mode": args.breakout_opt_mode,
            "breakout_ensemble_percentile": args.breakout_ensemble_percentile,
            "breakout_params_input": str(breakout_params_input_path) if breakout_params_input_path else None,
            "breakout_wfo": args.breakout_wfo,
            "breakout_wfo_train_window": args.breakout_wfo_train_window,
            "breakout_wfo_test_window": args.breakout_wfo_test_window,
            "breakout_wfo_step_window": wfo_step,
            "breakout_wfo_train_mode": args.breakout_wfo_train_mode,
            "breakout_wfo_n_trials": wfo_trials,
            "breakout_wfo_workers": args.breakout_wfo_workers,
            "breakout_wfo_checkpoint_path": str(wfo_checkpoint_path) if wfo_checkpoint_path else None,
            "breakout_benchmark_ticker": args.breakout_benchmark_ticker,
            "breakout_scalars": {str(k): float(v) for k, v in breakout_scalars.items()},
            "vol_lookback": args.vol_lookback_breakout,
            "benchmark": benchmark_payload,
            "results": breakout_results,
        }

    results_path = output_dir / f"backtest_results_bundle_{ts_tag}.pkl"
    pd.to_pickle(results_bundle, results_path)
    print(f"Saved backtest bundle to {results_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
