from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RiskConfig


@dataclass(frozen=True)
class SimulationResult:
    positions: pd.DataFrame
    weights: pd.DataFrame
    gross_returns: pd.Series
    fees: pd.Series
    slippage: pd.Series
    funding_cashflow: pd.Series
    net_returns: pd.Series
    equity: pd.Series
    turnover: pd.Series
    metrics: dict[str, float]


def rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    if frequency == "daily":
        return pd.Series(True, index=index)
    if frequency == "weekly":
        return pd.Series(index.weekday == 0, index=index)
    if frequency == "monthly":
        periods = index.to_period("M")
        return pd.Series(~periods.duplicated(), index=index)
    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def apply_rank_buffer(
    desired: pd.DataFrame, percentiles: pd.DataFrame, previous: pd.DataFrame, buffer: float
) -> pd.DataFrame:
    """Retain existing longs/shorts until they cross an outer percentile buffer."""
    if buffer <= 0:
        return desired
    result = desired.copy()
    prior_long = previous > 0
    prior_short = previous < 0
    result = result.where(~(prior_long & (percentiles >= buffer)), previous)
    result = result.where(~(prior_short & (percentiles <= 1.0 - buffer)), previous)
    return result


def risk_target_weights(
    raw_weights: pd.DataFrame,
    asset_volatility: pd.DataFrame,
    config: RiskConfig,
) -> pd.DataFrame:
    """Volatility-scale weights, then enforce ticker and gross-notional caps."""
    inverse_vol = raw_weights.div(asset_volatility.replace(0.0, np.nan))
    normalized = inverse_vol.div(inverse_vol.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    scaled = normalized * float(config.annual_volatility_target)
    scaled = scaled.clip(-config.single_ticker_risk_cap, config.single_ticker_risk_cap)
    gross = scaled.abs().sum(axis=1)
    factor = (float(config.gross_leverage_cap) / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
    return scaled.mul(factor, axis=0)


def simulate_portfolio(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: RiskConfig,
    *,
    funding_cashflows_usd: pd.Series | None = None,
    tradable: pd.DataFrame | None = None,
) -> SimulationResult:
    """Shared causal daily simulator for every sleeve and benchmark."""
    prices = prices.sort_index().astype(float)
    targets = target_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    if tradable is not None:
        targets = targets.where(tradable.reindex_like(targets).fillna(False), 0.0)
    mask = rebalance_mask(prices.index, config.rebalance)
    held = targets.where(mask, np.nan).ffill().fillna(0.0)
    delay = 1 if config.activation in {"next_open", "next_close"} else None
    if delay is None:
        raise ValueError("activation must be next_open or next_close")
    held = held.shift(delay).fillna(0.0)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    # Target observed at t-1 is shifted into ``held`` at t and earns t's return.
    gross = (held * returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    blended_fee = config.taker_share * config.taker_fee + (1.0 - config.taker_share) * config.maker_fee
    fees = turnover * blended_fee
    slippage = turnover * config.slippage_bps / 10_000.0
    if funding_cashflows_usd is None:
        funding_return = pd.Series(0.0, index=prices.index)
    else:
        funding_return = funding_cashflows_usd.reindex(prices.index).fillna(0.0) / config.portfolio_value
    net = gross - fees - slippage + funding_return
    equity = config.portfolio_value * (1.0 + net).cumprod()
    positions = held.mul(equity.shift(1).fillna(config.portfolio_value), axis=0).div(prices)
    return SimulationResult(
        positions=positions, weights=held, gross_returns=gross, fees=fees, slippage=slippage,
        funding_cashflow=funding_return * config.portfolio_value, net_returns=net, equity=equity,
        turnover=turnover, metrics=performance_metrics(net, turnover, funding_return),
    )


def performance_metrics(net_returns: pd.Series, turnover: pd.Series, funding_returns: pd.Series) -> dict[str, float]:
    clean = net_returns.dropna()
    annual_return = float(clean.mean() * 365) if len(clean) else np.nan
    annual_vol = float(clean.std() * np.sqrt(365)) if len(clean) > 1 else np.nan
    sharpe = annual_return / annual_vol if annual_vol and np.isfinite(annual_vol) else np.nan
    cumulative = (1.0 + clean).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0 if not cumulative.empty else pd.Series(dtype=float)
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "net_sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
        "annual_turnover": float(turnover.mean() * 365),
        "funding_return": float(funding_returns.sum()),
    }


def benchmark_weights(prices: pd.DataFrame, kind: str, fdv: pd.DataFrame | None = None) -> pd.DataFrame:
    available = prices.notna().astype(float)
    if kind == "btc_buy_hold":
        result = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if "BTCUSDT" not in result:
            raise KeyError("BTCUSDT is required for the BTC benchmark")
        result["BTCUSDT"] = available["BTCUSDT"]
        return result
    if kind == "equal_weight":
        return available.div(available.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    if kind == "fdv_weight":
        if fdv is None:
            raise ValueError("fdv is required for fdv_weight")
        eligible_fdv = fdv.reindex_like(prices).where(available.astype(bool))
        return eligible_fdv.div(eligible_fdv.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    if kind == "no_skill":
        signs = pd.DataFrame(
            np.where(np.indices(prices.shape)[1] % 2 == 0, 1.0, -1.0),
            index=prices.index, columns=prices.columns,
        ) * available
        return signs.div(signs.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    raise ValueError(f"Unsupported benchmark: {kind}")
