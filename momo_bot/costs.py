from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CommissionRate:
    symbol: str
    maker: float
    taker: float
    source: str


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    tick_size: float | None = None
    step_size: float | None = None
    min_qty: float | None = None
    min_notional: float | None = None


@dataclass(frozen=True)
class BacktestCostConfig:
    enabled: bool = True
    source: str = "default"
    taker_share: float = 1.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage_bps: float = 0.0
    include_funding: bool = True
    funding_mode: str = "zero"


@dataclass(frozen=True)
class CostBreakdown:
    fees: pd.Series
    slippage: pd.Series
    funding: pd.Series
    total: pd.Series
    turnover_notional: pd.Series


def calculate_costs(
    positions_usd: pd.Series,
    *,
    commission: CommissionRate,
    config: BacktestCostConfig,
    funding_rates: pd.Series | None = None,
) -> CostBreakdown:
    index = positions_usd.index
    zero = pd.Series(0.0, index=index)
    if not config.enabled:
        return CostBreakdown(
            fees=zero.copy(),
            slippage=zero.copy(),
            funding=zero.copy(),
            total=zero.copy(),
            turnover_notional=zero.copy(),
        )

    target = positions_usd.fillna(0.0)
    turnover = target.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(target.iloc[0])
    turnover = turnover.fillna(0.0)

    taker_share = min(max(float(config.taker_share), 0.0), 1.0)
    fee_rate = taker_share * float(commission.taker) + (1.0 - taker_share) * float(commission.maker)
    fees = turnover * fee_rate
    slippage = turnover * (float(config.slippage_bps) / 10_000.0)

    if config.include_funding and funding_rates is not None and not funding_rates.empty:
        aligned_funding = funding_rates.reindex(index, method="ffill").fillna(0.0)
        funding = target.shift(1).fillna(0.0) * aligned_funding
    else:
        funding = zero.copy()

    total = fees + slippage + funding
    return CostBreakdown(
        fees=fees,
        slippage=slippage,
        funding=funding,
        total=total,
        turnover_notional=turnover,
    )


def config_from_settings(settings, *, enabled: bool | None = None, source: str | None = None) -> BacktestCostConfig:
    return BacktestCostConfig(
        enabled=settings.cost_enabled if enabled is None else enabled,
        source=settings.cost_source if source is None else source,
        taker_share=settings.taker_share,
        maker_fee=settings.futures_maker_fee,
        taker_fee=settings.futures_taker_fee,
        slippage_bps=settings.slippage_bps,
        include_funding=settings.include_funding,
        funding_mode=settings.funding_mode,
    )
