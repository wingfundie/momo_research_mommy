from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


EWMAC_RULES: tuple[tuple[int, int], ...] = ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128))
CARVER_BREAKOUT_HORIZONS: tuple[int, ...] = (10, 20, 40, 80, 160)
CARVER_BREAKOUT_SCALARS: tuple[tuple[int, float], ...] = (
    (10, 0.60), (20, 0.67), (40, 0.70), (80, 0.73), (160, 0.74),
)
CARVER_BREAKOUT_TURNOVER: tuple[tuple[int, float], ...] = (
    (10, 74.7), (20, 35.1), (40, 17.4), (80, 8.7), (160, 4.2),
)


@dataclass(frozen=True)
class UniverseConfig:
    max_members: int = 100
    entry_rank: int = 90
    exit_rank: int = 110
    min_median_quote_volume_usd: float = 1_000_000.0
    min_history_days: int = 90
    max_gap_days: int = 2
    exclude_stablecoins: bool = True
    dex_min_liquidity_usd: float = 500_000.0
    dex_min_median_volume_usd: float = 100_000.0
    splice_overlap_days: int = 7


@dataclass(frozen=True)
class SignalConfig:
    rules: tuple[tuple[int, int], ...] = EWMAC_RULES
    volatility_lookback: int = 90
    forecast_cap: float = 20.0
    minimum_training_days: int = 365
    refit_schedule: str = "quarterly"
    shrinkage_to_equal: float = 0.80
    max_rule_weight: float = 0.25
    smoothing_days: int = 125


@dataclass(frozen=True)
class BreakoutSignalConfig:
    horizons: tuple[int, ...] = CARVER_BREAKOUT_HORIZONS
    forecast_scalars: tuple[tuple[int, float], ...] = CARVER_BREAKOUT_SCALARS
    expected_turnover: tuple[tuple[int, float], ...] = CARVER_BREAKOUT_TURNOVER
    smoothing_divisor: int = 4
    forecast_cap: float = 20.0
    cost_budget_sr: float = 0.15
    fdm_mode: str = "causal_expanding"
    fixed_fdm: float = 1.20
    fdm_cap: float = 2.5
    minimum_training_days: int = 365
    refit_schedule: str = "quarterly"


@dataclass(frozen=True)
class RiskConfig:
    portfolio_value: float = 100_000.0
    annual_volatility_target: float = 0.15
    gross_leverage_cap: float = 1.0
    single_ticker_risk_cap: float = 0.10
    rebalance: str = "daily"
    activation: str = "next_open"
    taker_share: float = 1.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage_bps: float = 5.0


@dataclass(frozen=True)
class ResearchConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ledger_path: Path = Path("data_store/crypto_momentum_research.sqlite")
    output_dir: Path = Path("data_store/crypto_momentum_research")
    model_version: str = "crypto-momentum-v1"
    shadow_only: bool = True


VOLATILITY_GRID = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
GROSS_CAP_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
TICKER_CAP_GRID = tuple(value / 100.0 for value in range(5, 41, 5))
VOLATILITY_LOOKBACK_GRID = (60, 90, 180, 360)
SLIPPAGE_BPS_GRID = (0.0, 2.0, 5.0, 10.0)
TAKER_SHARE_GRID = (0.0, 0.5, 1.0)
