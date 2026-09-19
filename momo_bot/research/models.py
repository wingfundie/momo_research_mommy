from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class AssetIdentity:
    binance_symbol: str
    base_asset: str
    quote_asset: str = "USDT"
    contract_type: str = "PERPETUAL"
    provider: str | None = None
    provider_id: str | None = None
    provider_symbol: str | None = None
    listing_time: datetime | None = None
    delisting_time: datetime | None = None
    mapping_method: str = "unique_symbol"
    mapping_provenance: str = ""


@dataclass(frozen=True)
class UniverseSnapshot:
    as_of: datetime
    symbol: str
    provider: str
    fully_diluted_valuation: float | None
    rank: int | None
    eligible: bool
    member: bool
    reasons: tuple[str, ...] = ()
    source_timestamp: datetime | None = None
    snapshot_hash: str = ""
    point_in_time: bool = True


@dataclass(frozen=True)
class PriceSegment:
    symbol: str
    source: str
    start: datetime
    end: datetime
    tradable: bool
    overlap_days: int = 0
    adjustment_ratio: float = 1.0
    validation_status: str = "unvalidated"
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SignalRecord:
    timestamp: datetime
    symbol: str
    model_name: str
    model_version: str
    data_hash: str
    config_hash: str
    component_forecasts: Mapping[str, float | None]
    combined_forecast: float | None
    cross_sectional_percentile: float | None = None
    direction: str = "UNAVAILABLE"
    quality_flags: tuple[str, ...] = ()
    target_quantity: float | None = None
    target_notional: float | None = None
    expected_funding_next: float | None = None
    expected_funding_24h: float | None = None


@dataclass(frozen=True)
class FundingEvent:
    symbol: str
    funding_time: datetime
    funding_rate: float
    mark_price: float
    position_quantity: float
    funding_cost_usd: float
    funding_cashflow_usd: float
    source: str = "binance_public"
    rate_type: str = "Regular"
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActualFundingRecord:
    transaction_id: str
    symbol: str
    asset: str
    timestamp: datetime
    income_usd: float
    matched_model: str | None = None
    matched_notional: float = 0.0
    unmatched_notional: float = 0.0
    unmatched_reason: str | None = None
    source: str = "binance_income"


@dataclass(frozen=True)
class PortfolioRun:
    run_id: str
    created_at: datetime
    sleeve: str
    model_version: str
    refit_schedule: str
    risk_scenario: Mapping[str, float | str]
    input_hashes: Mapping[str, str]
    config: Mapping[str, Any]
    provider_coverage: Mapping[str, Any]
    status: str
    metrics: Mapping[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
