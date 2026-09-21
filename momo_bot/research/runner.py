from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ResearchConfig, RiskConfig, SignalConfig
from .ledger import ResearchLedger
from .models import PortfolioRun, SignalRecord
from .portfolio import SimulationResult, risk_target_weights, simulate_portfolio
from .signals import (
    basket_rank_weights,
    breakout_forecast,
    combine_forecasts,
    continuous_rank_weights,
    cross_sectional_percentiles,
    expanding_refit_weights,
    long_only_rank_weights,
    rule_forecasts,
)


def frame_hash(frame: pd.DataFrame | pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def config_hash(value) -> str:
    return hashlib.sha256(json.dumps(asdict(value), sort_keys=True, default=str).encode()).hexdigest()


def build_time_series_forecasts(prices: pd.DataFrame, signal: SignalConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    components = {symbol: rule_forecasts(prices[symbol], signal) for symbol in prices}
    weights = expanding_refit_weights(prices, components, signal)
    final = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for symbol, rules in components.items():
        if weights.empty:
            final[symbol] = combine_forecasts(rules, cap=signal.forecast_cap)
        else:
            aligned = weights.reindex(rules.index).ffill()
            numerator = (rules * aligned).sum(axis=1, min_count=1)
            denominator = rules.notna().mul(aligned).sum(axis=1).replace(0.0, np.nan)
            final[symbol] = (numerator / denominator).clip(-signal.forecast_cap, signal.forecast_cap)
    return final, components


def build_reference_forecasts(
    prices: pd.DataFrame,
    signal: SignalConfig,
    *,
    mode: str,
    per_ticker_weights: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Comparable equal, pooled and legacy per-ticker forecast models."""
    components = {symbol: rule_forecasts(prices[symbol], signal) for symbol in prices}
    if mode == "pooled":
        return build_time_series_forecasts(prices, signal)[0]
    output = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for symbol, frame in components.items():
        if mode == "equal":
            weights = None
        elif mode == "per_ticker_optuna":
            if per_ticker_weights is None or symbol not in per_ticker_weights:
                raise ValueError(f"Missing legacy per-ticker weights for {symbol}")
            weights = per_ticker_weights[symbol]
        else:
            raise ValueError(f"Unknown reference mode: {mode}")
        output[symbol] = combine_forecasts(frame, weights, cap=signal.forecast_cap)
    return output


def construct_sleeve(
    forecasts: pd.DataFrame,
    prices: pd.DataFrame,
    risk: RiskConfig,
    *,
    sleeve: str,
    basket_fraction: float = 0.20,
    membership: pd.DataFrame | None = None,
    volatility_lookback: int = 90,
) -> pd.DataFrame:
    volatility = prices.pct_change(fill_method=None).rolling(
        volatility_lookback, min_periods=volatility_lookback
    ).std() * np.sqrt(365)
    if sleeve == "time_series":
        raw = forecasts.div(20.0)
        if membership is not None:
            raw = raw.where(membership.reindex_like(raw).fillna(False))
    else:
        percentiles = cross_sectional_percentiles(forecasts, membership)
        if sleeve == "cross_sectional_continuous":
            raw = continuous_rank_weights(percentiles)
        elif sleeve == "cross_sectional_basket":
            raw = basket_rank_weights(percentiles, basket_fraction)
        elif sleeve == "cross_sectional_long_only":
            raw = long_only_rank_weights(percentiles, forecasts, basket_fraction)
        else:
            raise ValueError(f"Unknown sleeve: {sleeve}")
    return risk_target_weights(raw, volatility, risk,
                               asset_returns=prices.pct_change(fill_method=None),
                               portfolio_volatility_lookback=volatility_lookback)


def run_research(
    prices: pd.DataFrame,
    config: ResearchConfig = ResearchConfig(),
    *,
    sleeve: str = "time_series",
    membership: pd.DataFrame | None = None,
    funding_cashflows_usd: pd.Series | None = None,
    funding_mode: str = "historical",
    ledger: ResearchLedger | None = None,
) -> tuple[PortfolioRun, SimulationResult, dict]:
    if funding_mode == "historical" and funding_cashflows_usd is None:
        raise ValueError("Headline research requires event-level historical funding")
    forecasts, components = build_time_series_forecasts(prices, config.signal)
    targets = construct_sleeve(forecasts, prices, config.risk, sleeve=sleeve, membership=membership,
                               volatility_lookback=config.signal.volatility_lookback)
    result = simulate_portfolio(prices, targets, config.risk, funding_cashflows_usd=funding_cashflows_usd)
    hashes = {"prices": frame_hash(prices), "membership": frame_hash(membership) if membership is not None else "none",
              "funding": frame_hash(funding_cashflows_usd) if funding_cashflows_usd is not None else funding_mode}
    deterministic_manifest = {
        "schema_version": 1, "model_version": config.model_version,
        "sleeve": sleeve, "funding_mode": funding_mode, "input_hashes": hashes,
        "signal_config": asdict(config.signal), "risk_config": asdict(config.risk),
        "provider_coverage": {}, "status": "complete",
    }
    run_id = hashlib.sha256(json.dumps(deterministic_manifest, sort_keys=True, default=str).encode()).hexdigest()[:20]
    manifest = {**deterministic_manifest, "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat()}
    run = PortfolioRun(run_id=run_id, created_at=datetime.now(timezone.utc), sleeve=sleeve,
                       model_version=config.model_version, refit_schedule=config.signal.refit_schedule,
                       risk_scenario=asdict(config.risk), input_hashes=hashes, config=asdict(config.signal),
                       provider_coverage={}, status="complete", metrics=result.metrics)
    if ledger is not None:
        daily = pd.DataFrame({"gross_return": result.gross_returns, "net_return": result.net_returns,
                              "equity": result.equity, "fees": result.fees, "slippage": result.slippage,
                              "funding_cashflow": result.funding_cashflow, "turnover": result.turnover})
        ledger.append_run(run, daily, manifest)
        ledger.append_signals(signal_records(forecasts, components, targets, config))
    return run, result, manifest


def signal_records(forecasts, components, targets, config: ResearchConfig) -> list[SignalRecord]:
    data_digest, cfg_digest = frame_hash(forecasts), config_hash(config.signal)
    percentiles = cross_sectional_percentiles(forecasts)
    output: list[SignalRecord] = []
    for timestamp in forecasts.index:
        for symbol in forecasts.columns:
            value = forecasts.at[timestamp, symbol]
            rules = components[symbol].loc[timestamp]
            output.append(SignalRecord(
                timestamp=pd.Timestamp(timestamp).to_pydatetime(), symbol=symbol, model_name="ewmac_pooled",
                model_version=config.model_version, data_hash=data_digest, config_hash=cfg_digest,
                component_forecasts={name: None if pd.isna(v) else float(v) for name, v in rules.items()},
                combined_forecast=None if pd.isna(value) else float(value),
                cross_sectional_percentile=None if pd.isna(percentiles.at[timestamp, symbol]) else float(percentiles.at[timestamp, symbol]),
                direction="UNAVAILABLE" if pd.isna(value) else ("LONG" if value > 0 else "SHORT" if value < 0 else "FLAT"),
                target_notional=float(targets.at[timestamp, symbol] * config.risk.portfolio_value),
            ))
    return output


def breakout_reference(prices: pd.DataFrame, horizon: int = 64) -> pd.DataFrame:
    return pd.DataFrame({symbol: breakout_forecast(prices[symbol], horizon=horizon) for symbol in prices})


def scenario_grid(config: ResearchConfig) -> Iterable[ResearchConfig]:
    """Yield the requested risk matrix deterministically."""
    from .config import GROSS_CAP_GRID, TICKER_CAP_GRID, VOLATILITY_GRID
    for target in VOLATILITY_GRID:
        for gross in GROSS_CAP_GRID:
            for ticker_cap in TICKER_CAP_GRID:
                for rebalance in ("daily", "weekly", "monthly"):
                    yield replace(config, risk=replace(config.risk, annual_volatility_target=target,
                                                       gross_leverage_cap=gross,
                                                       single_ticker_risk_cap=ticker_cap,
                                                       rebalance=rebalance))
