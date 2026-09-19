from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momo_bot.research.config import GROSS_CAP_GRID, TICKER_CAP_GRID, VOLATILITY_GRID, ResearchConfig
from momo_bot.research.portfolio import performance_metrics, risk_target_weights, simulate_portfolio
from momo_bot.research.runner import build_reference_forecasts, build_time_series_forecasts
from momo_bot.research.signals import (
    basket_rank_weights, breakout_forecast, continuous_rank_weights, cross_sectional_percentiles,
    long_only_rank_weights,
)


OUT = Path("data_store/crypto_momentum_research/results")


def funding_coefficients(prices: pd.DataFrame, events: pd.DataFrame):
    base_index = pd.DatetimeIndex(prices.index)
    price_days = base_index.tz_localize("UTC") if base_index.tz is None else base_index.tz_convert("UTC")
    long_prices = prices.copy()
    long_prices.index = price_days.normalize()
    long_prices.index.name = "reference_day"
    long_prices = long_prices.stack(future_stack=True).rename("reference_price").reset_index()
    long_prices.columns = ["reference_day", "symbol", "reference_price"]
    work = events.loc[events["symbol"].isin(prices.columns)].copy()
    work["funding_time"] = pd.to_datetime(work["funding_time"], utc=True)
    work["settlement_day"] = work["funding_time"].dt.normalize()
    work["midnight"] = work["funding_time"].dt.hour.eq(0)
    work["reference_day"] = work["settlement_day"] - pd.to_timedelta(work["midnight"].astype(int), unit="D")
    work = work.merge(long_prices, on=["reference_day", "symbol"], how="left")
    work["coefficient"] = -work["mark_price"] / work["reference_price"] * work["funding_rate"]
    def pivot(mask):
        grouped = work.loc[mask].groupby(["settlement_day", "symbol"])["coefficient"].sum()
        frame = grouped.unstack().reindex(index=price_days.normalize(), columns=prices.columns).fillna(0.0)
        frame.index = prices.index
        return frame
    return pivot(~work["midnight"]), pivot(work["midnight"])


def evaluate(prices, raw, risk, same_coeff, midnight_coeff):
    vol = prices.pct_change(fill_method=None).rolling(90, min_periods=90).std() * np.sqrt(365)
    targets = risk_target_weights(raw, vol, risk, asset_returns=prices.pct_change(fill_method=None))
    base = simulate_portfolio(prices, targets, risk)
    funding_return = (base.weights * same_coeff).sum(axis=1) + (base.weights.shift(1).fillna(0) * midnight_coeff).sum(axis=1)
    net = base.net_returns + funding_return
    metrics = performance_metrics(net, base.turnover, funding_return)
    metrics.update({"funding_paid_return": float(-funding_return.clip(upper=0).sum()),
                    "funding_received_return": float(funding_return.clip(lower=0).sum()),
                    "total_fees_return": float(base.fees.sum()), "total_slippage_return": float(base.slippage.sum())})
    return metrics, net, base.weights, funding_return


def evaluate_targets(prices, targets, risk, same_coeff, midnight_coeff):
    base = simulate_portfolio(prices, targets, risk)
    funding_return = (base.weights * same_coeff).sum(axis=1) + (base.weights.shift(1).fillna(0) * midnight_coeff).sum(axis=1)
    net = base.net_returns + funding_return
    metrics = performance_metrics(net, base.turnover, funding_return)
    metrics.update({"funding_paid_return": float(-funding_return.clip(upper=0).sum()),
                    "funding_received_return": float(funding_return.clip(lower=0).sum()),
                    "total_fees_return": float(base.fees.sum()), "total_slippage_return": float(base.slippage.sum())})
    return metrics, net, base.weights, funding_return


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = OUT.parent / "inputs"
    prices = pd.read_parquet(inputs / "research_prices.parquet").sort_index()
    prices = prices.loc[:, prices.notna().sum() >= 90]
    events = pd.read_parquet(inputs / "funding_events.parquet")
    same_coeff, midnight_coeff = funding_coefficients(prices, events)
    config = ResearchConfig()
    pooled, _ = build_time_series_forecasts(prices, config.signal)
    equal = build_reference_forecasts(prices, config.signal, mode="equal")
    breakout = pd.DataFrame({c: breakout_forecast(prices[c], 64) for c in prices})
    ranks = cross_sectional_percentiles(pooled)
    sleeves = {
        "time_series_pooled": pooled / 20.0,
        "time_series_equal": equal / 20.0,
        "breakout_64": breakout / 20.0,
        "cross_sectional_continuous": continuous_rank_weights(ranks),
        "cross_sectional_top_bottom_10": basket_rank_weights(ranks, .10),
        "cross_sectional_top_bottom_20": basket_rank_weights(ranks, .20),
        "cross_sectional_top_bottom_30": basket_rank_weights(ranks, .30),
        "cross_sectional_long_only_20": long_only_rank_weights(ranks, pooled, .20),
    }
    rows = []
    headline_daily = {}
    headline_weights = {}
    headline_funding = {}
    total = len(VOLATILITY_GRID) * len(GROSS_CAP_GRID) * len(TICKER_CAP_GRID) * 3 * 2
    count = 0
    for sleeve_name in ("time_series_pooled", "cross_sectional_continuous"):
        raw = sleeves[sleeve_name]
        for cap in TICKER_CAP_GRID:
            unit_risk = replace(config.risk, annual_volatility_target=1.0, gross_leverage_cap=1000.0,
                                single_ticker_risk_cap=cap)
            unit_targets = risk_target_weights(
                raw, prices.pct_change(fill_method=None).rolling(90, min_periods=90).std() * np.sqrt(365), unit_risk,
                asset_returns=prices.pct_change(fill_method=None)
            )
            for target in VOLATILITY_GRID:
                target_weights = unit_targets * target
                for gross in GROSS_CAP_GRID:
                    gross_now = target_weights.abs().sum(axis=1)
                    factor = (gross / gross_now.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
                    capped_targets = target_weights.mul(factor, axis=0)
                    for rebalance in ("daily", "weekly", "monthly"):
                        risk = replace(config.risk, annual_volatility_target=target, gross_leverage_cap=gross,
                                       single_ticker_risk_cap=cap, rebalance=rebalance)
                        metrics, net, weights, funding_return = evaluate_targets(
                            prices, capped_targets, risk, same_coeff, midnight_coeff
                        )
                        rows.append({"sleeve": sleeve_name, "model": "shrink_80_primary", "volatility_window": 90,
                                     "refit": "quarterly", "target_vol": target, "gross_cap": gross,
                                     "ticker_cap": cap, "rebalance": rebalance, "taker_share": 1.0,
                                     "slippage_bps": risk.slippage_bps, **metrics})
                        count += 1
                        if count % 100 == 0: print(f"risk grid {count}/{total}", flush=True)
    default_risk = config.risk
    for name, raw in sleeves.items():
        metrics, net, weights, funding_return = evaluate(prices, raw, default_risk, same_coeff, midnight_coeff)
        rows.append({"sleeve": name, "model": name, "volatility_window": 90, "refit": "quarterly",
                     "target_vol": default_risk.annual_volatility_target, "gross_cap": default_risk.gross_leverage_cap,
                     "ticker_cap": default_risk.single_ticker_risk_cap, "rebalance": default_risk.rebalance,
                     "taker_share": default_risk.taker_share, "slippage_bps": default_risk.slippage_bps, **metrics})
        headline_daily[name] = net; headline_weights[name] = weights.iloc[-1]; headline_funding[name] = funding_return
    for taker in (0.0, .5, 1.0):
        for slip in (0.0, 2.0, 5.0, 10.0):
            risk = replace(default_risk, taker_share=taker, slippage_bps=slip)
            metrics, *_ = evaluate(prices, sleeves["time_series_pooled"], risk, same_coeff, midnight_coeff)
            rows.append({"sleeve": "time_series_pooled", "model": "cost_sensitivity", "volatility_window": 90,
                         "refit": "quarterly", "target_vol": risk.annual_volatility_target, "gross_cap": risk.gross_leverage_cap,
                         "ticker_cap": risk.single_ticker_risk_cap, "rebalance": risk.rebalance,
                         "taker_share": taker, "slippage_bps": slip, **metrics})
    results = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    results.to_parquet(OUT / "scenario_results.parquet", index=False)
    results.to_csv(OUT / "scenario_results.csv", index=False)
    pd.DataFrame(headline_daily).to_parquet(OUT / "headline_daily_returns.parquet")
    pd.DataFrame(headline_weights).T.to_csv(OUT / "latest_ticker_weights.csv")
    pd.DataFrame(headline_funding).to_parquet(OUT / "headline_daily_funding_returns.parquet")
    actual_path = inputs / "actual_funding_income_recent.parquet"
    actual = pd.read_parquet(actual_path) if actual_path.exists() else pd.DataFrame()
    summary = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "prices": list(prices.shape),
               "price_start": str(prices.index.min()), "price_end": str(prices.index.max()),
               "funding_events": len(events), "account_funding_events": len(actual),
               "account_net_funding_usd": float(actual.income_usd.sum()) if not actual.empty else None,
               "scenarios": len(results), "universe_fallback": "current_universe_historical_fallback",
               "best": results.iloc[0].to_dict()}
    (OUT / "study_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    best = results.iloc[0]
    report = f"""# Full Crypto Momentum Study\n\n## Executive result\n\nThe study evaluated **{len(results):,}** portfolio scenarios from {prices.index.min():%Y-%m-%d} through {prices.index.max():%Y-%m-%d}. The highest raw net Sharpe was **{best.net_sharpe:.2f}** for `{best.sleeve}` at a {best.target_vol:.0%} volatility target, {best.gross_cap:.1f}x gross cap, {best.ticker_cap:.0%} ticker cap and `{best.rebalance}` rebalancing. This is an exploratory maximum across a broad grid and is not corrected for multiple testing.\n\n## Funding and costs\n\nThe model used {len(events):,} Binance funding events at their event timestamps. For the selected configuration, cumulative funding paid was {best.funding_paid_return:.2%}, funding received was {best.funding_received_return:.2%}, and net funding contribution was {best.funding_return:.2%}. The headline assumes 100% taker execution and {best.slippage_bps:.0f} bps one-way slippage. Recent authenticated account history contains {len(actual):,} funding records with net cashflow ${actual.income_usd.sum() if not actual.empty else 0:,.2f}.\n\n## Important limitation\n\nNo historical point-in-time FDV archive was configured. Membership therefore uses the current top-100 eligible universe across history and is labelled `current_universe_historical_fallback`. This creates survivor bias; these results must not be interpreted as final production evidence.\n\n## Outputs\n\nScenario metrics, daily sleeve returns, daily funding contributions and current ticker weights are stored beside this report.\n"""
    (OUT / "FULL_RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__": main()
