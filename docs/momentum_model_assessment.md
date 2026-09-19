# Momentum Model Repository Assessment

Assessment date: 2026-06-02

This document implements the proposed assessment plan for the local momentum bot repository. It maps the repo design, current usage, backtesting methodology, signal generation, sizing engine, data pipelines, and objective evaluation criteria.

## Executive Summary

The repository is a signal-generation and reporting system for crypto futures momentum/breakout models. The inspected code reads Binance futures data and current positions, generates Telegram reports and charts, optimizes strategy weights, and writes backtest bundles. I did not find order-placement logic in the inspected paths.

The core model stack is workable but not yet cleanly reproducible or objectively reliable as a momentum research platform. The largest issues are:

- The repo-local momentum file is named and treated as `4h` data, but its timestamp spacing is daily.
- The latest saved backtest bundle contains breakout results only; momentum is absent from the latest bundle.
- Live breakout signal generation does not reproduce the latest breakout backtest signal path when the bundle uses calibrated scalars and DM.
- Importing core strategy modules creates a Binance client immediately, causing import-time network/SSL failure in normal research scripts.
- The main backtest path does not model trading costs, slippage, or funding.
- The train/OOS split has a shared boundary timestamp in current usage.

The test suite now passes:

- `python -m pytest -q`: 12 passed, 4 warnings.
- `python -m compileall -q ...`: passed.

## Architecture And Usage Map

Primary entrypoints:

- `mom_break_bot.py`: combined Telegram bot for momentum and breakout commands. It loads latest bundles/params, refreshes market data, generates summary tables and charts, and reads current Binance positions for portfolio alignment reports.
- `reoptimize_all.py`: main optimization and backtest bundle generator for momentum and breakout.
- `calibrate_forecasts.py`: scalar and diversification multiplier calibration CLI.
- `build_results_dashboard.py`: Plotly Dash dashboard for saved backtest bundles.
- `optimizer.py`: older/batch breakout optimization helper.
- `mommy_bot.py`: momentum-only Telegram/support logic used by `mom_break_bot.py`.
- `portfolio_strategy.py`: central strategy, forecast wrapper, sizing, PnL, optimization objective, and legacy plotting utilities.
- `momo_bot/`: smaller package layer for config, candles, data cache, Binance data access, exchange client, and forecast/scalar primitives.

Core data flow:

```text
Binance klines / local pickle data
  -> freshness and completed-candle filtering
  -> EWMAC or breakout raw forecasts
  -> scalar application and clipping
  -> optimized weight combination
  -> diversification multiplier
  -> final signal
  -> risk-targeted position sizing
  -> delayed-fill PnL backtest
  -> saved params and bundle
  -> dashboard / Telegram tables / charts
```

Default commands from the repo docs:

- Setup: `python3 -m venv .venv && pip install -r requirements.txt`
- Run bot: `python3 mom_break_bot.py`
- Test: `pytest -q`

Observed local caveat: direct `pytest` was not on PATH in this shell, but `python -m pytest -q` works.

## Data Pipeline Assessment

Configured paths:

- Momentum price data: `data_store/tickers_price_data_4h.pkl`
- Momentum params: `data_store/optimized_crypto_weights_carver.pkl`
- Breakout price data: `C:\Users\HomePC\Desktop\acausal capital\momentum_run\mom bot\crypto_tickers_1d.pkl`
- Breakout params: `C:\Users\HomePC\Desktop\acausal capital\momentum_run\optimized_breakout_params_10204080160_vol80_120_carver_final.pkl`
- Breakout pair params: `C:\Users\HomePC\Desktop\acausal capital\momentum_run\optimized_breakout_params_pairs_10204080160.pkl`

Local artifact facts:

| Artifact | Shape / Keys | Date Range | Notes |
| --- | ---: | --- | --- |
| `tickers_price_data_4h.pkl` | 1977 x 442 | 2020-07-05 to 2025-12-20 | Mode timestamp delta is 1 day, not 4h. Missing cells: 57.961 percent. |
| external breakout price data | 2113 x 442 | 2020-07-05 to 2026-05-05 | Daily spacing. Missing cells: 54.468 percent. |
| `optimized_crypto_weights_carver.pkl` | 433 tickers | n/a | 353 success, 80 skipped. |
| latest breakout params in repo | 442 tickers | n/a | 350 success, 92 skipped. |
| external breakout params | 350 tickers | n/a | 350 success. |
| external pair params | 27 pairs | n/a | 23 success, 4 skipped. |

Freshness checks on 2026-06-02:

- Momentum `4h` path latest timestamp: 2025-12-20 00:00:00.
- Required latest completed `4h` candle: 2026-06-01 20:00:00.
- Breakout external daily path latest timestamp: 2026-05-05 00:00:00.
- Required latest completed daily candle: 2026-06-01 00:00:00.

Both configured data sources are stale.

Reproducibility concern:

- Breakout defaults depend on legacy files outside this repo. A fresh clone of this repo will not reproduce the current breakout configuration unless those external files are also present or env vars override the paths.

## Signal Generation Assessment

Momentum/EWMAC path:

- Raw forecast: fast EWMA minus slow EWMA.
- Vol normalization: rolling standard deviation of price differences.
- Signal scaling: Carver-style forecast scalars or calibrated scalars in optimization scripts.
- Signal cap: individual forecasts are clipped to +/-20.
- Combination: optimized per-ticker weights.
- Diversification multiplier: fixed or correlation-derived in `reoptimize_all.py`.
- Live momentum wrapper currently matches the local Carver fallback params for a representative ticker.

Momentum parity check:

- Ticker: `1000000MOGUSDT`.
- Backtest-style signal and live wrapper signal matched exactly at the last common point.
- This parity only holds for the current Carver fallback path. If a future momentum bundle uses calibrated scalars/DM, the live wrapper must be checked again because `carver_gen_signal` defaults are hardcoded unless explicitly passed.

Breakout path:

- Raw forecast: price position between rolling high and rolling low.
- Smoothing: EWMA over roughly horizon / 4.
- Signal scaling: Carver or calibrated scalars.
- Signal cap: individual forecasts clipped to +/-20.
- Combination: optimized per-ticker weights.
- Diversification multiplier: fixed or calibration/correlation-derived in backtests.

Breakout parity check:

- Latest bundle: `backtest_results_bundle_20260124_231556.pkl`.
- Latest bundle contains breakout only.
- Bundle scalar source: `calibration-json`.
- Bundle DM: `1.3696534265689793`.
- Live-style unified generation hardcodes Carver scalars and `div_multiplier = 1.24`.
- Ticker checked: `1000000MOGUSDT`.
- Last common backtest signal: `-5.251410890073924`.
- Last common live-style signal: `-4.010156778359704`.
- Last absolute difference: `1.2412541117142197`.
- Median absolute difference across common points: `3.2302208917043007`.
- Max absolute difference: `5.647596091135064`.

Conclusion: breakout live reports do not objectively reproduce the latest backtest methodology when calibrated bundle settings are used.

## Sizing And PnL Engine Assessment

Sizing function: `portfolio_strategy._position_size_from_forecast`.

Mechanics:

- Forecast is reindexed to price and filled with zero.
- Annual target volatility is converted to per-period risk using `_periods_per_year`.
- Rolling price-diff volatility is used as the denominator.
- Position units are calculated as:

```text
positions = (forecast / 10) * (per_period_risk / rolling_price_diff_vol)
positions_usd = positions * price
```

- Optional max leverage caps USD notional and recomputes units.

Strengths:

- Zero forecast maps to flat exposure.
- Forecast magnitude scales notional directionally.
- Higher price-diff volatility reduces position size.
- Max leverage cap is implemented and now covered by characterization tests.

Weaknesses:

- Using price-diff volatility is sensitive to instrument price scale and contract conventions.
- Main backtest does not include fees, slippage, funding, borrow, spread, minimum notional, quantity precision, or exchange filters.
- `max_leverage` defaults to `None`, so notional can become large during low-volatility windows.
- Live Telegram position-size output is advisory/reporting only; no order-sizing/execution engine was found.

PnL path:

- `get_pnl` applies positions with delayed fill behavior by shifting positions before calculating price-difference PnL.
- Main `_run_backtest` returns per-period returns, cumulative equity, and USD notional positions.
- Transaction cost handling exists in the simplified grid backtest but not in the primary optimization/backtest path.

## Backtesting And Optimization Assessment

Main backtest generator: `reoptimize_all.py`.

Methodology:

- Reads price frame.
- Resolves factor/horizon lists.
- Splits train/test by `oos_fraction` or `oos_periods`.
- Calibrates scalars from train data or reads calibration JSON.
- Computes fixed or correlation-based diversification multiplier.
- Optimizes per-ticker weights with Optuna.
- Builds full-series signals from optimized weights.
- Sizes positions and computes delayed-fill PnL.
- Saves params and backtest result bundle.

Current bundle state:

- `backtest_results_bundle_20260124_231556.pkl`: breakout only, no OOS.
- `backtest_results_bundle_20260124_140034.pkl`: breakout only, OOS fraction 0.2.
- `backtest_results_bundle_20260124_022445.pkl`: momentum and breakout, OOS fraction 0.2.

Train/OOS boundary issue:

- `_split_train_test_index` returns `index[-oos_n]`.
- Callers create `train_frame = price_frame.loc[:split_ts]`.
- Metrics use `returns_train = returns.loc[:split_ts]` and `returns_oos = returns.loc[split_ts:]`.
- The split timestamp is included in both train and OOS slices.

This is not a large leak by row count, but it violates clean train/OOS separation and should be fixed before using OOS results as decision evidence.

Metric coverage:

- Included: Sharpe, CAGR, max drawdown, total return, skew, upper/lower tail quantiles, t-stat, optional deflated Sharpe.
- Missing or incomplete for objective trading assessment: fees, funding, slippage, turnover in the main path, realized volatility versus target, exposure utilization, drawdown duration, benchmark-relative alpha, hit rate, and parameter stability.

## Live Bot And Reporting Assessment

`mom_break_bot.py` is the main combined bot.

Observed behavior:

- Registers momentum commands always.
- Registers breakout commands only if breakout data and params are available.
- Loads latest strategy params preferentially from latest bundle, then falls back to param pickle.
- Generates Telegram tables sorted by SR or signal strength.
- Generates Matplotlib/Plotly charts saved under `charts/` and `momo_charts/`.
- Reads current Binance futures positions to compare portfolio direction with model direction.

No order execution was found:

- Searches found Binance position reads via `futures_position_information`.
- Searches did not find `create_order`, `futures_create_order`, cancel-order, or equivalent execution calls.

Operational issue:

- Importing `portfolio_strategy.py` or `reoptimize_all.py` normally can create a Binance client immediately through a module-level `client = get_binance_client()`. In this assessment, that caused an SSL certificate failure on Binance ping before any model function could run. Core research modules should be importable without network side effects.

## Objective Scorecard

| Area | Rating | Evidence |
| --- | --- | --- |
| Data integrity | Red | Configured data is stale; momentum file labeled `4h` has daily spacing; high missing-cell rates due ticker universe evolution. |
| Causality | Yellow | Delayed-fill PnL and completed-candle filtering exist, but train/OOS split shares one timestamp. |
| Signal validity | Yellow | Forecast mechanics are understandable, but live breakout does not match latest calibrated backtest signal path. |
| Sizing engine | Yellow | Risk-targeted sizing exists and basic invariants pass, but no exchange constraints or default leverage cap. |
| Backtest realism | Red | Main backtest lacks fees, slippage, funding, and turnover/cost accounting. |
| Generalization | Yellow/Red | OOS and walk-forward machinery exists, but latest bundle has no OOS and only breakout results. |
| Operational reliability | Yellow/Red | Bot has stale-data checks, but import-time Binance client creation can break research/test workflows. |
| Reproducibility | Red | Breakout depends on external legacy files and latest bundle lacks complete strategy coverage. |
| Test coverage | Yellow | 12 tests now pass, but high-level signal/backtest/live parity and cost model tests are still missing. |

## Findings And Recommended Fixes

1. Import-time Binance client creation is a high-priority operational bug.
   - Evidence: importing `portfolio_strategy.py` attempted a Binance ping and failed with SSL verification error.
   - Fix: remove module-level `client = get_binance_client()` from research modules; create clients lazily inside live/API functions.

2. Momentum data frequency is inconsistent with code assumptions.
   - Evidence: `tickers_price_data_4h.pkl` has daily modal timestamp delta, while code annualizes as `4h`.
   - Impact: Sharpe annualization and target-risk sizing are materially distorted if the data is actually daily.
   - Fix: either regenerate true 4h data or rename/reconfigure the pipeline to `1d`; add schema checks that fail when modal spacing does not match configured frequency.

3. Data is stale.
   - Evidence: momentum latest `2025-12-20`, breakout latest `2026-05-05`, both stale on 2026-06-02.
   - Fix: refresh data, then validate completed-candle freshness before any signal/backtest report.

4. Breakout live signal generation diverges from latest backtest methodology.
   - Evidence: representative ticker signal difference of `1.2413` at the last common point and median absolute difference of `3.2302`.
   - Fix: pass bundle/calibration scalars and per-ticker or section DM into `carver_gen_signal_unified`; remove hardcoded `div_multiplier = 1.24`.

5. Latest backtest bundle is incomplete for whole-system assessment.
   - Evidence: latest bundle contains `breakout` only and no `momentum`.
   - Fix: regenerate a current `--mode both` bundle after data cleanup; preserve config metadata and exact artifact paths.

6. Train/OOS boundary is shared.
   - Evidence: training and OOS metrics both include `split_ts`.
   - Fix: use train `< split_ts` and OOS `>= split_ts`, or define split as the last train timestamp and use OOS `> split_ts`. Add tests for non-overlap.

7. Primary backtest realism is incomplete.
   - Evidence: `_run_backtest` uses PnL from positions without cost/funding model.
   - Fix: add fee bps, slippage bps, funding estimate, turnover, and position-change cost accounting to the primary path.

8. Breakout reproducibility depends on external files.
   - Evidence: configured breakout defaults resolve outside this repo.
   - Fix: document required external artifact locations or move/copy canonical artifacts into `data_store/` with env overrides.

9. High-level tests are still thin.
   - Added: sizing zero forecast, leverage cap, split helper semantics.
   - Still needed: live/backtest parity, data schema validation, no import-time network call, clean OOS non-overlap, cost model accounting, and bundle schema tests.

## Recommended Implementation Sequence

1. Remove import-time exchange side effects.
   - Make core model/research imports offline-safe.
   - Add a test that importing `portfolio_strategy` and `reoptimize_all` does not create a Binance client.

2. Fix data frequency and freshness.
   - Decide whether momentum is daily or true 4h.
   - Regenerate data accordingly.
   - Add a reusable schema validator for price frames.

3. Align live and backtest signal construction.
   - Store scalars, DM, factors/horizons, vol lookback, and cap mode in params/bundles.
   - Make live calls consume the same config that generated the params.
   - Add parity tests for one momentum and one breakout ticker/date.

4. Fix train/OOS separation.
   - Make train and OOS index slices disjoint.
   - Regenerate OOS and walk-forward bundles.

5. Add realistic trading-cost modeling.
   - Include fees, slippage, funding, turnover, min notional, precision, and leverage/exposure reports.
   - Report before-cost and after-cost metrics.

6. Regenerate objective evaluation artifacts.
   - Run current `--mode both` backtest with clean data.
   - Run OOS and walk-forward variants.
   - Compare against BTC buy-and-hold, equal-weight crypto basket, and randomized/no-signal baselines.
   - Use the dashboard only after bundle schema validation passes.

## Acceptance Criteria For A Reliable Momentum Model

- Core modules import without network calls.
- Configured data frequency matches actual timestamp spacing.
- Data is fresh to the latest completed candle before signal reporting.
- Live and backtest signals match within floating-point tolerance for the same ticker/date/config.
- Train and OOS windows are strictly disjoint.
- Backtests report after-cost performance.
- Target realized volatility is measured against configured target volatility.
- Parameters are stable across time splits and walk-forward windows.
- Latest bundle contains both momentum and breakout when the system is evaluated as a whole.
- Tests cover schema, signal parity, sizing, PnL timing, OOS split, and no-exchange-import behavior.

