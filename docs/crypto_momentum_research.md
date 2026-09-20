# Crypto momentum research rebuild

This pipeline is a shadow research system. It does not alter the existing production signal process or place orders.

## Scope

- Pooled fitting uses the merged active and legacy/delisted Binance USDT perpetual history.
- Portfolio membership uses the archived top-100 FDV snapshot, stablecoin/index exclusions, and a $1m trailing 30-day median Binance quote-volume floor.
- Historical point-in-time FDV membership is used only when supplied. The current run is explicitly labelled as a current-universe historical fallback.
- Headline next-open P&L uses forward Binance open-to-open returns. Next-close uses forward close-to-close returns as a sensitivity.
- Historical public funding is settled event by event. Actual account `FUNDING_FEE` records are archived separately and reconciled without assuming all account exposure belongs to the model.
- Breakout has signal-model parity with EWMAC: pooled/shrunk/equal horizon weights, quarterly through frozen refits, four volatility windows, individual-horizon controls and a legacy per-ticker Optuna reference.
- Legacy Optuna coverage is 64 valid tickers and 33 explicit equal-weight fallbacks for both EWMAC and breakout. These static full-sample references are excluded from causal headline selection.

## Rebuild order

```text
python scripts/build_crypto_research_inputs.py --prices data_store/crypto_research_prices_1d.pkl
python scripts/enrich_crypto_universe_metadata.py
python scripts/archive_portfolio_open_prices.py
python scripts/sync_research_funding.py
python scripts/execute_complete_crypto_study.py
python scripts/reconcile_actual_funding_archive.py
python scripts/rebuild_complete_study_ledger.py
python scripts/build_full_crypto_study_report.py
python scripts/build_crypto_momentum_telegram_summary.py
```

The Binance asynchronous account-history endpoint has high request weight. `archive_binance_funding_income.py` deliberately limits new jobs and polls through `MOMO_MAX_NEW_ASYNC_JOBS` and `MOMO_MAX_ASYNC_POLLS`.

## Principal outputs

- `complete_results/tested_configurations.json`: every tested model, risk, execution-cost and delay configuration.
- `complete_results/selected_configurations.json`: validation-selected headline settings with their disclosed 2026 holdout results.
- `complete_results/study_manifest.json`: data hashes, coverage and selection protocol.
- `complete_results/crypto_momentum_research.sqlite`: canonical research ledger.
- `complete_results/daily_signal_records.parquet`: model/ticker forecasts, ranks, positions, hashes and quality flags.
- `complete_results/actual_funding_reconciliation.parquet`: actual account funding, public funding rate, implied account notional and model-direction reconciliation.
- `reports/crypto_momentum_complete_study_20260920.html`: full offline research report.
- `reports/crypto_momentum_ticker_analytics_20260920.html`: searchable all-ticker analytics companion with component signals, attribution, risk and funding.

## OOS interpretation

Refits and signal activation are causal. Configuration selection uses 2024–25 only, and 2026 is disclosed afterward as a historical holdout. This is not prospective OOS evidence because the methodology followed earlier exploratory work and historical point-in-time FDV snapshots were not available for the current run. Freeze a selected JSON configuration before using prospective logs for promotion decisions.

## Dashboard

```text
python crypto_momentum_dashboard.py
```

The dashboard reads only the frozen complete-results artifacts. It exposes portfolio comparisons, individual ticker strength, the full risk grid, modeled and actual funding, expected funding, cohort attribution and capacity screens.

## FirstMate

This repository is suitable for FirstMate `direct-PR` mode after the shared contracts are merged. FirstMate itself is a cloned agent distro, not a Python package. Run it from WSL2 with GitHub CLI authentication, tmux and a supported terminal harness. Codex Desktop is not currently a FirstMate runtime backend.
