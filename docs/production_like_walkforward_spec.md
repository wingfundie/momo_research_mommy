# Production-like crypto trend walk-forward specification

Status: locked from the Lavish review completed on 20 September 2026.

## Locked decisions

1. Run two historical nested walk-forward variants:
   - Expanding history: every selection and refit uses all observations available before its cutoff.
   - Rolling history: every selection and fitted parameter uses the trailing 730 calendar days, with a 365-day minimum. The 730-day implementation default retains the planned 2022 outer fold with the available September 2019 start date.
2. Use annual outer roster decisions for 2022, 2023, 2024, 2025 and 2026 year-to-date. Preserve every selected model's declared quarterly, semiannual, annual or frozen internal refit behavior during its test fold.
3. Publish rolling metrics over 90-day and 365-day windows.
4. Seed the prospective shadow roster from the completed results already stored in the repository. Historical outer-fold rosters must still be reselected causally at each cutoff.
5. Select three time-series momentum configurations and three breakout configurations using prior-only net Sharpe, followed by lower turnover, lower gross cap, lower target volatility and configuration ID as deterministic tie-breaks.
6. Holdout performance cannot participate in historical or prospective roster selection.

The complete machine-readable contract and exact seed configuration IDs are in `configs/production_like_walkforward_v1.json`.

## Locked prospective shadow roster

### Time-series momentum

- `82f7747eb0444a6af97f` — `ts_equal_vol60`
- `9ec0476face086163247` — `ts_shrink_75_semiannual_vol60`
- `207ba4a7e9eaed7159d9` — `ts_shrink_90_semiannual_vol60`

### Breakout

- `17dc3b5200249930172d` — `breakout_crypto_equal_frozen_vol60`
- `28a6f4703071d8cd7b6d` — `breakout_crypto_shrink_75_frozen_vol60`
- `01d51dec9a9d2826e1d0` — `breakout_crypto_shrink_90_frozen_vol60`

These are raw validation leaders selected from the existing completed studies, not from 2026 holdout performance. The three breakout seeds are nearly identical frozen-volatility-60 variants. The report must disclose their similarity and must not claim that they supply three independent sources of diversification.

## Required reporting

- Model-by-model and fold-by-fold net return, volatility, Sharpe, Sortino, Calmar, drawdown, turnover, fees, slippage, funding, BTC beta and tail-risk metrics.
- Stitched outer-fold OOS results for six model slots, both family ensembles and the combined portfolio.
- 90-day and 365-day rolling Sharpe, return, volatility, beta, correlation, turnover and funding drag.
- Daily signed weights, long/short gross exposure, net exposure, ticker counts, concentration, component attribution and roster/refit markers for every strategy.
- Separate expanding-versus-rolling-window comparisons using the same outer folds and cost assumptions.
- A prospective append-only shadow ledger that reconciles emitted signals with reconstructed signals and simulated or actual funding.

## Headline acceptance rule

The historical headline is the concatenation of outer test folds only. No observation may be produced by a roster selected using that observation or later data. A full-history backcast of today's six seed models is diagnostic only and must remain separate from the causal walk-forward record.

## Execution and rebuild

```text
python scripts/execute_production_like_walkforward.py
python scripts/initialize_production_shadow.py
python scripts/build_production_like_walkforward_report.py
```

Versioned results are written to `data_store/crypto_momentum_research/production_like_walkforward_v1/`. The comprehensive report is `reports/crypto_trend_production_like_walkforward_20260920.html`; the per-strategy holdings companion is `reports/crypto_trend_production_like_composition_20260920.html`.

The directory also contains `all_tested_configurations.json`, with every unique exact configuration and its reproducible hash, and `selected_deployable_configurations.json`, with the three selected configurations per family and outer fold. Fold-level scores remain in `candidate_score_ledger.parquet`.
