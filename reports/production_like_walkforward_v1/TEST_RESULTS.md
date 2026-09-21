# Validation and test results

Run date: 20 September 2026
Methodology: `crypto-trend-production-like-walkforward-v1`

## Study execution

- 178,560 unique configurations tested.
- 892,800 causal configuration/fold evaluations completed.
- Five untouched annual outer folds: 2022, 2023, 2024, 2025 and 2026 YTD.
- Two history treatments: expanding and rolling 730-day.
- 97 portfolio assets and 581 fitted price histories.
- Exactly three momentum and three breakout models selected per family/fold.
- The selected portfolio was reconstructed from timestamped held positions with fees, slippage and event-level funding.

## Funding and futures eligibility

- 433,572 funding events processed.
- 433,171 events have finite settlement mark prices.
- 116,992 missing marks were backfilled from Binance USD-M eight-hour mark-price candles.
- 401 unresolved events occur before the first eligible futures date.
- Zero unresolved funding marks occur after eligibility.
- Maximum absolute simulated position before futures eligibility: `0.0`.
- Funding is nonzero in every outer-test year beginning in 2022.

## Automated validation

`python -m pytest -q`

```text
99 passed, 3 dependency deprecation warnings in 22.24s
```

`python scripts/validate_production_like_results.py`

```text
PASS: causal selection, configuration catalog, accounting, shadow ledger, and production baseline reconcile.
```

Both standalone HTML reports passed the editorial report validator:

```text
PASS: offline render dependencies, local links and anchors.
PASS: offline render dependencies, local links and anchors.
```

Visual QA was performed at 1440 × 1200 pixels on both reports. Accounting reconciled exactly between ticker contributions and portfolio net returns, gross/net exposure and the fee/slippage/funding decomposition. Legacy production manifest and report hashes remained unchanged.

## Headline stitched results

| History treatment | Portfolio | Net Sharpe | Annual return | Annual volatility | Maximum drawdown |
|---|---|---:|---:|---:|---:|
| Rolling 730-day | Breakout ensemble | 0.505 | 10.62% | 21.04% | -21.47% |
| Rolling 730-day | Momentum ensemble | 0.279 | 5.72% | 20.49% | -31.10% |
| Rolling 730-day | 50/50 combined | 0.405 | 8.24% | 20.38% | -24.14% |
| Expanding | 50/50 combined | -0.124 | -2.44% | 19.74% | -34.23% |

The rolling combined fold Sharpes were `-0.089`, `0.936`, `0.687`, `0.017` and `0.469` for 2022 through 2026 YTD respectively. These remain historical results and should not be read as prospective validation.
