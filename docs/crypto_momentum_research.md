# Crypto momentum research pipeline

This repository now contains a separate, shadow-only crypto momentum research system under
`momo_bot/research`. It does not replace or mutate the legacy production signal path.

## Safety boundary

- Research output is never submitted as an order.
- Live credentials are read only from the environment and are not stored in run manifests.
- The headline runner refuses to proceed without historical event-level funding. Zero and estimated
  funding are permitted only when explicitly selected and remain labelled sensitivities.
- Pre-perpetual spot, aggregate and qualified DEX prices may warm a forecast, but the associated
  tradability mask prevents positions and P&L before the futures listing.

## Data flow

1. Discover Binance USD-M USDT perpetual contracts and preserve their listing/delisting interval.
2. Resolve provider identities, choosing the highest-FDV record for symbol collisions and preserving
   that decision as provenance.
3. Reconstitute the eligible universe at completed month boundaries with rank 90/110 entry/exit
   hysteresis, a 100-member maximum, stablecoin exclusion and a $1m trailing-volume floor.
4. Calculate causal EWMAC components with one configured volatility window. Warm-up values remain
   unavailable and the combined forecast is capped at ±20.
5. Construct standalone time-series or cross-sectional sleeves, apply next-candle activation and
   simulate the whole portfolio with leverage, ticker, fee and slippage constraints.
6. Settle every funding timestamp against the quantity held immediately before the event. Store paid,
   received and net funding by symbol, and reconcile actual `FUNDING_FEE` income separately.
7. Persist immutable inputs and run manifests to SQLite, then export CSV/Parquet and render the research
   dashboard/report from those same records.

## Commands

```bash
python scripts/run_crypto_momentum_research.py \
  --prices path/to/daily_prices.parquet \
  --funding-cashflows path/to/daily_funding_cashflows.parquet \
  --sleeve time_series

python crypto_momentum_dashboard.py
```

The dashboard defaults to `http://127.0.0.1:8060`. Set `MOMO_RESEARCH_LEDGER`, `MOMO_DASH_HOST` or
`MOMO_DASH_PORT` to override local paths/network settings.

## FirstMate handoff

Use `wingfundie/momo_research_mommy` in direct-PR mode from WSL2/Linux. Workers must not receive
production credentials, place live orders or enable automatic merges. Merge shared contracts first;
then isolate universe/data, funding/costs and strategy work in separate worktrees. Causality and
accounting changes require explicit review by the integration owner.

## Known external-data boundary

Free providers do not guarantee complete historical point-in-time FDV snapshots. A run using current
membership as a historical fallback must say so in its manifest and report. Imported snapshots are
accepted only with an asserted source and matching SHA-256 checksum.
