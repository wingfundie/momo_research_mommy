# Production regression baseline

Captured before the crypto research rebuild on 2026-09-19.

- Legacy suite: 56 passed, 3 dependency deprecation warnings.
- `tickers_price_data_4h.pkl`: 1,977 rows × 442 symbols, 2020-07-05 through 2025-12-20;
  SHA-256 `9846371ac7e5b064a0cca3ad2de359de54811338b640ddf3624d843a76026298`.
- `optimized_crypto_weights_carver.pkl`: SHA-256
  `8fb6b2ad33a95300105f8c9a0eac1b0a377417fc9f3d2308c94ea98c2249c7db`.
- `crypto_tickers_1d.pkl`: SHA-256
  `e24f61adad18594e3c7e8854a0b913d6fffe60835ba74fc4c398c24231bb653d`.
- `optimized_breakout_params.pkl`: SHA-256
  `04bcae1ce001ed612acb38db56b679fe61dcb406c93591f1b49060f752085720`.

The data and model bundles remain ignored by Git. These hashes allow the exact local baseline inputs to
be identified without publishing generated data. The new implementation lives under `momo_bot/research`
and the legacy production modules are intentionally unchanged.
