# Crypto Trend Production-Like Walk-Forward v1

This directory is the GitHub-safe publication bundle for the causal crypto momentum and Carver-style breakout study completed on 20 September 2026.

## Read the reports

- [Comprehensive walk-forward report](crypto_trend_production_like_walkforward_20260920.html)
- [Portfolio composition and ticker-position report](crypto_trend_production_like_composition_20260920.html)
- [System architecture and methodology](ARCHITECTURE.md)
- [Validation and test results](TEST_RESULTS.md)

Download the HTML files and open them locally for the fully interactive Plotly charts.

## Reproducible configurations

- `methodology_config.json` is the locked study design.
- `configurations/selected_deployable_configurations.json` contains every selected fold-level configuration.
- `configurations/selected_outer_fold_rosters.json` contains the annual three-momentum/three-breakout rosters.
- `configurations/all_tested_configurations.json.zip` contains all 178,560 exact tested configurations. The uncompressed JSON is 85,870,216 bytes.

## Results

- `metrics/` contains readable CSV summaries for stitched performance, outer folds and annual selection.
- `ledgers/research_result_ledgers.zip` contains the complete Parquet result ledgers: all 892,800 candidate-fold scores, selections, daily portfolio state, daily ticker positions, rolling metrics, fold metrics and stitched returns.
- `study_manifest.json` records methodology version, causal folds, input hashes, funding coverage and run status.
- `report_manifest.json` records the inputs used to generate the HTML reports.
- `SHA256SUMS.txt` verifies every published artifact except the checksum file itself.

## Data handling

The bundle contains derived research results, configurations and documentation only. It does not contain exchange credentials, `.env` files, raw downloaded market data, SQLite shadow ledgers or production-order state. The source data remains local and is identified only by deterministic hashes in the study manifest.

## Important limitations

- Results are causal historical reconstructions, not elapsed prospective evidence.
- The universe uses the current-universe historical fallback and therefore retains survivor bias.
- The broad configuration search carries multiple-testing and winner's-curse risk.
- Production signals were not modified; the six-model portfolio remains shadow-only.
