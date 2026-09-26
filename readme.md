## momo_bot_local

Telegram bot + strategy code for momentum/breakout style signals.

### Setup

1) Create a virtualenv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Create a `.env` file (see `.env.example`) and set:
- `TELEGRAM_BOT_TOKEN`
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

3) Data files
- Momentum reads from `./data_store/tickers_price_data_4h.pkl` and params from `./data_store/optimized_crypto_weights_carver.pkl` by default.
- Breakout is optional and only enabled if both files exist:
  - `MOMO_BREAKOUT_PRICE_DATA_PATH` (defaults to `./data_store/crypto_tickers_1d.pkl`)
  - `MOMO_BREAKOUT_PARAMS_PATH` (defaults to `./data_store/optimized_breakout_params.pkl`)

You can override paths via env vars in `.env`.

### Run

```bash
python3 mom_break_bot.py
```

### Corrected XSec20 momentum commands

Momentum commands use the pinned `xsm_ic20_dollar_neutral_vol60` model by default.
Append `legacy` to use the original per-ticker parameter engine, for example
`/mom_sig_str legacy` or `/mc SOL legacy`.

- `/mom_sig_str [N]` and `/mom_sig_sr [N]`: relative-strength and standalone-Sharpe rankings.
- `/portfolio [share|usd]`, `/rebalance`, `/risk [share|usd]`: the read-only $100k model portfolio.
- `/performance [30d|90d|1y|all] [asset]`: model performance, optionally against any Binance USDT asset.
- `/distribution [30d|90d|180d|1y|all]`: ticker paths, 25th/75th percentiles, mean signal and breadth.
- `/model`, `/changes [N]`, `/health`: model lineage, daily changes, and data freshness.
- `/portfolio_momo`: assess current Binance positions against momentum and XSec signals; it never drives model sizing.

Every registered bot command also accepts slash-free private-chat input. For example,
`portfolio usd`, `mc SOL`, `performance 1y BTC`, and `mom_sig_str 10` behave exactly like
their `/portfolio usd`, `/mc SOL`, `/performance 1y BTC`, and `/mom_sig_str 10` forms.

Historical commands accept flexible lookbacks such as `45d`, `12w`, `6m`, `2y`, and `all`.
Use `manual` or `help` in Telegram, or read [the complete command manual](docs/telegram_bot_manual.md).

All generated charts share the dark composition-card theme used by the Telegram analytics views.

Refresh completed daily candles, opens, funding and the immutable runtime snapshot before the
bot's reporting window:

```bash
python scripts/refresh_xsec20_snapshot.py
```

Run `python scripts/refresh_xsec20_snapshot.py --refresh-universe` on the monthly maintenance
schedule. A failed required close/open coverage check keeps the last published inputs intact;
Telegram commands continue to serve that snapshot with a stale-data warning.

Publish the validated production snapshot and the lightweight research dashboard inputs to the
cross-asset dashboard after a successful refresh:

```bash
python scripts/publish_dashboard_snapshot.py \
  --publish \
  --base-url https://acausal-cross-asset-dashboard.onrender.com \
  --env-file "../upd_dash_board/.env"
```

The publisher uploads an immutable version under `systematic/crypto/momentum/versions/` and asks
the dashboard to activate it only after all hashes and table contracts pass. A failed upload or
activation leaves the previous dashboard version active.

For the daily scheduled job, refresh and publish in one fail-fast command:

```bash
python scripts/refresh_and_publish_dashboard.py \
  --base-url https://acausal-cross-asset-dashboard.onrender.com \
  --env-file "../upd_dash_board/.env"
```

Use `--refresh-universe` on the monthly run. The publish step does not run when market-data
refresh or runtime validation fails, and the dashboard keeps serving its last active version.

### Shadow crypto momentum research

The causal portfolio research pipeline is isolated from production. It supports time-series and
cross-sectional momentum, event-level funding, point-in-time universe snapshots, deterministic
manifests and a dedicated dashboard. See [docs/crypto_momentum_research.md](docs/crypto_momentum_research.md).

### Tests

```bash
pytest -q
```

### Notes
- Do **not** commit `.env` (it contains secrets); `.gitignore` excludes it.
- The bot writes chart images to `./charts/` and `./momo_charts/` at runtime.
