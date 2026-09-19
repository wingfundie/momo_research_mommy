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

### Tests

```bash
pytest -q
```

### Notes
- Do **not** commit `.env` (it contains secrets); `.gitignore` excludes it.
- The bot writes chart images to `./charts/` and `./momo_charts/` at runtime.
