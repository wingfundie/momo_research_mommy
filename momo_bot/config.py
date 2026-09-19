from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


_load_dotenv_if_available()


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _float_env(name: str, default: float) -> float:
    raw = _env(name, None)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = _env(name, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _optional_float_env(name: str) -> Optional[float]:
    raw = _env(name, None)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name, None)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_windows_path(raw: str) -> str:
    if os.name != "nt":
        return raw
    cleaned = raw.strip().strip('"').strip("'")
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", cleaned)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        return f"{drive}:\\{rest}"
    match = re.match(r"^\\\\?mnt\\\\([a-zA-Z])\\\\(.*)$", cleaned)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\").replace("\\\\", "\\")
        return f"{drive}:\\{rest}"
    return cleaned


def _path_from_raw(raw: str) -> Path:
    return Path(_normalize_windows_path(raw)).expanduser()


def _path_env(name: str, default: Optional[Path]) -> Optional[Path]:
    raw = _env(name, None)
    if raw is None:
        return default
    return _path_from_raw(raw)


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    charts_dir: Path
    momo_charts_dir: Path

    telegram_bot_token: Optional[str]
    telegram_verify_ssl: bool

    binance_api_key: Optional[str]
    binance_api_secret: Optional[str]
    binance_verify_ssl: bool

    momentum_price_data_path: Path
    momentum_params_path: Path

    breakout_price_data_path: Optional[Path]
    breakout_params_path: Optional[Path]
    breakout_pairs_params_path: Optional[Path]
    portfolio_value: float
    target_vol_annual: float
    max_leverage: Optional[float]
    momentum_data_frequency: str
    breakout_data_frequency: str
    strict_data_frequency: bool
    new_ticker_start_date: str
    keep_inactive_tickers: bool
    allow_older_strategy_bundle: bool

    cost_source: str
    cost_enabled: bool
    futures_maker_fee: float
    futures_taker_fee: float
    taker_share: float
    slippage_bps: float
    include_funding: bool
    funding_mode: str

    market_data_mode: str
    refresh_universe: str
    binance_rest_weight_per_minute: int
    binance_rest_cooldown_buffer_seconds: int
    binance_ws_shard_size: int

    @property
    def breakout_enabled(self) -> bool:
        return bool(
            self.breakout_price_data_path
            and self.breakout_params_path
            and self.breakout_price_data_path.exists()
            and self.breakout_params_path.exists()
        )


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = _path_env("MOMO_DATA_DIR", base_dir / "data_store") or (base_dir / "data_store")
    charts_dir = _path_env("MOMO_CHARTS_DIR", base_dir / "charts") or (base_dir / "charts")
    momo_charts_dir = _path_env("MOMO_MOMO_CHARTS_DIR", base_dir / "momo_charts") or (base_dir / "momo_charts")

    momentum_price_data_path = _path_env(
        "MOMO_MOMENTUM_PRICE_DATA_PATH", data_dir / "tickers_price_data_4h.pkl"
    ) or (data_dir / "tickers_price_data_4h.pkl")
    momentum_params_path = _path_env(
        "MOMO_MOMENTUM_PARAMS_PATH", data_dir / "optimized_crypto_weights_carver.pkl"
    ) or (data_dir / "optimized_crypto_weights_carver.pkl")

    breakout_price_default = data_dir / "crypto_tickers_1d.pkl"
    breakout_params_default = data_dir / "optimized_breakout_params.pkl"
    breakout_pairs_params_default = data_dir / "optimized_breakout_params_pairs.pkl"
    portfolio_value = _float_env("MOMO_PORTFOLIO_VALUE", 100000.0)
    target_vol_annual = _float_env("MOMO_TARGET_VOL_ANNUAL", 0.2)
    max_leverage = _optional_float_env("MOMO_MAX_LEVERAGE")
    momentum_data_frequency = (_env("MOMO_MOMENTUM_DATA_FREQUENCY", "auto") or "auto").lower()
    breakout_data_frequency = (_env("MOMO_BREAKOUT_DATA_FREQUENCY", "auto") or "auto").lower()
    strict_data_frequency = _bool_env("MOMO_STRICT_DATA_FREQUENCY", False)
    new_ticker_start_date = _env("MOMO_NEW_TICKER_START_DATE", "2020-07-05") or "2020-07-05"
    keep_inactive_tickers = _bool_env("MOMO_KEEP_INACTIVE_TICKERS", True)
    allow_older_strategy_bundle = _bool_env("MOMO_ALLOW_OLDER_STRATEGY_BUNDLE", True)

    cost_source = (_env("MOMO_COST_SOURCE", "default") or "default").lower()
    cost_enabled = _bool_env("MOMO_COST_ENABLED", True)
    futures_maker_fee = _float_env("MOMO_FUTURES_MAKER_FEE", 0.0002)
    futures_taker_fee = _float_env("MOMO_FUTURES_TAKER_FEE", 0.0004)
    taker_share = _float_env("MOMO_TAKER_SHARE", 1.0)
    slippage_bps = _float_env("MOMO_SLIPPAGE_BPS", 0.0)
    include_funding = _bool_env("MOMO_INCLUDE_FUNDING", True)
    funding_mode = (_env("MOMO_FUNDING_MODE", "zero") or "zero").lower()
    market_data_mode = (
        _env("MOMO_MARKET_DATA_MODE", "websocket_closed_candles")
        or "websocket_closed_candles"
    ).lower()
    refresh_universe = (_env("MOMO_REFRESH_UNIVERSE", "optimized") or "optimized").lower()
    binance_rest_weight_per_minute = _int_env("MOMO_BINANCE_REST_WEIGHT_PER_MINUTE", 600)
    binance_rest_cooldown_buffer_seconds = _int_env("MOMO_BINANCE_REST_COOLDOWN_BUFFER_SECONDS", 60)
    binance_ws_shard_size = _int_env("MOMO_BINANCE_WS_SHARD_SIZE", 100)

    return Settings(
        base_dir=base_dir,
        data_dir=data_dir,
        charts_dir=charts_dir,
        momo_charts_dir=momo_charts_dir,
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
        telegram_verify_ssl=_bool_env("MOMO_TELEGRAM_VERIFY_SSL", True),
        binance_api_key=_env("BINANCE_API_KEY"),
        binance_api_secret=_env("BINANCE_API_SECRET"),
        binance_verify_ssl=_bool_env("MOMO_BINANCE_VERIFY_SSL", True),
        momentum_price_data_path=momentum_price_data_path,
        momentum_params_path=momentum_params_path,
        breakout_price_data_path=_path_env("MOMO_BREAKOUT_PRICE_DATA_PATH", breakout_price_default),
        breakout_params_path=_path_env("MOMO_BREAKOUT_PARAMS_PATH", breakout_params_default),
        breakout_pairs_params_path=_path_env("MOMO_BREAKOUT_PAIRS_PARAMS_PATH", breakout_pairs_params_default),
        portfolio_value=portfolio_value,
        target_vol_annual=target_vol_annual,
        max_leverage=max_leverage,
        momentum_data_frequency=momentum_data_frequency,
        breakout_data_frequency=breakout_data_frequency,
        strict_data_frequency=strict_data_frequency,
        new_ticker_start_date=new_ticker_start_date,
        keep_inactive_tickers=keep_inactive_tickers,
        allow_older_strategy_bundle=allow_older_strategy_bundle,
        cost_source=cost_source,
        cost_enabled=cost_enabled,
        futures_maker_fee=futures_maker_fee,
        futures_taker_fee=futures_taker_fee,
        taker_share=taker_share,
        slippage_bps=slippage_bps,
        include_funding=include_funding,
        funding_mode=funding_mode,
        market_data_mode=market_data_mode,
        refresh_universe=refresh_universe,
        binance_rest_weight_per_minute=binance_rest_weight_per_minute,
        binance_rest_cooldown_buffer_seconds=binance_rest_cooldown_buffer_seconds,
        binance_ws_shard_size=binance_ws_shard_size,
    )


settings = load_settings()
