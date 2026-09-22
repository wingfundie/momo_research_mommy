from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from momo_bot.binance_rate_limit import BinanceRestCircuitOpen
from momo_bot.candles import check_freshness, normalize_timestamp
from momo_bot.data_validation import infer_frequency, resolve_coverage_starts, validate_price_coverage


Updater = Callable[[pd.DataFrame, str, Path], pd.DataFrame]
logger = logging.getLogger(__name__)


def _read_pickle_compat(path: Path) -> pd.DataFrame:
    """Read pandas-3 StringDtype pickles from the pandas-2 bot runtime."""
    try:
        return pd.read_pickle(path)
    except TypeError:
        import pandas.core.arrays.string_ as string_module

        original = string_module.StringDtype

        class CompatibleStringDtype(original):
            def __init__(self, storage=None, na_value=pd.NA):
                super().__init__(storage)

        string_module.StringDtype = CompatibleStringDtype
        pd.StringDtype = CompatibleStringDtype
        return pd.read_pickle(path)


@dataclass(frozen=True)
class RefreshResult:
    data: pd.DataFrame
    was_updated: bool


class MarketDataStore:
    """
    Minimal in-process cache around a wide price DataFrame stored on disk.
    Focus: keep data up-to-date without re-reading from disk for every command.
    """

    def __init__(self, *, path: Path, freq: str) -> None:
        self._path = path
        self._freq = freq
        self._lock = threading.RLock()
        self._data: Optional[pd.DataFrame] = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def freq(self) -> str:
        return self._freq

    def load(self) -> pd.DataFrame:
        with self._lock:
            if self._data is not None:
                return self._data
            if not self._path.exists():
                raise FileNotFoundError(str(self._path))
            self._data = _read_pickle_compat(self._path)
            return self._data

    def _coverage_starts(
        self,
        df: pd.DataFrame,
        required_tickers: list[str],
        onboard_dates: dict[str, pd.Timestamp] | None,
    ) -> dict[str, pd.Timestamp]:
        resolved = resolve_coverage_starts(
            df,
            required_tickers=required_tickers,
            onboard_date_by_ticker=onboard_dates,
        )
        metadata_path = self._path.with_suffix(".meta.json")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                stored = metadata.get("expected_start_by_ticker", {})
                if isinstance(stored, dict) and stored:
                    resolved.update(
                        {
                            ticker: pd.to_datetime(timestamp)
                            for ticker, timestamp in stored.items()
                            if ticker in required_tickers
                        }
                    )
            except (OSError, ValueError, TypeError):
                pass
        return resolved

    def _confirmed_unavailable(self) -> dict[str, list[pd.Timestamp]]:
        metadata_path = self._path.with_suffix(".meta.json")
        if not metadata_path.exists():
            return {}
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored = metadata.get("confirmed_unavailable_by_ticker", {})
            if not isinstance(stored, dict):
                return {}
            return {
                ticker: [pd.to_datetime(timestamp) for timestamp in timestamps]
                for ticker, timestamps in stored.items()
                if isinstance(timestamps, list)
            }
        except (OSError, ValueError, TypeError):
            return {}

    def refresh(
        self,
        *,
        updater: Updater,
        required_tickers: list[str] | None = None,
        expected_start_by_ticker: dict[str, pd.Timestamp] | None = None,
    ) -> RefreshResult:
        """
        Ensures the stored data is up to date (to the last completed candle).
        If stale, calls `updater(existing_df, freq, path)` which must also persist to `path`.
        """
        with self._lock:
            df = self.load()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("Price data index must be a DatetimeIndex")

            effective_freq = infer_frequency(df.index) if self._freq == "auto" else self._freq
            if required_tickers:
                coverage_starts = self._coverage_starts(
                    df,
                    required_tickers,
                    expected_start_by_ticker,
                )
                coverage = validate_price_coverage(
                    df,
                    required_tickers=required_tickers,
                    freq=effective_freq,
                    expected_start_by_ticker=coverage_starts,
                    allowed_missing_by_ticker=self._confirmed_unavailable(),
                )
                if coverage.is_publishable:
                    return RefreshResult(data=df, was_updated=False)
            else:
                freshness = check_freshness(df.index, freq=effective_freq)
                if freshness.is_fresh:
                    return RefreshResult(data=df, was_updated=False)

            try:
                updated = updater(df, effective_freq, self._path)
            except BinanceRestCircuitOpen as exc:
                logger.warning("Using cached %s data because Binance REST is cooling down: %s", self._path, exc)
                return RefreshResult(data=df, was_updated=False)
            if required_tickers:
                coverage_starts = self._coverage_starts(
                    updated,
                    required_tickers,
                    expected_start_by_ticker,
                )
                coverage = validate_price_coverage(
                    updated,
                    required_tickers=required_tickers,
                    freq=effective_freq,
                    expected_start_by_ticker=coverage_starts,
                    allowed_missing_by_ticker=self._confirmed_unavailable(),
                )
                if not coverage.is_publishable:
                    raise RuntimeError(
                        "Refreshed price data failed coverage validation: "
                        f"fresh={coverage.fresh_ticker_count}/{coverage.required_ticker_count}, "
                        f"missing_internal_cells={coverage.missing_internal_cells}"
                    )
            self._data = updated
            return RefreshResult(data=updated, was_updated=True)

    def merge_closed_candle(self, candle: Any) -> bool:
        """
        Merge one closed Binance kline into the local wide close-price store.

        Unknown symbols are ignored so optimized-only streams cannot expand the
        stored universe accidentally.
        """
        with self._lock:
            df = self.load().copy()
            symbol = str(candle.symbol).upper()
            if symbol not in df.columns:
                return False

            timestamp = normalize_timestamp(candle.open_time)
            df.loc[timestamp, symbol] = float(candle.close)
            df = df.sort_index()
            df = df.loc[~df.index.duplicated(keep="last")]
            self._atomic_write(df)
            self._data = df
            return True

    def _atomic_write(self, df: pd.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
        try:
            df.to_pickle(temp_path)
            os.replace(temp_path, self._path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
