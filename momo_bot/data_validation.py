from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from momo_bot.candles import expected_last_complete_open_time, normalize_dt_index, normalize_timestamp


@dataclass(frozen=True)
class PriceFrameValidation:
    path: Path
    configured_frequency: Optional[str]
    inferred_frequency: str
    effective_frequency: str
    is_frequency_match: bool
    rows: int
    columns: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    duplicate_index_count: int
    missing_cell_pct: float
    warnings: list[str]


@dataclass(frozen=True)
class PriceCoverageValidation:
    required_timestamp: pd.Timestamp
    required_ticker_count: int
    present_ticker_count: int
    fresh_ticker_count: int
    missing_latest_tickers: list[str]
    missing_internal_cells: int
    unresolved_internal_cells: int
    confirmed_unavailable_cells: int
    tickers_with_internal_gaps: dict[str, int]
    duplicate_index_count: int
    is_publishable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "required_timestamp": self.required_timestamp.isoformat(),
            "required_ticker_count": self.required_ticker_count,
            "present_ticker_count": self.present_ticker_count,
            "fresh_ticker_count": self.fresh_ticker_count,
            "missing_latest_tickers": self.missing_latest_tickers,
            "missing_internal_cells": self.missing_internal_cells,
            "unresolved_internal_cells": self.unresolved_internal_cells,
            "confirmed_unavailable_cells": self.confirmed_unavailable_cells,
            "tickers_with_internal_gaps": self.tickers_with_internal_gaps,
            "duplicate_index_count": self.duplicate_index_count,
            "is_publishable": self.is_publishable,
        }


def infer_frequency(index: pd.DatetimeIndex) -> str:
    if index is None or len(index) < 2:
        return "unknown"

    idx = index.sort_values()
    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return "unknown"

    mode_delta = deltas.mode().iloc[0]
    if mode_delta == pd.Timedelta(hours=4):
        return "4h"
    if mode_delta == pd.Timedelta(hours=1):
        return "1h"
    if mode_delta == pd.Timedelta(days=1):
        return "1d"
    return f"irregular:{mode_delta}"


def resolve_effective_frequency(
    *,
    configured_frequency: Optional[str],
    inferred_frequency: str,
    strict: bool,
) -> tuple[str, bool, list[str]]:
    configured = (configured_frequency or "auto").lower()
    warnings: list[str] = []

    if configured == "auto":
        if inferred_frequency == "unknown" or inferred_frequency.startswith("irregular:"):
            warnings.append(f"Could not infer a regular data frequency ({inferred_frequency}).")
        return inferred_frequency, True, warnings

    is_match = configured == inferred_frequency
    if not is_match:
        message = f"Configured frequency {configured!r} does not match inferred frequency {inferred_frequency!r}."
        if strict:
            raise ValueError(message)
        warnings.append(message)
    return configured, is_match, warnings


def validate_price_frame(
    df: pd.DataFrame,
    *,
    path: Path,
    configured_frequency: Optional[str] = "auto",
    strict: bool = False,
) -> PriceFrameValidation:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Price data must be a pandas DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Price data index must be a DatetimeIndex.")

    inferred = infer_frequency(df.index)
    effective, is_match, warnings = resolve_effective_frequency(
        configured_frequency=configured_frequency,
        inferred_frequency=inferred,
        strict=strict,
    )

    duplicate_count = int(df.index.duplicated().sum())
    if duplicate_count:
        warnings.append(f"Price data has {duplicate_count} duplicate timestamps.")
    if not df.index.is_monotonic_increasing:
        warnings.append("Price data index is not sorted ascending.")

    missing_pct = float(df.isna().mean().mean() * 100.0) if df.size else 0.0

    return PriceFrameValidation(
        path=Path(path),
        configured_frequency=configured_frequency,
        inferred_frequency=inferred,
        effective_frequency=effective,
        is_frequency_match=is_match,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        start=df.index.min() if len(df.index) else None,
        end=df.index.max() if len(df.index) else None,
        duplicate_index_count=duplicate_count,
        missing_cell_pct=missing_pct,
        warnings=warnings,
    )


def resolve_coverage_starts(
    df: pd.DataFrame,
    *,
    required_tickers: list[str],
    onboard_date_by_ticker: Mapping[str, pd.Timestamp] | None = None,
) -> dict[str, pd.Timestamp]:
    starts: dict[str, pd.Timestamp] = {}
    for ticker in sorted(set(required_tickers)):
        onboard_date = None
        if onboard_date_by_ticker and ticker in onboard_date_by_ticker:
            onboard_date = normalize_timestamp(onboard_date_by_ticker[ticker])

        valid = df[ticker].dropna() if ticker in df.columns else pd.Series(dtype=float)
        if onboard_date is not None and not valid.empty:
            valid = valid.loc[valid.index >= onboard_date]
        if not valid.empty:
            starts[ticker] = normalize_timestamp(valid.index.min())
        elif onboard_date is not None:
            starts[ticker] = onboard_date
    return starts


def missing_coverage_dates(
    df: pd.DataFrame,
    *,
    required_tickers: list[str],
    freq: str,
    required_timestamp: pd.Timestamp,
    expected_start_by_ticker: Mapping[str, pd.Timestamp],
) -> dict[str, list[pd.Timestamp]]:
    required = normalize_timestamp(required_timestamp)
    expected_delta = pd.Timedelta(freq)
    missing: dict[str, list[pd.Timestamp]] = {}
    for ticker in sorted(set(required_tickers)):
        if ticker not in df.columns or ticker not in expected_start_by_ticker:
            continue
        start = normalize_timestamp(expected_start_by_ticker[ticker])
        expected_index = pd.date_range(start, required, freq=expected_delta)
        valid_index = pd.DatetimeIndex(df[ticker].dropna().loc[:required].index)
        ticker_missing = expected_index.difference(valid_index)
        if len(ticker_missing):
            missing[ticker] = [normalize_timestamp(timestamp) for timestamp in ticker_missing]
    return missing


def validate_price_coverage(
    df: pd.DataFrame,
    *,
    required_tickers: list[str],
    freq: str,
    required_timestamp: pd.Timestamp | None = None,
    expected_start_by_ticker: Mapping[str, pd.Timestamp] | None = None,
    allowed_missing_by_ticker: Mapping[str, Iterable[pd.Timestamp]] | None = None,
) -> PriceCoverageValidation:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Price data must be a pandas DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Price data index must be a DatetimeIndex.")

    required = normalize_timestamp(
        required_timestamp or expected_last_complete_open_time(freq=freq)
    )
    ordered = df.copy()
    ordered.index = normalize_dt_index(ordered.index)
    ordered = ordered.sort_index()
    duplicate_count = int(ordered.index.duplicated().sum())

    tickers = sorted(set(required_tickers))
    present = [ticker for ticker in tickers if ticker in ordered.columns]
    missing_latest: list[str] = []
    gaps: dict[str, int] = {}
    unresolved_gap_count = 0
    confirmed_gap_count = 0
    fresh_count = 0
    expected_delta = pd.Timedelta(freq)

    for ticker in tickers:
        if ticker not in ordered.columns:
            missing_latest.append(ticker)
            continue

        series = ordered[ticker]
        valid = series.dropna()
        if valid.empty:
            missing_latest.append(ticker)
            continue

        first_valid = normalize_timestamp(valid.index.min())
        expected_start = first_valid
        if expected_start_by_ticker and ticker in expected_start_by_ticker:
            expected_start = normalize_timestamp(expected_start_by_ticker[ticker])
        valid_through_required = valid.loc[valid.index <= required]
        if valid_through_required.empty or normalize_timestamp(valid_through_required.index.max()) < required:
            missing_latest.append(ticker)
        else:
            fresh_count += 1

        expected_index = pd.date_range(expected_start, required, freq=expected_delta)
        actual_index = pd.DatetimeIndex(valid_through_required.index)
        missing_count = len(expected_index.difference(actual_index))
        if missing_count:
            gaps[ticker] = int(missing_count)
            allowed = {
                normalize_timestamp(timestamp)
                for timestamp in (allowed_missing_by_ticker or {}).get(ticker, [])
            }
            missing_dates = set(expected_index.difference(actual_index))
            confirmed_gap_count += len(missing_dates & allowed)
            unresolved_gap_count += len(missing_dates - allowed)

    missing_internal_cells = int(sum(gaps.values()))
    publishable = (
        bool(tickers)
        and len(present) == len(tickers)
        and not missing_latest
        and unresolved_gap_count == 0
        and duplicate_count == 0
    )
    return PriceCoverageValidation(
        required_timestamp=required,
        required_ticker_count=len(tickers),
        present_ticker_count=len(present),
        fresh_ticker_count=fresh_count,
        missing_latest_tickers=missing_latest,
        missing_internal_cells=missing_internal_cells,
        unresolved_internal_cells=int(unresolved_gap_count),
        confirmed_unavailable_cells=int(confirmed_gap_count),
        tickers_with_internal_gaps=gaps,
        duplicate_index_count=duplicate_count,
        is_publishable=publishable,
    )
