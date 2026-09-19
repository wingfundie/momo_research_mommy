from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar

from momo_bot.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BAN_UNTIL_RE = re.compile(r"banned\s+until\s+(\d{12,})", re.IGNORECASE)


class BinanceRestCircuitOpen(RuntimeError):
    """Raised when Binance REST access is disabled by a persisted cooldown."""


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def parse_banned_until_ms(message: str) -> int | None:
    match = _BAN_UNTIL_RE.search(message or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def is_rate_limit_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    text = str(exc)
    return (
        code == -1003
        or status_code in {418, 429}
        or "Too many requests" in text
        or "Way too many requests" in text
        or "banned until" in text.lower()
    )


def _is_binance_api_exception(exc: BaseException) -> bool:
    try:
        from binance.exceptions import BinanceAPIException
    except Exception:
        return False
    return isinstance(exc, BinanceAPIException)


def kline_weight(limit: int) -> int:
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


class BinanceRestGuard:
    def __init__(
        self,
        *,
        state_path: Path,
        weight_per_minute: int,
        cooldown_buffer_seconds: int,
    ) -> None:
        self.state_path = state_path
        self.weight_per_minute = max(1, int(weight_per_minute))
        self.cooldown_buffer_ms = max(0, int(cooldown_buffer_seconds)) * 1000
        self._capacity = float(self.weight_per_minute)
        self._tokens = float(self.weight_per_minute)
        self._rate_per_second = float(self.weight_per_minute) / 60.0
        self._last_refill = time.monotonic()
        self._lock = threading.RLock()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            logger.warning("Could not read Binance REST state file: %s", self.state_path)
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
        temp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, self.state_path)

    def assert_not_banned(self) -> None:
        state = self._load_state()
        until_ms = state.get("cooldown_until_ms")
        if until_ms is None:
            return
        try:
            until_ms_int = int(until_ms)
        except (TypeError, ValueError):
            return
        if utc_now_ms() < until_ms_int:
            raise BinanceRestCircuitOpen(
                f"Binance REST disabled until {until_ms_int} because a prior rate-limit ban was recorded."
            )

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_refill)
        if elapsed:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
            self._last_refill = now

    def acquire(self, weight: int) -> None:
        requested = max(1, int(weight))
        while True:
            with self._lock:
                self.assert_not_banned()
                self._refill_locked()
                if self._tokens >= requested:
                    self._tokens -= requested
                    return
                missing = requested - self._tokens
                sleep_seconds = missing / self._rate_per_second
            time.sleep(max(0.05, sleep_seconds))

    def record_rate_limit(self, exc: BaseException, *, endpoint: str | None = None) -> None:
        message = str(exc)
        until_ms = parse_banned_until_ms(message)
        if until_ms is None:
            until_ms = utc_now_ms() + 60 * 60 * 1000
        until_ms += self.cooldown_buffer_ms
        state = {
            "cooldown_until_ms": until_ms,
            "recorded_at_ms": utc_now_ms(),
            "endpoint": endpoint,
            "reason": message,
        }
        with self._lock:
            self._write_state(state)
        logger.error("Binance REST circuit opened until %s after %s: %s", until_ms, endpoint, message)

    def call(self, endpoint: str, weight: int, fn: Callable[..., T], **kwargs: Any) -> T:
        self.acquire(weight)
        try:
            return fn(**kwargs)
        except Exception as exc:
            if is_rate_limit_error(exc):
                self.record_rate_limit(exc, endpoint=endpoint)
                raise BinanceRestCircuitOpen(str(exc)) from exc
            if _is_binance_api_exception(exc):
                raise
            raise


@lru_cache(maxsize=1)
def get_rest_guard() -> BinanceRestGuard:
    return BinanceRestGuard(
        state_path=settings.data_dir / "binance_rest_state.json",
        weight_per_minute=settings.binance_rest_weight_per_minute,
        cooldown_buffer_seconds=settings.binance_rest_cooldown_buffer_seconds,
    )
