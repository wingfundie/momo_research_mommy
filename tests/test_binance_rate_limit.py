from __future__ import annotations

import json

import pytest


def test_rest_guard_persists_cooldown_and_blocks_followup(tmp_path):
    from momo_bot.binance_rate_limit import BinanceRestCircuitOpen, BinanceRestGuard

    class RateLimitError(Exception):
        status_code = 429

    state_path = tmp_path / "binance_rest_state.json"
    guard = BinanceRestGuard(
        state_path=state_path,
        weight_per_minute=60,
        cooldown_buffer_seconds=0,
    )

    with pytest.raises(BinanceRestCircuitOpen):
        guard.call("test_endpoint", 1, lambda: (_ for _ in ()).throw(RateLimitError("Too many requests")))

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["cooldown_until_ms"] > state["recorded_at_ms"]

    called = False

    def should_not_run():
        nonlocal called
        called = True

    with pytest.raises(BinanceRestCircuitOpen):
        guard.call("test_endpoint", 1, should_not_run)

    assert called is False


def test_parse_banned_until_ms():
    from momo_bot.binance_rate_limit import parse_banned_until_ms

    assert parse_banned_until_ms("IP banned until 1782749039914.") == 1782749039914
    assert parse_banned_until_ms("Too many requests") is None


def test_kline_weight_matches_binance_tiers():
    from momo_bot.binance_rate_limit import kline_weight

    assert kline_weight(99) == 1
    assert kline_weight(100) == 2
    assert kline_weight(500) == 5
    assert kline_weight(1001) == 10
