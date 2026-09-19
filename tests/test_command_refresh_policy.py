from __future__ import annotations

import pandas as pd
from types import SimpleNamespace


def test_breakout_loader_does_not_require_complete_command_refresh(monkeypatch, tmp_path):
    import mom_break_bot

    idx = pd.DatetimeIndex(["2024-01-01"])
    frame = pd.DataFrame({"AAAUSDT": [1.0]}, index=idx)
    path = tmp_path / "breakout.pkl"
    frame.to_pickle(path)

    captured = {}

    class FakeStore:
        def refresh(self, *, updater, required_tickers=None, expected_start_by_ticker=None):
            captured["required_tickers"] = required_tickers
            updated = updater(frame, "1d", path)
            return SimpleNamespace(data=updated)

    monkeypatch.setattr(mom_break_bot, "_breakout_store", FakeStore())
    monkeypatch.setattr(
        mom_break_bot,
        "_load_latest_strategy_params",
        lambda strategy: ({"AAAUSDT": {"weights": {"x": 1.0}}}, {"data_frequency": "1d"}, "test"),
    )
    monkeypatch.setattr(
        mom_break_bot.momo,
        "update_historical_data",
        lambda df, **kwargs: (captured.setdefault("kwargs", kwargs), df)[1],
    )

    price_data, params, *_rest = mom_break_bot.load_data_breakout()

    assert list(price_data.columns) == ["AAAUSDT"]
    assert list(params) == ["AAAUSDT"]
    assert captured["required_tickers"] is None
    assert captured["kwargs"]["require_complete"] is False
    assert captured["kwargs"]["target_tickers"] == ["AAAUSDT"]
