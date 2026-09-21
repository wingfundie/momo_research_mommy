import pandas as pd
import pytest

from momo_bot.research.funding import paginate_funding_history, require_funding_coverage, settle_funding_events


def test_funding_pagination_dedupes_and_advances():
    pages = [
        [{"symbol": "BTCUSDT", "fundingTime": 1000, "fundingRate": "0.001", "markPrice": "100"},
         {"symbol": "BTCUSDT", "fundingTime": 2000, "fundingRate": "0.002", "markPrice": "110"}],
        [{"symbol": "BTCUSDT", "fundingTime": 2000, "fundingRate": "0.002", "markPrice": "110"},
         {"symbol": "BTCUSDT", "fundingTime": 3000, "fundingRate": "0.003", "markPrice": "120"}],
    ]
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        return pages.pop(0) if pages else []
    result = paginate_funding_history(fetch, "BTCUSDT", pd.Timestamp(0, unit="ms"), pd.Timestamp(4000, unit="ms"), limit=2)
    assert result["funding_time"].is_unique
    assert len(result) == 3
    assert calls[1]["startTime"] == 2001


def test_funding_signs_and_midnight_pre_rebalance_ordering():
    positions = pd.Series([2.0, -3.0], index=pd.to_datetime(["2024-01-01 16:00Z", "2024-01-02 00:00Z"]))
    funding = pd.DataFrame({"symbol": ["BTCUSDT"], "funding_time": pd.to_datetime(["2024-01-02 00:00Z"]),
                            "funding_rate": [.01], "mark_price": [100.]})
    event = settle_funding_events(funding, positions)[0]
    assert event.position_quantity == 2.0
    assert event.funding_cost_usd == 2.0
    assert event.funding_cashflow_usd == -2.0


def test_short_receives_when_rate_positive_and_missing_coverage_fails():
    positions = pd.Series([-2.0], index=pd.to_datetime(["2024-01-01T00:00:00Z"]))
    funding = pd.DataFrame({"symbol": ["ETHUSDT"], "funding_time": pd.to_datetime(["2024-01-02T00:00:00Z"]),
                            "funding_rate": [.01], "mark_price": [100.]})
    assert settle_funding_events(funding, positions)[0].funding_cashflow_usd == 2.0
    with pytest.raises(ValueError):
        require_funding_coverage(funding, ["BTCUSDT"], "2024-01-01", "2024-01-03")
