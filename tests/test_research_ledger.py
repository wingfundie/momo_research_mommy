from datetime import datetime, timezone

import pandas as pd

from momo_bot.research.ledger import ResearchLedger
from momo_bot.research.models import FundingEvent


def test_append_only_funding_dedupes_and_exports(tmp_path):
    event = FundingEvent("BTCUSDT", datetime.now(timezone.utc), .001, 100, 2, .2, -.2)
    with ResearchLedger(tmp_path / "ledger.sqlite") as ledger:
        ledger.append_funding([event, event])
        assert len(ledger.read("funding_events")) == 1
        ledger.export(tmp_path / "export")
    assert (tmp_path / "export" / "funding_events.csv").exists()
