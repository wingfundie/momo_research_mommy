import hashlib
from pathlib import Path

import pandas as pd
import pytest

from momo_bot.research.config import UniverseConfig
from momo_bot.research.models import AssetIdentity
from momo_bot.research.universe import build_monthly_snapshot, import_snapshot, resolve_provider_assets, splice_warmup_history


def test_collision_chooses_highest_fdv_and_hysteresis_retains_incumbent():
    assets = {"ABCUSDT": AssetIdentity("ABCUSDT", "ABC")}
    providers = pd.DataFrame([{"provider_id": "low", "symbol": "ABC", "fdv": 10},
                              {"provider_id": "high", "symbol": "ABC", "fdv": 100}])
    resolved, provenance = resolve_provider_assets(assets, providers)
    assert resolved["ABCUSDT"].provider_id == "high"
    assert provenance.iloc[0]["candidate_count"] == 2


def test_universe_filters_stablecoin_and_low_volume():
    assets = {"ABCUSDT": AssetIdentity("ABCUSDT", "ABC", provider_id="abc"),
              "USDCUSDT": AssetIdentity("USDCUSDT", "USDC", provider_id="usdc")}
    providers = pd.DataFrame([{"provider_id": "abc", "symbol": "ABC", "fdv": 100},
                              {"provider_id": "usdc", "symbol": "USDC", "fdv": 200}])
    result = build_monthly_snapshot(pd.Timestamp("2024-02-01"), assets, providers,
                                    pd.Series({"ABCUSDT": 2e6, "USDCUSDT": 2e6}))
    by_symbol = {r.symbol: r for r in result}
    assert by_symbol["ABCUSDT"].member
    assert not by_symbol["USDCUSDT"].member
    assert "stablecoin" in by_symbol["USDCUSDT"].reasons


def test_spliced_warmup_never_becomes_tradable():
    earlier_index = pd.date_range("2024-01-01", periods=20, tz="UTC")
    futures_index = pd.date_range("2024-01-14", periods=20, tz="UTC")
    earlier = pd.Series(range(100, 120), index=earlier_index, dtype=float)
    futures = earlier.reindex(futures_index).combine_first(pd.Series(range(113, 133), index=futures_index)) * 2
    joined, tradable, segment = splice_warmup_history(futures, earlier, symbol="ABCUSDT", source="spot")
    assert not tradable.loc[: futures.index.min() - pd.Timedelta(days=1)].any()
    assert tradable.loc[futures.index.min():].all()
    assert segment.adjustment_ratio == pytest.approx(2)


def test_import_requires_checksum(tmp_path: Path):
    path = tmp_path / "snapshot.csv"; path.write_text("symbol,fdv\nBTC,1\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert import_snapshot(path, source="vendor", expected_sha256=digest).attrs["source"] == "vendor"
    with pytest.raises(ValueError): import_snapshot(path, source="vendor", expected_sha256="bad")
