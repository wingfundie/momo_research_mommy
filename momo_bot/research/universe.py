from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import requests

from .config import UniverseConfig
from .models import AssetIdentity, PriceSegment, UniverseSnapshot


STABLE_BASES = {
    "USDT", "USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "FRAX", "PYUSD",
    "USDE", "USD1", "EUR", "EURC",
}


def eligible_usdt_perpetuals(exchange_info: Mapping) -> dict[str, AssetIdentity]:
    """Return USD-M USDT perpetual contracts, including historical/delisted rows."""
    result: dict[str, AssetIdentity] = {}
    for row in exchange_info.get("symbols", []):
        if row.get("quoteAsset") != "USDT" or row.get("contractType") != "PERPETUAL":
            continue
        symbol = str(row["symbol"])
        result[symbol] = AssetIdentity(
            binance_symbol=symbol,
            base_asset=str(row.get("baseAsset", symbol.removesuffix("USDT"))),
            contract_type="PERPETUAL",
            listing_time=_ms_datetime(row.get("onboardDate")),
            delisting_time=_ms_datetime(row.get("deliveryDate")) if row.get("status") != "TRADING" else None,
            mapping_provenance="binance_futures_exchange_info",
        )
    return result


def fetch_coingecko_fdv(session: requests.Session | None = None, *, pages: int = 4) -> pd.DataFrame:
    client = session or requests.Session()
    rows: list[dict] = []
    for page in range(1, pages + 1):
        response = client.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page},
            timeout=30,
        )
        response.raise_for_status()
        rows.extend(response.json())
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["provider_id", "symbol", "name", "fdv", "provider_timestamp"])
    return pd.DataFrame({
        "provider_id": frame["id"].astype(str),
        "symbol": frame["symbol"].astype(str).str.upper(),
        "name": frame["name"].astype(str),
        "fdv": pd.to_numeric(frame["fully_diluted_valuation"], errors="coerce"),
        "provider_timestamp": pd.to_datetime(frame["last_updated"], utc=True, errors="coerce"),
    })


def fetch_coinmarketcap_fdv(api_key: str, session: requests.Session | None = None, *, limit: int = 1000) -> pd.DataFrame:
    if not api_key:
        raise ValueError("CoinMarketCap API key is not configured")
    client = session or requests.Session()
    response = client.get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                          params={"convert": "USD", "limit": limit},
                          headers={"X-CMC_PRO_API_KEY": api_key}, timeout=30)
    response.raise_for_status()
    data = response.json().get("data", [])
    return pd.DataFrame([{"provider_id": str(row["id"]), "symbol": row["symbol"].upper(),
                          "name": row["name"], "fdv": row.get("quote", {}).get("USD", {}).get("fully_diluted_market_cap"),
                          "provider_timestamp": row.get("last_updated")} for row in data])


def fetch_fdv_cascade(*, session: requests.Session | None = None, coinmarketcap_key: str | None = None) -> tuple[str, pd.DataFrame]:
    """Try usable FDV providers in order; never silently substitute market cap for FDV."""
    errors: list[str] = []
    for name, loader in (
        ("coingecko", lambda: fetch_coingecko_fdv(session)),
        ("coinmarketcap", lambda: fetch_coinmarketcap_fdv(coinmarketcap_key or "", session)),
    ):
        try:
            frame = loader()
            frame["fdv"] = pd.to_numeric(frame["fdv"], errors="coerce")
            if frame["fdv"].notna().any():
                return name, frame
            errors.append(f"{name}: no FDV values")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("No FDV provider succeeded; " + "; ".join(errors))


def quote_volume_from_klines(klines: Mapping[str, list[list]]) -> pd.DataFrame:
    """Build a UTC daily quote-volume matrix from Binance kline payloads (field 7)."""
    series = {}
    for symbol, rows in klines.items():
        if not rows:
            continue
        index = pd.to_datetime([row[0] for row in rows], unit="ms", utc=True).normalize()
        series[symbol] = pd.Series([float(row[7]) for row in rows], index=index)
    return pd.DataFrame(series).sort_index()


def resolve_provider_assets(
    assets: Mapping[str, AssetIdentity], provider_rows: pd.DataFrame, provider: str = "coingecko"
) -> tuple[dict[str, AssetIdentity], pd.DataFrame]:
    """Resolve symbol collisions deterministically to the highest-FDV provider row."""
    required = {"provider_id", "symbol", "fdv"}
    if not required.issubset(provider_rows.columns):
        raise ValueError(f"Provider rows missing {sorted(required - set(provider_rows.columns))}")
    rows = provider_rows.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.upper()
    rows["fdv"] = pd.to_numeric(rows["fdv"], errors="coerce")
    rows = rows.sort_values(["symbol", "fdv", "provider_id"], ascending=[True, False, True])
    winners = rows.drop_duplicates("symbol", keep="first").set_index("symbol")
    resolved: dict[str, AssetIdentity] = {}
    provenance: list[dict] = []
    for symbol, asset in assets.items():
        match = winners.loc[asset.base_asset] if asset.base_asset in winners.index else None
        if match is None:
            resolved[symbol] = asset
            continue
        provider_id = str(match["provider_id"])
        collisions = int((rows["symbol"] == asset.base_asset).sum())
        resolved[symbol] = AssetIdentity(
            **{**asdict(asset), "provider": provider, "provider_id": provider_id,
               "provider_symbol": asset.base_asset, "mapping_method": "highest_fdv_symbol",
               "mapping_provenance": f"{provider}:symbol={asset.base_asset};candidates={collisions}"}
        )
        provenance.append({"binance_symbol": symbol, "provider": provider, "provider_id": provider_id,
                           "candidate_count": collisions, "fdv": float(match["fdv"])})
    return resolved, pd.DataFrame(provenance)


def trailing_median_quote_volume(quote_volume: pd.DataFrame, days: int = 30) -> pd.Series:
    return quote_volume.sort_index().tail(days).median(axis=0, skipna=True)


def build_monthly_snapshot(
    as_of: pd.Timestamp,
    assets: Mapping[str, AssetIdentity],
    provider_rows: pd.DataFrame,
    median_quote_volume: pd.Series,
    previous_members: Iterable[str] = (),
    config: UniverseConfig = UniverseConfig(),
    *,
    provider: str = "coingecko",
    point_in_time: bool = True,
) -> list[UniverseSnapshot]:
    """Apply FDV ranks and 90/110 hysteresis at a completed month boundary."""
    previous = set(previous_members)
    by_id = provider_rows.set_index("provider_id") if not provider_rows.empty else pd.DataFrame()
    candidates: list[dict] = []
    for symbol, asset in assets.items():
        provider_row = by_id.loc[asset.provider_id] if asset.provider_id and asset.provider_id in by_id.index else None
        fdv = float(provider_row["fdv"]) if provider_row is not None and pd.notna(provider_row["fdv"]) else np.nan
        volume = float(median_quote_volume.get(symbol, np.nan))
        stable = asset.base_asset.upper() in STABLE_BASES
        candidates.append({"symbol": symbol, "asset": asset, "fdv": fdv, "volume": volume, "stable": stable})
    frame = pd.DataFrame(candidates)
    rankable = frame["fdv"].notna() & (frame["volume"] >= config.min_median_quote_volume_usd)
    if config.exclude_stablecoins:
        rankable &= ~frame["stable"]
    frame["rank"] = frame["fdv"].where(rankable).rank(method="first", ascending=False).astype("Int64")
    frame = frame.sort_values(["rank", "symbol"], na_position="last")
    selected: list[str] = []
    for row in frame.itertuples():
        basic = pd.notna(row.fdv) and row.volume >= config.min_median_quote_volume_usd and not (
            config.exclude_stablecoins and row.stable
        )
        # Bootstrap the first snapshot at the requested top-N; hysteresis applies thereafter.
        threshold = config.max_members if not previous else (config.exit_rank if row.symbol in previous else config.entry_rank)
        if basic and pd.notna(row.rank) and int(row.rank) <= threshold and len(selected) < config.max_members:
            selected.append(row.symbol)
    canonical = frame[["symbol", "fdv", "volume", "rank"]].to_json(orient="records", date_format="iso")
    snapshot_hash = hashlib.sha256(canonical.encode()).hexdigest()
    output: list[UniverseSnapshot] = []
    for row in frame.itertuples():
        reasons: list[str] = []
        if pd.isna(row.fdv): reasons.append("missing_fdv")
        if row.volume < config.min_median_quote_volume_usd or pd.isna(row.volume): reasons.append("low_volume")
        if config.exclude_stablecoins and row.stable: reasons.append("stablecoin")
        output.append(UniverseSnapshot(
            as_of=pd.Timestamp(as_of).to_pydatetime(), symbol=row.symbol, provider=provider,
            fully_diluted_valuation=None if pd.isna(row.fdv) else float(row.fdv),
            rank=None if pd.isna(row.rank) else int(row.rank), eligible=not reasons,
            member=row.symbol in selected, reasons=tuple(reasons), snapshot_hash=snapshot_hash,
            point_in_time=point_in_time,
        ))
    return output


def import_snapshot(path: Path, *, source: str, expected_sha256: str) -> pd.DataFrame:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise ValueError("Imported universe snapshot checksum does not match")
    frame = pd.read_csv(path)
    frame.attrs.update(source=source, sha256=digest)
    return frame


def monthly_reconstitution_cutoff(as_of) -> pd.Timestamp:
    """Last completed UTC instant of the month preceding ``as_of``."""
    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    month_start = timestamp.to_period("M").start_time.tz_localize("UTC")
    return month_start - pd.Timedelta(nanoseconds=1)


def splice_warmup_history(
    futures: pd.Series,
    earlier: pd.Series,
    *,
    symbol: str,
    source: str,
    config: UniverseConfig = UniverseConfig(),
) -> tuple[pd.Series, pd.Series, PriceSegment]:
    """Ratio-adjust validated pre-perpetual prices; tradability remains futures-only."""
    futures = futures.dropna().sort_index().astype(float)
    earlier = earlier.dropna().sort_index().astype(float)
    if futures.empty or earlier.empty:
        raise ValueError("Both futures and earlier history are required")
    overlap = futures.index.intersection(earlier.index)
    overlap = overlap[: config.splice_overlap_days]
    if len(overlap) < config.splice_overlap_days:
        raise ValueError("Insufficient overlap for price splice")
    ratio_series = futures.loc[overlap] / earlier.loc[overlap]
    ratio = float(ratio_series.median())
    relative_error = (ratio_series / ratio - 1.0).abs().median()
    if not np.isfinite(ratio) or ratio <= 0 or relative_error > 0.10:
        raise ValueError("Price splice failed overlap validation")
    adjusted = earlier * ratio
    prelisting = adjusted.loc[adjusted.index < futures.index.min()]
    joined = pd.concat([prelisting, futures]).sort_index()
    if joined.index.to_series().diff().dt.days.max() > config.max_gap_days:
        raise ValueError("Joined history contains an excessive gap")
    tradable = pd.Series(joined.index >= futures.index.min(), index=joined.index, dtype=bool)
    segment = PriceSegment(symbol=symbol, source=source, start=prelisting.index.min().to_pydatetime(),
                           end=prelisting.index.max().to_pydatetime(), tradable=False,
                           overlap_days=len(overlap), adjustment_ratio=ratio,
                           validation_status="accepted")
    return joined, tradable, segment


def archive_provider_response(payload: object, destination: Path, *, provider: str, timestamp: pd.Timestamp) -> str:
    envelope = {"provider": provider, "retrieved_at": pd.Timestamp(timestamp).isoformat(), "payload": payload}
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _ms_datetime(value):
    if value in (None, 0, "0"):
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True).to_pydatetime()
