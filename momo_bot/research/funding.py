from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .models import ActualFundingRecord, FundingEvent


def request_income_archive(signed_request: Callable[..., dict], start, end) -> str:
    """Request Binance's asynchronous USD-M transaction-history archive."""
    payload = signed_request(
        "GET", "/fapi/v1/income/asyn",
        params={"startTime": int(pd.Timestamp(start).timestamp() * 1000),
                "endTime": int(pd.Timestamp(end).timestamp() * 1000)},
    )
    download_id = payload.get("downloadId")
    if not download_id:
        raise RuntimeError(f"Binance did not return a transaction download id: {payload}")
    return str(download_id)


def income_archive_status(signed_request: Callable[..., dict], download_id: str) -> dict:
    """Return asynchronous archive status/link; the caller controls polling cadence."""
    return signed_request("GET", "/fapi/v1/income/asyn/id", params={"downloadId": download_id})


def fetch_commission_rate(fetch: Callable[..., dict], symbol: str) -> dict[str, float | str]:
    payload = fetch(symbol=symbol)
    if isinstance(payload, list):
        payload = payload[0]
    return {"symbol": symbol, "maker": float(payload["makerCommissionRate"]),
            "taker": float(payload["takerCommissionRate"]), "source": "binance_api"}


def paginate_funding_history(
    fetch_page: Callable[..., list[dict]], symbol: str, start, end, *, limit: int = 1000
) -> pd.DataFrame:
    """Fetch every Binance funding event without timestamp duplication."""
    cursor = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    rows: list[dict] = []
    while cursor <= end_ms:
        page = fetch_page(symbol=symbol, startTime=cursor, endTime=end_ms, limit=limit) or []
        if not page:
            break
        rows.extend(page)
        latest = max(int(item["fundingTime"]) for item in page)
        if latest < cursor:
            raise RuntimeError("Funding pagination did not advance")
        cursor = latest + 1
        if len(page) < limit:
            break
    if not rows:
        return pd.DataFrame(columns=["symbol", "funding_time", "funding_rate", "mark_price"])
    result = pd.DataFrame(rows)
    result["symbol"] = result.get("symbol", symbol).fillna(symbol)
    result["funding_time"] = pd.to_datetime(result["fundingTime"], unit="ms", utc=True)
    result["funding_rate"] = pd.to_numeric(result["fundingRate"], errors="raise")
    result["mark_price"] = pd.to_numeric(result.get("markPrice"), errors="coerce")
    return result[["symbol", "funding_time", "funding_rate", "mark_price"]].drop_duplicates(
        ["symbol", "funding_time"], keep="last"
    ).sort_values("funding_time").reset_index(drop=True)


def settle_funding_events(
    funding: pd.DataFrame,
    positions: pd.Series,
    *,
    strict: bool = True,
) -> list[FundingEvent]:
    """Settle each event against the last position strictly before its timestamp."""
    positions = positions.sort_index().astype(float)
    pos_times = pd.DatetimeIndex(positions.index)
    events: list[FundingEvent] = []
    for row in funding.sort_values("funding_time").itertuples():
        time = pd.Timestamp(row.funding_time)
        location = pos_times.searchsorted(time, side="left") - 1
        if location < 0:
            if strict:
                raise ValueError(f"No pre-event position for {row.symbol} at {time}")
            quantity = 0.0
        else:
            quantity = float(positions.iloc[location])
        if not np.isfinite(row.mark_price):
            raise ValueError(f"Missing funding mark price for {row.symbol} at {time}")
        cost = quantity * float(row.mark_price) * float(row.funding_rate)
        events.append(FundingEvent(
            symbol=row.symbol, funding_time=time.to_pydatetime(), funding_rate=float(row.funding_rate),
            mark_price=float(row.mark_price), position_quantity=quantity,
            funding_cost_usd=cost, funding_cashflow_usd=-cost,
        ))
    return events


def require_funding_coverage(funding: pd.DataFrame, required_symbols: Iterable[str], start, end) -> None:
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    event_times = pd.to_datetime(funding["funding_time"], utc=True)
    missing = []
    for symbol in sorted(set(required_symbols)):
        subset = funding[(funding["symbol"] == symbol) & event_times.between(start, end)]
        if subset.empty:
            missing.append(symbol)
    if missing:
        raise ValueError(f"Missing required funding history for: {', '.join(missing)}")


def paginate_income_history(
    fetch_page: Callable[..., list[dict]], start, end, *, limit: int = 1000
) -> pd.DataFrame:
    cursor = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    rows: list[dict] = []
    while cursor <= end_ms:
        page = fetch_page(incomeType="FUNDING_FEE", startTime=cursor, endTime=end_ms, limit=limit) or []
        if not page:
            break
        rows.extend(page)
        latest = max(int(item["time"]) for item in page)
        cursor = latest + 1
        if len(page) < limit:
            break
    if not rows:
        return pd.DataFrame(columns=["transaction_id", "symbol", "asset", "timestamp", "income_usd"])
    result = pd.DataFrame(rows)
    result["transaction_id"] = result["tranId"].astype(str)
    result["timestamp"] = pd.to_datetime(result["time"], unit="ms", utc=True)
    result["income_usd"] = pd.to_numeric(result["income"], errors="raise")
    return result[["transaction_id", "symbol", "asset", "timestamp", "income_usd"]].drop_duplicates(
        "transaction_id", keep="last"
    ).sort_values("timestamp")


def import_income_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {"tranId": "transaction_id", "time": "timestamp", "income": "income_usd"}
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    required = {"transaction_id", "symbol", "asset", "timestamp", "income_usd"}
    if not required.issubset(frame):
        raise ValueError(f"Income CSV missing {sorted(required - set(frame.columns))}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["income_usd"] = pd.to_numeric(frame["income_usd"], errors="raise")
    return frame[list(required)].drop_duplicates("transaction_id")


def reconcile_actual_funding(
    actual: pd.DataFrame, model_events: Iterable[FundingEvent], actual_notional: pd.Series | None = None
) -> list[ActualFundingRecord]:
    modeled = pd.DataFrame([asdict(event) for event in model_events])
    output: list[ActualFundingRecord] = []
    for row in actual.itertuples():
        candidates = modeled[(modeled["symbol"] == row.symbol) &
                             (pd.to_datetime(modeled["funding_time"], utc=True) == pd.Timestamp(row.timestamp))]
        candidate = candidates.iloc[0] if not candidates.empty else None
        model_notional = 0.0 if candidate is None else abs(float(candidate["position_quantity"] * candidate["mark_price"]))
        account_notional = model_notional if actual_notional is None else abs(float(actual_notional.get(row.transaction_id, 0.0)))
        sign_agrees = candidate is not None and np.sign(float(row.income_usd)) == np.sign(float(candidate["funding_cashflow_usd"]))
        matched = min(model_notional, account_notional) if sign_agrees else 0.0
        output.append(ActualFundingRecord(
            transaction_id=str(row.transaction_id), symbol=row.symbol, asset=row.asset,
            timestamp=pd.Timestamp(row.timestamp).to_pydatetime(), income_usd=float(row.income_usd),
            matched_model="crypto_momentum" if matched else None, matched_notional=matched,
            unmatched_notional=max(account_notional - matched, 0.0),
            unmatched_reason=None if matched else ("direction_mismatch" if candidate is not None else "no_model_event"),
        ))
    return output


def expected_funding(position_notional: float, current_rate: float, settlements_next_24h: int = 3) -> tuple[float, float]:
    next_cashflow = -float(position_notional) * float(current_rate)
    return next_cashflow, next_cashflow * int(settlements_next_24h)
