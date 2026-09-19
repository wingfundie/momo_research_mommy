from __future__ import annotations

import datetime as dt
from datetime import timezone
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from momo_bot.config import settings
from momo_bot.exchange import get_binance_client
from momo_bot.binance_rate_limit import (
    BinanceRestCircuitOpen,
    get_rest_guard,
    kline_weight,
)


def get_utc_string(dt_object: dt.datetime) -> str:
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def get_utc(date: str) -> str:
    date_obj = dt.datetime.strptime(date, "%Y-%m-%d")
    date_utc = date_obj.astimezone(timezone.utc)
    return date_utc.strftime("%Y-%m-%d %H:%M:%S %Z")


def gen_lookback_df(
    ticker: str,
    start_date: str,
    end_date: str,
    freq: str = "4h",
    *,
    client: Optional[Client] = None,
) -> pd.DataFrame:
    headers = [
        "time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]
    cl = client or get_binance_client()
    historicals = get_rest_guard().call(
        "futures_historical_klines",
        10,
        lambda: cl.futures_historical_klines(
            ticker, freq, start_str=get_utc(start_date), end_str=get_utc(end_date)
        ),
    )
    ticker_df = pd.DataFrame(historicals, columns=headers)
    ticker_df["time"] = pd.to_datetime(ticker_df["time"], unit="ms")
    ticker_df = ticker_df.set_index("time")
    ticker_df = ticker_df.astype("float")
    return ticker_df


def get_fresh_lookback_df(
    ticker: str,
    start_date: str | dt.datetime,
    freq: str = "4h",
    *,
    client: Optional[Client] = None,
) -> pd.DataFrame:
    """Fetch a continuous futures-kline range using explicit API pagination."""
    headers = [
        "time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]
    cl = client or get_binance_client()
    start_ts = pd.Timestamp(start_date)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    end_ts = pd.Timestamp.now(tz="UTC")
    next_start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    rows: list[list[object]] = []

    while next_start_ms <= end_ms:
        page = get_rest_guard().call(
            "futures_klines",
            kline_weight(1000),
            cl.futures_klines,
            symbol=ticker,
            interval=freq,
            startTime=next_start_ms,
            endTime=end_ms,
            limit=1000,
        )
        if not page:
            break
        rows.extend(page)
        last_open_ms = int(page[-1][0])
        if last_open_ms < next_start_ms:
            raise RuntimeError(f"Binance kline pagination did not advance for {ticker}.")
        next_start_ms = last_open_ms + 1
        if len(page) < 1000:
            break

    full_df = pd.DataFrame(rows, columns=headers)
    if full_df.empty:
        return full_df
    full_df["time"] = pd.to_datetime(full_df["time"], unit="ms")
    full_df = full_df.set_index("time")
    full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()
    full_df = full_df.drop(columns=["Ignore"])
    return full_df.astype("float")


def get_usdt_perpetual_futures_onboard_dates(
    api_key: Optional[str] = None, api_secret: Optional[str] = None
) -> dict[str, pd.Timestamp]:
    try:
        requests_params = {"verify": settings.binance_verify_ssl}
        if api_key and api_secret:
            cl = Client(api_key, api_secret, requests_params=requests_params)
        else:
            cl = Client(requests_params=requests_params)

        exchange_info = get_rest_guard().call(
            "futures_exchange_info",
            1,
            cl.futures_exchange_info,
        )
        perpetual_tickers: dict[str, pd.Timestamp] = {}
        for item in exchange_info["symbols"]:
            if (
                item.get("contractType") == "PERPETUAL"
                and item.get("status") == "TRADING"
                and item.get("quoteAsset") == "USDT"
            ):
                onboard_date = item.get("onboardDate")
                if onboard_date is None:
                    continue
                perpetual_tickers[item["symbol"]] = pd.to_datetime(
                    onboard_date, unit="ms", utc=True
                ).tz_localize(None).floor("D")
        return dict(sorted(perpetual_tickers.items()))
    except BinanceRestCircuitOpen:
        raise
    except BinanceAPIException:
        return {}
    except Exception:
        return {}


def get_all_usdt_perpetual_futures_tickers(
    api_key: Optional[str] = None, api_secret: Optional[str] = None
) -> list[str]:
    onboard_dates = get_usdt_perpetual_futures_onboard_dates(
        api_key=api_key,
        api_secret=api_secret,
    )
    if not onboard_dates:
        return []
    return list(onboard_dates)
