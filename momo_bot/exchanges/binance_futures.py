from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from momo_bot.costs import BacktestCostConfig, CommissionRate, SymbolRules
from momo_bot.exchange import get_binance_client
from momo_bot.binance_rate_limit import get_rest_guard

logger = logging.getLogger(__name__)


class BinanceFuturesExchange:
    """
    Lightweight Binance USD-M futures metadata adapter.

    It prefers account/API values and falls back to the supplied cost config so
    research and tests can run offline.
    """

    def __init__(self, *, cost_config: BacktestCostConfig, client=None) -> None:
        self.cost_config = cost_config
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = get_binance_client()
        return self._client

    def get_commission_rate(self, symbol: str) -> CommissionRate:
        if self.cost_config.source == "binance":
            try:
                payload = get_rest_guard().call(
                    "futures_commission_rate",
                    20,
                    self.client.futures_commission_rate,
                    symbol=symbol,
                )
                if isinstance(payload, list):
                    payload = payload[0]
                return CommissionRate(
                    symbol=symbol,
                    maker=float(payload["makerCommissionRate"]),
                    taker=float(payload["takerCommissionRate"]),
                    source="binance_api",
                )
            except Exception as exc:
                logger.warning("Falling back to configured futures fees for %s: %s", symbol, exc)

        return CommissionRate(
            symbol=symbol,
            maker=float(self.cost_config.maker_fee),
            taker=float(self.cost_config.taker_fee),
            source=self.cost_config.source if self.cost_config.source != "binance" else "fallback",
        )

    @lru_cache(maxsize=1)
    def _exchange_info(self) -> dict:
        return get_rest_guard().call(
            "futures_exchange_info",
            1,
            self.client.futures_exchange_info,
        )

    def get_symbol_rules(self, symbol: str) -> SymbolRules:
        try:
            info = self._exchange_info()
            symbols = info.get("symbols", [])
            item = next((x for x in symbols if x.get("symbol") == symbol), None)
            if not item:
                return SymbolRules(symbol=symbol)

            filters = {f.get("filterType"): f for f in item.get("filters", [])}
            price_filter = filters.get("PRICE_FILTER", {})
            lot_filter = filters.get("LOT_SIZE", {})
            min_notional_filter = filters.get("MIN_NOTIONAL", {})
            return SymbolRules(
                symbol=symbol,
                tick_size=_safe_float(price_filter.get("tickSize")),
                step_size=_safe_float(lot_filter.get("stepSize")),
                min_qty=_safe_float(lot_filter.get("minQty")),
                min_notional=_safe_float(
                    min_notional_filter.get("notional") or min_notional_filter.get("minNotional")
                ),
            )
        except Exception as exc:
            logger.warning("Could not load Binance symbol rules for %s: %s", symbol, exc)
            return SymbolRules(symbol=symbol)

    def get_funding_rates(self, symbol: str, start, end) -> pd.Series:
        if not self.cost_config.include_funding or self.cost_config.funding_mode == "zero":
            return pd.Series(dtype=float)
        if self.cost_config.funding_mode != "historical":
            return pd.Series(dtype=float)

        try:
            start_ms = int(pd.Timestamp(start).timestamp() * 1000)
            end_ms = int(pd.Timestamp(end).timestamp() * 1000)
            payload = get_rest_guard().call(
                "futures_funding_rate",
                1,
                self.client.futures_funding_rate,
                symbol=symbol,
                startTime=start_ms,
                endTime=end_ms,
            )
            if not payload:
                return pd.Series(dtype=float)
            rows = pd.DataFrame(payload)
            rows["fundingTime"] = pd.to_datetime(rows["fundingTime"], unit="ms")
            rows["fundingRate"] = rows["fundingRate"].astype(float)
            return rows.set_index("fundingTime")["fundingRate"].sort_index()
        except Exception as exc:
            logger.warning("Could not load Binance funding rates for %s: %s", symbol, exc)
            return pd.Series(dtype=float)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
