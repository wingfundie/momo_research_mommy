from __future__ import annotations

import logging
from functools import lru_cache

from binance.client import Client

from momo_bot.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_binance_client() -> Client:
    """
    Returns a cached Binance Client instance.

    If `BINANCE_API_KEY` / `BINANCE_API_SECRET` are not set, returns an unauthenticated client.
    Public endpoints (e.g., klines/exchange_info) should still work; private endpoints (positions)
    will fail with an auth error.
    """
    requests_params = {"verify": settings.binance_verify_ssl}
    if settings.binance_api_key and settings.binance_api_secret:
        return Client(settings.binance_api_key, settings.binance_api_secret, requests_params=requests_params)
    logger.warning("BINANCE_API_KEY/BINANCE_API_SECRET not set; using unauthenticated Binance client.")
    return Client(requests_params=requests_params)
