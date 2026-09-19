from __future__ import annotations

import pandas as pd


def portfolio_summary(signals: pd.DataFrame, funding: pd.DataFrame) -> str:
    longs = signals[signals["direction"] == "LONG"].nlargest(5, "combined_forecast")
    shorts = signals[signals["direction"] == "SHORT"].nsmallest(5, "combined_forecast")
    paid = -funding.loc[funding["funding_cashflow_usd"] < 0, "funding_cashflow_usd"].sum() if not funding.empty else 0.0
    received = funding.loc[funding["funding_cashflow_usd"] > 0, "funding_cashflow_usd"].sum() if not funding.empty else 0.0
    format_side = lambda frame: ", ".join(f"{r.symbol} {r.combined_forecast:+.1f}" for r in frame.itertuples()) or "none"
    return (f"Crypto momentum shadow\nLong strength: {format_side(longs)}\nShort strength: {format_side(shorts)}\n"
            f"Funding paid ${paid:,.2f} | received ${received:,.2f} | net ${received-paid:,.2f}")
