from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_research_report(runs: pd.DataFrame, destination: Path | str) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if runs.empty:
        body = "No completed research runs are available."
    else:
        metrics = runs.copy()
        if "metrics_json" in metrics:
            expanded = metrics["metrics_json"].map(lambda value: __import__("json").loads(value)).apply(pd.Series)
            metrics = pd.concat([metrics.drop(columns="metrics_json"), expanded], axis=1)
        best = metrics.sort_values("net_sharpe", ascending=False).iloc[0] if "net_sharpe" in metrics else metrics.iloc[0]
        body = (
            f"The strongest completed configuration was `{best.get('run_id', 'unknown')}` in the "
            f"`{best.get('sleeve', 'unknown')}` sleeve, with net Sharpe "
            f"{best.get('net_sharpe', float('nan')):.2f}. This is a research ranking, not independent evidence: "
            "selecting the best result from a broad grid materially inflates apparent performance."
        )
    text = f"""# Crypto Momentum Research Report

## Executive assessment

{body}

## Methodology

Results use causal forecasts, next-candle activation, portfolio-level accounting, event-level funding,
fees and slippage. Time-series and cross-sectional sleeves are reported independently; their equal-risk
combination is diagnostic only. Universe fallbacks that use a current constituent set must be labelled
separately from point-in-time results.

## Limitations

- Historical FDV coverage depends on archived provider snapshots or checksum-verified imports.
- Free-provider outages can reduce universe coverage and must appear in each run manifest.
- Raw net Sharpe rankings are exposed for exploration but do not correct for multiple testing.
- Shadow results cannot authorize live trading or change the production signal process.
"""
    destination.write_text(text, encoding="utf-8")
    return destination
