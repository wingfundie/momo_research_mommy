from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from momo_bot.research.config import ResearchConfig
from momo_bot.research.ledger import ResearchLedger
from momo_bot.research.reporting import write_research_report
from momo_bot.research.runner import run_research


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet": return pd.read_parquet(path)
    if path.suffix in {".pkl", ".pickle"}: return pd.read_pickle(path)
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal crypto momentum research in shadow mode")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--funding-cashflows", type=Path)
    parser.add_argument("--funding-mode", choices=["historical", "zero", "estimated"], default="historical")
    parser.add_argument("--sleeve", choices=["time_series", "cross_sectional_continuous", "cross_sectional_basket", "cross_sectional_long_only"], default="time_series")
    parser.add_argument("--ledger", type=Path, default=Path("data_store/crypto_momentum_research.sqlite"))
    args = parser.parse_args()
    prices = load_frame(args.prices)
    funding = None
    if args.funding_cashflows:
        funding = load_frame(args.funding_cashflows).squeeze("columns")
    config = replace(ResearchConfig(), ledger_path=args.ledger)
    with ResearchLedger(args.ledger) as ledger:
        run, result, _ = run_research(prices, config, sleeve=args.sleeve,
                                      funding_cashflows_usd=funding, funding_mode=args.funding_mode, ledger=ledger)
        ledger.export(config.output_dir)
        report = write_research_report(ledger.read("portfolio_runs"), config.output_dir / "RESEARCH_REPORT.md")
    print(f"Completed {run.run_id}: net Sharpe {result.metrics['net_sharpe']:.3f}")
    print(report)


if __name__ == "__main__": main()
