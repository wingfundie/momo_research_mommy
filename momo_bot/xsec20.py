from __future__ import annotations

import json
import math
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from momo_bot.binance_data import get_fresh_lookback_df
from momo_bot.config import settings


MODEL = "xsm_ic20_dollar_neutral_vol60"
CONFIG_ID = "cc09fe90ed5150b8a67f"
TARGET_VOL = 0.15
GROSS_CAP = 2.0
TICKER_RISK_CAP = 0.25
VOLATILITY_WINDOW = 60
PORTFOLIO_VALUE = 100_000.0
TAKER_SHARE = 1.0
SLIPPAGE_BPS = 5.0
SNAPSHOT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class XSecSnapshot:
    manifest: dict[str, Any]
    ticker_history: pd.DataFrame
    portfolio: pd.DataFrame
    standalone_sr: pd.Series

    @property
    def data_cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.manifest["data_cutoff"])

    @property
    def latest(self) -> pd.DataFrame:
        if self.ticker_history.empty:
            return pd.DataFrame()
        stamp = self.ticker_history.index.get_level_values("timestamp").max()
        return self.ticker_history.xs(stamp, level="timestamp").copy()

    def ticker(self, symbol: str) -> pd.DataFrame:
        canonical = normalize_symbol(symbol)
        if canonical not in self.ticker_history.index.get_level_values("symbol"):
            raise KeyError(f"No XSec20 history is available for {canonical}")
        return self.ticker_history.xs(canonical, level="symbol").copy()


def normalize_symbol(value: str) -> str:
    symbol = value.strip().upper().replace("/", "").replace("-", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    return symbol


def round_quantity(quantity: float, *, step_size: float | None, min_qty: float | None = None) -> float:
    if not np.isfinite(quantity):
        return 0.0
    if step_size and step_size > 0:
        rounded = math.floor(abs(quantity) / step_size + 1e-12) * step_size
        quantity = math.copysign(rounded, quantity)
        decimals = max(0, -int(math.floor(math.log10(step_size)))) if step_size < 1 else 0
        quantity = round(quantity, decimals + 1)
    if min_qty and abs(quantity) < min_qty:
        return 0.0
    return float(quantity)


def annual_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "cumulative_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    growth = (1.0 + clean).cumprod()
    annual_return = float(clean.mean() * 365)
    annual_volatility = float(clean.std(ddof=1) * np.sqrt(365)) if len(clean) > 1 else np.nan
    return {
        "cumulative_return": float(growth.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": annual_return / annual_volatility if annual_volatility and np.isfinite(annual_volatility) else np.nan,
        "max_drawdown": float((growth / growth.cummax() - 1.0).min()),
    }


def compare_returns(portfolio: pd.Series, benchmark: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    aligned = pd.concat(
        [portfolio.rename("portfolio"), benchmark.rename("benchmark")], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        raise ValueError("Portfolio and benchmark have no overlapping observations")
    portfolio_metrics = annual_metrics(aligned["portfolio"])
    benchmark_metrics = annual_metrics(aligned["benchmark"])
    benchmark_variance = float(aligned["benchmark"].var(ddof=1))
    beta = (
        float(aligned["portfolio"].cov(aligned["benchmark"]) / benchmark_variance)
        if benchmark_variance > 0
        else np.nan
    )
    metrics = {
        "start": aligned.index.min(),
        "end": aligned.index.max(),
        "observations": len(aligned),
        "portfolio": portfolio_metrics,
        "benchmark": benchmark_metrics,
        "excess_cumulative_return": portfolio_metrics["cumulative_return"] - benchmark_metrics["cumulative_return"],
        "correlation": float(aligned["portfolio"].corr(aligned["benchmark"])),
        "beta": beta,
    }
    return aligned, metrics


class XSec20Service:
    def __init__(self, *, root: Path | None = None) -> None:
        self.root = Path(root or settings.base_dir)
        self.inputs = self.root / "data_store/crypto_momentum_research/inputs"
        self.cache_dir = self.root / "data_store/xsec20_runtime"
        self.benchmark_dir = self.cache_dir / "benchmarks"
        self._lock = threading.Lock()
        self._snapshot: XSecSnapshot | None = None

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def _source_paths(self) -> dict[str, Path]:
        return {
            "close_prices": self.root / "data_store/crypto_research_prices_1d.pkl",
            "universe": self.inputs / "current_universe_snapshot.csv",
            "open_prices": self.inputs / "portfolio_open_prices.parquet",
            "funding": self.inputs / "funding_events.parquet",
            "commissions": self.inputs / "commission_rates.csv",
        }

    def _source_mtimes(self) -> dict[str, int | None]:
        return {
            name: path.stat().st_mtime_ns if path.exists() else None
            for name, path in self._source_paths().items()
        }

    def get_snapshot(self, *, refresh: bool = False) -> XSecSnapshot:
        with self._lock:
            if (
                self._snapshot is not None
                and not refresh
                and self._cache_matches_sources(self._snapshot)
            ):
                return self._snapshot
            cached = self._load_cached()
            if cached is not None and not refresh and self._cache_matches_sources(cached):
                self._snapshot = cached
                return cached
            try:
                self._snapshot = self.build_snapshot()
            except Exception:
                if cached is not None:
                    stale_manifest = dict(cached.manifest)
                    stale_manifest["refresh_error"] = True
                    self._snapshot = XSecSnapshot(
                        manifest=stale_manifest,
                        ticker_history=cached.ticker_history,
                        portfolio=cached.portfolio,
                        standalone_sr=cached.standalone_sr,
                    )
                else:
                    raise
            return self._snapshot

    def _load_cached(self) -> XSecSnapshot | None:
        paths = {
            "manifest": self.manifest_path,
            "ticker": self.cache_dir / "ticker_history.parquet",
            "portfolio": self.cache_dir / "portfolio_daily.parquet",
            "sr": self.cache_dir / "standalone_sr.parquet",
        }
        if not all(path.exists() for path in paths.values()):
            return None
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        ticker = pd.read_parquet(paths["ticker"])
        ticker["timestamp"] = pd.to_datetime(ticker["timestamp"])
        ticker = ticker.set_index(["timestamp", "symbol"]).sort_index()
        portfolio = pd.read_parquet(paths["portfolio"])
        portfolio["timestamp"] = pd.to_datetime(portfolio["timestamp"])
        portfolio = portfolio.set_index("timestamp").sort_index()
        sr_frame = pd.read_parquet(paths["sr"])
        sr = sr_frame.set_index("symbol")["standalone_sr"]
        return XSecSnapshot(manifest, ticker, portfolio, sr)

    def _cache_matches_sources(self, snapshot: XSecSnapshot) -> bool:
        if (
            snapshot.manifest.get("config_id") != CONFIG_ID
            or snapshot.manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        ):
            return False
        return snapshot.manifest.get("source_mtimes") == self._source_mtimes()

    def build_snapshot(self) -> XSecSnapshot:
        # These are the canonical research implementations. Import lazily so the
        # regular bot can still start and serve a cached snapshot if optional
        # research dependencies or inputs are temporarily unavailable.
        from scripts.execute_complete_crypto_study import (
            calibration_path,
            components,
            fast_risk_unit,
            load_full_price_panel,
        )
        from scripts.execute_cross_sectional_trend_study import (
            CrossSectionalModel,
            cross_sectional_constructions,
            exact_config,
            ic_weighted_forecast,
        )
        from scripts.execute_full_crypto_study import funding_coefficients
        from scripts.execute_production_like_walkforward import (
            delayed_positions,
            funding_tradability_mask,
            search_score,
            simulate_arrays,
        )
        from scripts.execute_cross_sectional_trend_study import config_id

        all_prices = load_full_price_panel()
        fit_prices = all_prices.loc[:, all_prices.notna().sum().ge(90)]
        universe = pd.read_csv(self.inputs / "current_universe_snapshot.csv")
        members = universe.loc[
            universe["member"].astype(str).str.lower().isin(["true", "1"]), "symbol"
        ]
        symbols = [symbol for symbol in members if symbol in fit_prices]
        if not symbols:
            raise ValueError("The XSec20 portfolio universe is empty")
        portfolio_prices = fit_prices[symbols]
        open_prices = pd.read_parquet(self.inputs / "portfolio_open_prices.parquet").reindex(
            index=portfolio_prices.index, columns=symbols
        )
        events = pd.read_parquet(self.inputs / "funding_events.parquet")
        events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True)
        same, midnight = funding_coefficients(portfolio_prices, events)
        tradable = funding_tradability_mask(portfolio_prices.index, symbols, events)
        commissions = pd.read_csv(self.inputs / "commission_rates.csv").set_index("symbol")
        maker = commissions["maker"].reindex(symbols).fillna(0.0002)
        taker = commissions["taker"].reindex(symbols).fillna(0.0004)
        close_returns = portfolio_prices.pct_change(fill_method=None).fillna(0.0)
        open_returns = open_prices.shift(-1).div(open_prices).sub(1).fillna(0.0)
        asset_vol = close_returns.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std() * np.sqrt(365)

        # Warm-up windows are intentionally sparse. The canonical research
        # functions use NaN-aware reductions that emit benign RuntimeWarnings
        # before enough observations exist.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_components = components(fit_prices, VOLATILITY_WINDOW)
            fit_returns = fit_prices.pct_change(fill_method=None).to_numpy(float)
            scalars, _, _ = calibration_path(
                raw_components,
                fit_returns,
                fit_prices.index,
                "quarterly",
                0.80,
                0.25,
                125,
                history_days=None,
            )
            locations = [fit_prices.columns.get_loc(symbol) for symbol in portfolio_prices]
            scaled = np.clip(raw_components[:, locations] * scalars[:, None, :], -20, 20)
            absolute_forecast = ic_weighted_forecast(
                scaled,
                portfolio_prices,
                open_prices,
                (20,),
                history_days=None,
            )
        ranks = absolute_forecast.rank(axis=1, pct=True)
        raw_weights = cross_sectional_constructions(
            absolute_forecast, close_returns, include_buffers=False
        )["dollar_neutral"][0]
        model = CrossSectionalModel(
            family="xs_momentum",
            name=MODEL,
            forecast=raw_weights,
            config={
                "signal_family": "xs_momentum",
                "component_horizons": ["2/8", "4/16", "8/32", "16/64", "32/128"],
                "ic_horizons": [20],
                "construction": "dollar_neutral",
                "rank_buffer": 0.0,
                "volatility_window": 60,
                "component_eligibility": "all_components",
            },
        )
        config = exact_config(
            model,
            phase="standard_holdout",
            history_days=None,
            target=TARGET_VOL,
            gross=GROSS_CAP,
            cap=TICKER_RISK_CAP,
            rebalance="daily",
        )
        calculated_id = config_id(config)
        if calculated_id != CONFIG_ID:
            raise RuntimeError(f"Pinned XSec20 configuration changed: {calculated_id} != {CONFIG_ID}")

        unit = fast_risk_unit(
            raw_weights,
            close_returns,
            asset_vol,
            TICKER_RISK_CAP,
            VOLATILITY_WINDOW,
        ).where(tradable, 0.0)
        sim = simulate_arrays(
            unit.to_numpy(float),
            portfolio_prices.index,
            open_returns.to_numpy(float),
            same.to_numpy(float),
            midnight.to_numpy(float),
            maker.to_numpy(float),
            taker.to_numpy(float),
            target_vol=TARGET_VOL,
            gross_cap=GROSS_CAP,
            rebalance="daily",
            detail=True,
            columns=portfolio_prices.columns,
        )

        standalone_desired = (
            absolute_forecast.div(10.0)
            .mul(TARGET_VOL)
            .div(asset_vol.replace(0.0, np.nan))
            .clip(lower=-GROSS_CAP, upper=GROSS_CAP)
        )
        standalone_held = pd.DataFrame(
            delayed_positions(
                standalone_desired.fillna(0.0).to_numpy(float),
                portfolio_prices.index,
                "daily",
            ),
            index=portfolio_prices.index,
            columns=portfolio_prices.columns,
        )
        standalone_trades = standalone_held.diff().abs().fillna(standalone_held.abs())
        standalone_fee = standalone_trades.mul(taker, axis=1)
        standalone_slippage = standalone_trades * SLIPPAGE_BPS / 10_000.0
        standalone_funding = standalone_held * same + standalone_held.shift(1).fillna(0.0) * midnight
        standalone_return = (
            standalone_held * open_returns
            - standalone_fee
            - standalone_slippage
            + standalone_funding
        )
        standalone_sr = {}
        for symbol in symbols:
            valid = standalone_desired[symbol].notna()
            values = standalone_return.loc[valid, symbol].dropna()
            if len(values) < 365 or values.std(ddof=1) <= 0:
                standalone_sr[symbol] = np.nan
            else:
                standalone_sr[symbol] = float(values.mean() / values.std(ddof=1) * np.sqrt(365))
        standalone_sr_series = pd.Series(standalone_sr, name="standalone_sr")

        ticker_history = self._ticker_history(
            portfolio_prices=portfolio_prices,
            absolute_forecast=absolute_forecast,
            ranks=ranks,
            raw_weights=raw_weights,
            target=sim.target,
            held=sim.held,
            contribution=sim.contribution,
            standalone_return=standalone_return,
            standalone_sr=standalone_sr_series,
        )
        net = sim.net.fillna(0.0)
        nav = PORTFOLIO_VALUE * (1.0 + net).cumprod()
        portfolio = pd.DataFrame(
            {
                "net_return": net,
                "gross_return": sim.gross_return,
                "fee": sim.fee,
                "slippage": sim.slippage,
                "funding": sim.funding,
                "turnover": sim.turnover,
                "nav": nav,
                "drawdown": nav.div(nav.cummax()).sub(1.0),
                "long_exposure": sim.held.clip(lower=0).sum(axis=1),
                "short_exposure": sim.held.clip(upper=0).sum(axis=1),
                "gross_exposure": sim.held.abs().sum(axis=1),
                "net_exposure": sim.held.sum(axis=1),
                "realized_volatility_60d": net.rolling(60, min_periods=60).std() * np.sqrt(365),
            },
            index=portfolio_prices.index,
        )
        portfolio.index.name = "timestamp"
        validation = search_score(
            sim.net.to_numpy(float),
            sim.turnover.to_numpy(float),
            portfolio_prices.index,
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2025-12-31"),
        )
        holdout = search_score(
            sim.net.to_numpy(float),
            sim.turnover.to_numpy(float),
            portfolio_prices.index,
            pd.Timestamp("2026-01-01"),
            portfolio_prices.index.max(),
        )
        holdout_net_exposure = portfolio.loc[
            portfolio.index >= pd.Timestamp("2026-01-01"), "net_exposure"
        ].abs()
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "model": MODEL,
            "config_id": CONFIG_ID,
            "methodology_version": "cross-sectional-trend-v1",
            "data_cutoff": portfolio_prices.index.max().isoformat(),
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "universe_as_of": str(universe["as_of"].iloc[0]) if "as_of" in universe else None,
            "universe_assets": len(symbols),
            "fit_assets": fit_prices.shape[1],
            "portfolio_value": PORTFOLIO_VALUE,
            "target_vol": TARGET_VOL,
            "gross_cap": GROSS_CAP,
            "ticker_risk_cap": TICKER_RISK_CAP,
            "volatility_window": VOLATILITY_WINDOW,
            "rebalance": "daily",
            "activation": "next_open",
            "slippage_bps": SLIPPAGE_BPS,
            "taker_share": TAKER_SHARE,
            "validation_net_sharpe": validation[0],
            "validation_annual_turnover": validation[1],
            "holdout_net_sharpe": holdout[0],
            "holdout_annual_turnover": holdout[1],
            "holdout_mean_absolute_net_exposure": float(holdout_net_exposure.mean()),
            "holdout_max_absolute_net_exposure": float(holdout_net_exposure.max()),
            "raw_rank_neutral": True,
            "final_weights_strictly_neutral": False,
            "source_mtimes": self._source_mtimes(),
        }
        snapshot = XSecSnapshot(manifest, ticker_history, portfolio, standalone_sr_series)
        self._persist(snapshot)
        return snapshot

    @staticmethod
    def _ticker_history(
        *,
        portfolio_prices: pd.DataFrame,
        absolute_forecast: pd.DataFrame,
        ranks: pd.DataFrame,
        raw_weights: pd.DataFrame,
        target: pd.DataFrame,
        held: pd.DataFrame,
        contribution: pd.DataFrame,
        standalone_return: pd.DataFrame,
        standalone_sr: pd.Series,
    ) -> pd.DataFrame:
        frames = []
        for name, frame in (
            ("price", portfolio_prices),
            ("absolute_forecast", absolute_forecast),
            ("rank", ranks),
            ("xsec_signal", raw_weights),
            ("target_weight", target),
            ("held_weight", held),
            ("contribution", contribution),
            ("standalone_return", standalone_return),
        ):
            series = frame.stack(future_stack=True).rename(name)
            frames.append(series)
        history = pd.concat(frames, axis=1)
        history.index.names = ["timestamp", "symbol"]
        history["standalone_sr"] = history.index.get_level_values("symbol").map(standalone_sr)
        history["target_notional"] = history["target_weight"] * PORTFOLIO_VALUE
        history["held_notional"] = history["held_weight"] * PORTFOLIO_VALUE
        history["estimated_quantity"] = history["target_notional"].div(history["price"].replace(0.0, np.nan))
        history["xsec_side"] = np.select(
            [history["target_weight"].gt(1e-10), history["target_weight"].lt(-1e-10)],
            ["LONG", "SHORT"],
            default="FLAT",
        )
        history["trend_side"] = np.where(history["absolute_forecast"].ge(0), "LONG", "SHORT")
        return history.sort_index()

    def _persist(self, snapshot: XSecSnapshot) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot.ticker_history.reset_index().to_parquet(
            self.cache_dir / "ticker_history.parquet", index=False
        )
        snapshot.portfolio.reset_index().to_parquet(
            self.cache_dir / "portfolio_daily.parquet", index=False
        )
        snapshot.standalone_sr.rename("standalone_sr").rename_axis("symbol").reset_index().to_parquet(
            self.cache_dir / "standalone_sr.parquet", index=False
        )
        self.manifest_path.write_text(
            json.dumps(snapshot.manifest, indent=2, default=str), encoding="utf-8"
        )

    def benchmark_returns(
        self, symbol: str, *, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        canonical = normalize_symbol(symbol)
        cache_path = self.benchmark_dir / f"{canonical}_1d.parquet"
        prices = pd.Series(dtype=float)
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            cached["timestamp"] = pd.to_datetime(cached["timestamp"])
            prices = cached.set_index("timestamp")["close"].sort_index()
        if prices.empty or prices.index.min() > start or prices.index.max() < end:
            try:
                from scripts.execute_complete_crypto_study import load_full_price_panel

                panel = load_full_price_panel()
                if canonical in panel:
                    prices = panel[canonical].dropna().rename("close")
            except Exception:
                pass
        if prices.empty or prices.index.max() < end:
            fetched = get_fresh_lookback_df(canonical, start, freq="1d")
            if fetched.empty:
                raise KeyError(f"No Binance daily history is available for {canonical}")
            fresh = fetched["Close"].rename("close")
            prices = fresh.combine_first(prices).sort_index()
            self.benchmark_dir.mkdir(parents=True, exist_ok=True)
            prices.rename_axis("timestamp").reset_index().to_parquet(cache_path, index=False)
        prices = prices.loc[(prices.index >= start) & (prices.index <= end)].dropna()
        returns = prices.pct_change(fill_method=None).dropna()
        if not returns.empty:
            # Passive benchmark pays the same assumed taker fee and 5 bps once.
            returns.iloc[0] -= 0.0004 + SLIPPAGE_BPS / 10_000.0
        return returns.rename(canonical)

    def sector_symbols(self, sector: str) -> set[str]:
        mappings = pd.read_csv(self.inputs / "symbol_mappings.csv")
        metadata_path = self.inputs / "coingecko_asset_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        terms = {
            "l1": ("Layer 1", "Smart Contract Platform"),
            "eth_beta": ("Ethereum Ecosystem", "Layer 2"),
            "sol_beta": ("Solana Ecosystem",),
            "meme": ("Meme", "Dog-Themed", "Cat-Themed"),
        }.get(sector, ())
        selected = set()
        for row in mappings.itertuples():
            categories = metadata.get(row.provider_id, {}).get("categories", [])
            if any(any(term.lower() in str(category).lower() for term in terms) for category in categories):
                selected.add(row.binance_symbol)
        return selected

    def is_stale(self, snapshot: XSecSnapshot) -> bool:
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        return snapshot.data_cutoff.normalize() < today - pd.Timedelta(days=1)


_SERVICE: XSec20Service | None = None


def get_xsec20_service() -> XSec20Service:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = XSec20Service()
    return _SERVICE
