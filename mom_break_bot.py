# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, timezone
from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.ext import CallbackContext
from telegram.request import HTTPXRequest
import asyncio 
import ast
import dateparser
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import colors 
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize
import tqdm
import time
import sys
import pickle
import logging
from binance.exceptions import BinanceAPIException
from binance.client import Client
import portfolio_strategy as cta
import trading_bot as pos
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
import mommy_bot as momo
import html
import urllib3
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from momo_bot.config import settings
from momo_bot.binance_rate_limit import BinanceRestCircuitOpen
from momo_bot.binance_stream import stream_closed_klines
from momo_bot.candles import check_freshness
from momo_bot.data_store import MarketDataStore
from momo_bot.data_validation import infer_frequency
from momo_bot import xsec_telegram

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# --- Suppress specific pandas FutureWarning ---
warnings.filterwarnings("ignore", category=FutureWarning, module='pandas')
if not settings.binance_verify_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- File Paths and Clients (loaded from env/config) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# DATA LOADING & PROCESSING (Breakout)
# ==============================================================================

# --- NEW: Breakout Functions ---
_breakout_store = (
    MarketDataStore(path=settings.breakout_price_data_path, freq=settings.breakout_data_frequency)
    if settings.breakout_price_data_path
    else None
)

_BUNDLE_CACHE: Dict[str, Any] = {"path": None, "bundle": None}


def _find_latest_file(data_dir: Path, patterns: Iterable[str], *, exclude_substr: str | None = None) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(data_dir.glob(pattern))
    if exclude_substr:
        candidates = [path for path in candidates if exclude_substr not in path.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_latest_bundle(data_dir: Path) -> tuple[Dict[str, Any] | None, Path | None]:
    latest_path = _find_latest_file(data_dir, ["backtest_results_bundle_*.pkl"])
    if latest_path is None:
        return None, None
    if _BUNDLE_CACHE["path"] == latest_path:
        return _BUNDLE_CACHE["bundle"], latest_path
    bundle = pd.read_pickle(latest_path)
    _BUNDLE_CACHE["path"] = latest_path
    _BUNDLE_CACHE["bundle"] = bundle
    return bundle, latest_path


def _load_latest_bundle_containing(data_dir: Path, strategy: str) -> tuple[Dict[str, Any] | None, Path | None]:
    bundles = sorted(data_dir.glob("backtest_results_bundle_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in bundles:
        bundle = pd.read_pickle(path)
        if isinstance(bundle, dict) and strategy in bundle:
            return bundle, path
    return None, None


def _normalize_params(raw_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for ticker, payload in raw_params.items():
        if not isinstance(payload, dict):
            continue
        if "weights" not in payload:
            continue
        entry = dict(payload)
        entry.setdefault("status", "success")
        normalized[ticker] = entry
    return normalized


def _params_from_bundle(bundle: Dict[str, Any], strategy: str) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    section = bundle.get(strategy, {}) if isinstance(bundle, dict) else {}
    results = section.get("results", {}) if isinstance(section, dict) else {}
    params: Dict[str, Dict[str, Any]] = {}
    for ticker, payload in results.items():
        raw_params = payload.get("params") if isinstance(payload, dict) else None
        if not raw_params:
            continue
        entry = dict(raw_params)
        entry.setdefault("status", "success")
        if "weights" in entry:
            params[ticker] = entry
    return params, section


def _load_latest_strategy_params(strategy: str) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any] | None, str]:
    data_dir = settings.data_dir
    bundle, bundle_path = _load_latest_bundle(data_dir)
    if bundle and strategy in bundle:
        params, section = _params_from_bundle(bundle, strategy)
        if params:
            return params, section, f"bundle:{bundle_path}"

    if settings.allow_older_strategy_bundle:
        bundle, bundle_path = _load_latest_bundle_containing(data_dir, strategy)
        if bundle and bundle_path is not None:
            params, section = _params_from_bundle(bundle, strategy)
            if params:
                return params, section, f"bundle:{bundle_path}"

    if strategy == "momentum":
        patterns = ["optimized_momentum_weights_*.pkl", "optimized_crypto_weights_carver.pkl", "optimized_crypto_params.pkl"]
        fallback_path = settings.momentum_params_path
        exclude = None
    elif strategy == "breakout":
        patterns = ["optimized_breakout_params_*.pkl", "optimized_breakout_params.pkl"]
        fallback_path = settings.breakout_params_path
        exclude = "pairs"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    latest_path = _find_latest_file(data_dir, patterns, exclude_substr=exclude)
    if latest_path is None and fallback_path is not None:
        fallback_path = Path(fallback_path)
        if fallback_path.exists():
            latest_path = fallback_path

    if latest_path is None:
        return {}, None, "missing"

    raw_params = pd.read_pickle(latest_path)
    return _normalize_params(raw_params), None, str(latest_path)


def _resolve_breakout_rules(section: Dict[str, Any] | None) -> list[int]:
    if section and isinstance(section.get("breakout_horizons"), list):
        return [int(x) for x in section["breakout_horizons"]]
    return [10, 20, 40, 80, 160]


def _resolve_ewmac_rules(section: Dict[str, Any] | None) -> list[tuple[int, int]]:
    raw = section.get("ewmac_factors") if section else None
    if not raw:
        return [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128)]
    rules: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, tuple) and len(item) == 2:
            rules.append((int(item[0]), int(item[1])))
        elif isinstance(item, list) and len(item) == 2:
            rules.append((int(item[0]), int(item[1])))
        else:
            parsed = ast.literal_eval(str(item))
            rules.append((int(parsed[0]), int(parsed[1])))
    return rules


def _resolve_ewmac_scalars(section: Dict[str, Any] | None) -> dict[tuple[int, int], float] | None:
    raw = section.get("ewmac_scalars") if section else None
    if not isinstance(raw, dict):
        return None
    scalars: dict[tuple[int, int], float] = {}
    for key, value in raw.items():
        if isinstance(key, tuple) and len(key) == 2:
            pair = (int(key[0]), int(key[1]))
        else:
            parsed = ast.literal_eval(str(key))
            pair = (int(parsed[0]), int(parsed[1]))
        scalars[pair] = float(value)
    return scalars


def _resolve_breakout_scalars(section: Dict[str, Any] | None) -> dict[int, float] | None:
    raw = section.get("breakout_scalars") if section else None
    if not isinstance(raw, dict):
        return None
    return {int(k): float(v) for k, v in raw.items()}


def _breakout_runtime_config(section: Dict[str, Any] | None) -> dict[str, Any]:
    return {
        # Keep live breakout reports on fixed Carver scalars and fixed DM for now.
        # Bundle params/weights can still come from the latest valid bundle.
        "forecast_scalars": None,
        "diversification_multiplier": 1.24,
        "cap_final_forecast": bool(section.get("breakout_final_cap", True)) if section else True,
        "vol_lookback_for_position_sizing": int(section.get("vol_lookback", 180)) if section else 180,
        "data_frequency": section.get("effective_frequency") or section.get("data_frequency") if section else "1d",
    }


def _momentum_runtime_config(section: Dict[str, Any] | None) -> dict[str, Any]:
    return {
        "ewmac_factors": _resolve_ewmac_rules(section),
        "forecast_scalars": _resolve_ewmac_scalars(section),
        "diversification_multiplier": float(section["ewmac_dm_value"])
        if section and "ewmac_dm_value" in section
        else 1.12,
        "cap_final_forecast": bool(section.get("ewmac_final_cap", True)) if section else True,
        "data_frequency": section.get("effective_frequency") or section.get("data_frequency") if section else "4h",
    }

def load_data_breakout():
    """Loads all data and parameters for the BREAKOUT strategy."""
    if not settings.breakout_price_data_path:
        raise FileNotFoundError(
            "Breakout paths not configured. Set MOMO_BREAKOUT_PRICE_DATA_PATH."
        )

    # Keep daily data up to date and cached in-process.
    if _breakout_store is None:
        raise FileNotFoundError("Breakout data store not configured.")

    optimized_parameters, breakout_section, params_source = _load_latest_strategy_params("breakout")
    if not optimized_parameters:
        raise FileNotFoundError("Breakout params not found in data_store or configured path.")
    optimized_tickers = sorted(optimized_parameters.keys())

    def _updater(df: pd.DataFrame, freq: str, path: os.PathLike) -> pd.DataFrame:
        return momo.update_historical_data(
            df,
            freq=freq,
            out_path=str(path),
            target_tickers=optimized_tickers,
            require_complete=False,
            signal_tickers=list(optimized_parameters.keys()),
            metadata_extra={"params_source": params_source},
        )

    refresh = _breakout_store.refresh(updater=_updater)
    tickers_price_data = refresh.data.drop_duplicates()
    print(f"Using breakout params from {params_source}")
    ticker_symbol_map = {ticker: ticker for ticker in optimized_parameters.keys()}
    runtime_config = _breakout_runtime_config(breakout_section)
    data_frequency = runtime_config["data_frequency"]
    breakout_rules = _resolve_breakout_rules(breakout_section)
    return tickers_price_data, optimized_parameters, ticker_symbol_map, data_frequency, breakout_rules, runtime_config

pairs_list = [
    'BTCUSDT/XRPUSDT',
    'BTCUSDT/ETHUSDT',
    'BTCUSDT/HBARUSDT',
    'BTCUSDT/ADAUSDT',
    'BTCUSDT/DOTUSDT',
    'BTCUSDT/XLMUSDT',
    'BTCUSDT/INJUSDT',
    'BTCUSDT/SOLUSDT',
    'BTCUSDT/SEIUSDT',
    'BTCUSDT/SUIUSDT',
    'ETHUSDT/ARBUSDT',
    'ETHUSDT/SOLUSDT',
    'ETHUSDT/EIGENUSDT',
    'ETHUSDT/ENAUSDT',
    'ETHUSDT/OPUSDT',
    'ETHUSDT/STRKUSDT',
    'AAVEUSDT/ENAUSDT',
    'XRPUSDT/XLMUSDT',
    'AAVEUSDT/EIGENUSDT',
    'SUIUSDT/APTUSDT',
    'TONUSDT/SUIUSDT',
    'HYPEUSDT/SOLUSDT',
    'DOGEUSDT/1000PEPEUSDT',
    '1000BONKUSDT/1000PEPEUSDT',
    '1000BONKUSDT/FARTCOINUSDT',
    'GRASSUSDT/TAOUSDT',
    'JUPUSDT/UNIUSDT',
    'JUPUSDT/SOLUSDT',
    'AAVEUSDT/ETHUSDT',
    'TONUSDT/SUIUSDT',
    'ETHUSDT/APTUSDT',
    'ETHUSDT/1000BONKUSDT'
]

# Create pairs data
def create_pairs_df(pairs_list: list, price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a DataFrame of pairs time series from a list of pair strings
    and a DataFrame of individual ticker prices.

    Args:
        pairs_list (list): A list of strings, where each string represents a pair
                           formatted as 'TICKER1/TICKER2'.
        price_data (pd.DataFrame): A DataFrame with a DatetimeIndex and columns
                                   for each individual ticker's price.

    Returns:
        pd.DataFrame: A new DataFrame where each column is a time series
                      representing the ratio of a pair.
    """
    all_pairs_series = []
    
    print("Processing pairs...")
    for pair_string in pairs_list:
        try:
            # 1. Split the string to get the two tickers
            ticker1, ticker2 = pair_string.split('/')
            
            # 2. Check if both tickers exist in the price data
            if ticker1 in price_data.columns and ticker2 in price_data.columns:
                # 3. Calculate the ratio of the two price series
                pair_series = price_data[ticker1] / price_data[ticker2]
                
                # 4. Name the new series with the original pair string
                pair_series.name = pair_string
                
                all_pairs_series.append(pair_series)
            else:
                print(f"Warning: Could not create pair '{pair_string}'. Ticker(s) not found in price data.")

        except ValueError:
            print(f"Warning: Skipping invalid pair string format: '{pair_string}'")
        except Exception as e:
            print(f"An error occurred processing '{pair_string}': {e}")
            
    if not all_pairs_series:
        print("No valid pairs were created.")
        return pd.DataFrame()

    # 5. Combine all the individual pair series into a single DataFrame
    # join='outer' ensures all dates are kept, filling missing values with NaN
    final_pairs_df = pd.concat(all_pairs_series, axis=1, join='outer')
    
    # Optional: Sort the index to ensure chronological order
    final_pairs_df.sort_index(inplace=True)
    
    print("Successfully created pairs DataFrame.")
    return final_pairs_df

def run_breakout_curr_sig(
    tickers_price_data,
    data_frequency,
    optimized_parameters,
    ticker_symbol_map,
    breakout_rules: list[int] | None = None,
    runtime_config: dict[str, Any] | None = None,
):
    """Runner function for the BREAKOUT strategy."""
    should_update = True
    if not tickers_price_data.empty and isinstance(tickers_price_data.index, pd.DatetimeIndex):
        freshness = check_freshness(tickers_price_data.index, freq=data_frequency)
        if freshness.is_fresh:
            should_update = False
            print("✅ Breakout data is up to date (UTC completed candle check).")
            print(f"   Latest data timestamp: {freshness.latest.strftime('%Y-%m-%d %H:%M:%S') if freshness.latest else 'None'}")
            print(f"   Required timestamp:    {freshness.required.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("⚠️ Breakout data is stale. Update required (UTC completed candle check).")
            print(f"   Latest data timestamp: {freshness.latest.strftime('%Y-%m-%d %H:%M:%S') if freshness.latest else 'None'}")
            print(f"   Required timestamp:    {freshness.required.strftime('%Y-%m-%d %H:%M:%S')}")

    if should_update:
        print("Breakout data is stale or empty. Updating historical data...")
        out_path = str(settings.breakout_price_data_path) if settings.breakout_price_data_path else None
        try:
            new_data = momo.update_historical_data(
                tickers_price_data,
                freq=data_frequency,
                out_path=out_path,
                target_tickers=list(optimized_parameters.keys()),
                signal_tickers=list(optimized_parameters.keys()),
            )
        except BinanceRestCircuitOpen as exc:
            print(f"Skipping breakout refresh because Binance REST is cooling down: {exc}")
            new_data = tickers_price_data
    else:
        new_data = tickers_price_data

    # Ensure we only use fully completed candles for signal generation.
    if not new_data.empty and isinstance(new_data.index, pd.DatetimeIndex):
        required = check_freshness(new_data.index, freq=data_frequency).required
        new_data = new_data.loc[:required]
    
    if breakout_rules is None:
        breakout_rules = [10, 20, 40, 80, 160]
    runtime_config = runtime_config or {}

    _pnl_dict, _signal_dict, summary_df = cta.carver_gen_signal_unified(
        price_frame=new_data.drop_duplicates(),
        tickers=list(optimized_parameters.keys()),
        ticker_dict=ticker_symbol_map,
        optimized_inputs_dict=optimized_parameters,
        rule_variations=breakout_rules,
        strategy_type='breakout',
        data_frequency=data_frequency,
        pnl_dict_gen=True,
        portfolio_value=settings.portfolio_value,
        target_vol_annual=settings.target_vol_annual,
        max_leverage=settings.max_leverage,
        vol_lookback_for_position_sizing=runtime_config.get("vol_lookback_for_position_sizing", 180),
        forecast_scalars=runtime_config.get("forecast_scalars"),
        diversification_multiplier=runtime_config.get("diversification_multiplier"),
        cap_final_forecast=runtime_config.get("cap_final_forecast", True),
    )
    return summary_df

def run_breakout_curr_pairs(
    tickers_price_data,
    data_frequency,
    optimized_parameters,
    ticker_symbol_map,
    breakout_rules: list[int] | None = None,
    runtime_config: dict[str, Any] | None = None,
):
    """Runner function for the BREAKOUT strategy."""
    should_update = True
    if not tickers_price_data.empty and isinstance(tickers_price_data.index, pd.DatetimeIndex):
        freshness = check_freshness(tickers_price_data.index, freq=data_frequency)
        if freshness.is_fresh:
            should_update = False
            print("✅ Breakout data is up to date (UTC completed candle check).")
            print(f"   Latest data timestamp: {freshness.latest.strftime('%Y-%m-%d %H:%M:%S') if freshness.latest else 'None'}")
            print(f"   Required timestamp:    {freshness.required.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("⚠️ Breakout data is stale. Update required (UTC completed candle check).")
            print(f"   Latest data timestamp: {freshness.latest.strftime('%Y-%m-%d %H:%M:%S') if freshness.latest else 'None'}")
            print(f"   Required timestamp:    {freshness.required.strftime('%Y-%m-%d %H:%M:%S')}")

    if should_update:
        print("Breakout data is stale or empty. Updating historical data...")
        out_path = str(settings.breakout_price_data_path) if settings.breakout_price_data_path else None
        try:
            new_data = momo.update_historical_data(
                tickers_price_data,
                freq=data_frequency,
                out_path=out_path,
                target_tickers=list(optimized_parameters.keys()),
                signal_tickers=list(optimized_parameters.keys()),
            )
        except BinanceRestCircuitOpen as exc:
            print(f"Skipping breakout pairs refresh because Binance REST is cooling down: {exc}")
            new_data = tickers_price_data
    else:
        new_data = tickers_price_data

    # Ensure we only use fully completed candles for signal generation.
    if not new_data.empty and isinstance(new_data.index, pd.DatetimeIndex):
        required = check_freshness(new_data.index, freq=data_frequency).required
        new_data = new_data.loc[:required]
    
    if breakout_rules is None:
        breakout_rules = [10, 20, 40, 80, 160]
    runtime_config = runtime_config or {}

    pairs_data = create_pairs_df(pairs_list, new_data.drop_duplicates())

    _pnl_dict, _signal_dict, summary_df = cta.carver_gen_signal_unified(
        price_frame=pairs_data,
        tickers=list(optimized_parameters.keys()),
        ticker_dict={x: x for x in optimized_parameters.keys()},
        optimized_inputs_dict=optimized_parameters,
        rule_variations=breakout_rules,
        strategy_type='breakout',
        data_frequency=data_frequency,
        pnl_dict_gen=True,
        portfolio_value=settings.portfolio_value,
        target_vol_annual=settings.target_vol_annual,
        max_leverage=settings.max_leverage,
        vol_lookback_for_position_sizing=runtime_config.get("vol_lookback_for_position_sizing", 180),
        forecast_scalars=runtime_config.get("forecast_scalars"),
        diversification_multiplier=runtime_config.get("diversification_multiplier"),
        cap_final_forecast=runtime_config.get("cap_final_forecast", True),
    )
    return summary_df


# Getting current signals for current positioning
def get_position_break(cur_sig):
    positions = pos.gen_ls_frame(pos.get_futures_positions())
    longs,shorts = positions[0], positions[1]
    longs['direction'] = 'LONG'
    shorts['direction'] = 'SHORT'

    total_positions = pd.concat([longs.iloc[:-1], shorts.iloc[:-1]], axis=0)
    positions_side = total_positions.symbol.tolist()

    # Filter the current signals to only include those in the positions

    cur_sig = cur_sig.set_index('Name')
    my_posit = cur_sig.loc[[x for x in positions_side if x in cur_sig.index], :]
    my_posit = my_posit.rename_axis('symbol')

    # Setting up the direction of the positions
    total_positions_edt = total_positions.loc[:, ['symbol', 'direction', 'notional']]
    total_positions_edt.set_index('symbol', inplace=True)
    my_posit = pd.concat([my_posit, total_positions_edt], axis=1, join='inner')
    my_posit = my_posit.reset_index()
    my_posit = my_posit.drop(columns=['Position Size'])
    # print(f'MY POSIT: {my_posit.columns}')
    my_posit.columns = ['COIN', 'BREAK DIR', 'SIGNAL', 'SR', 'PORT DIR', 'NTL']
    my_posit['COIN'] = my_posit['COIN'].str.removesuffix('USDT')
    my_posit['NTL']= (my_posit['NTL'].astype(float))/1000
    my_posit['NTL'] = my_posit['NTL'].apply(lambda x: f"{x:.2f}k")
    my_posit['BREAKOUT'] = np.where(my_posit['BREAK DIR'] == my_posit['PORT DIR'], 'YES', 'NO')
    return my_posit


from plotly.subplots import make_subplots

def generate_plotly_chart(
    pnl_series: pd.Series,
    signal_df: pd.DataFrame,
    ticker: str,
    lookback: int = 90,
    max_zero_cross_labels: int = 8,
) -> go.Figure:
    """
    Generates an interactive Plotly chart with correctly aligned and formatted date axes.
    """
    # Ensure the index is a proper DatetimeIndex
    if not isinstance(pnl_series.index, pd.DatetimeIndex):
        pnl_series.index = pd.to_datetime(pnl_series.index)
    if not isinstance(signal_df.index, pd.DatetimeIndex):
        signal_df.index = pd.to_datetime(signal_df.index)

    # --- FIX: Ensure all data uses the exact same aligned index ---
    
    # 1. First, prepare the primary data for the main (bottom) plot.
    # The .dropna() here is the source of the index mismatch.
    plot_data = signal_df.iloc[-lookback:].dropna(subset=['weighted_signal', 'price'])

    # 2. Now, use the final index from plot_data to slice the PnL data.
    # This guarantees both DataFrames have the exact same index.
    pnl_to_plot = pnl_series.loc[plot_data.index]
    if not pnl_to_plot.empty:
        pnl_to_plot = pnl_to_plot - pnl_to_plot.iloc[0]

    if pnl_to_plot.empty or plot_data.empty:
        return go.Figure().update_layout(title_text=f"Not enough recent data to plot for {ticker}", template="plotly_dark")

    # --- End of FIX ---

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(f"Cumulative PnL (Last {len(pnl_to_plot)} periods)", f"Signal vs. Price"),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]] # <-- THIS IS THE FIX
    )

    print(pnl_to_plot.index)
    print(plot_data.index)
    
    # --- Chart 1: Cumulative PnL ---
    fig.add_trace(go.Scatter(
        x=pnl_to_plot.index, y=pnl_to_plot, name='Cumulative PnL', line=dict(color='dodgerblue')
    ), row=1, col=1)

    # --- Chart 2: Signal vs. Price ---
    signal = plot_data['weighted_signal']
    fig.add_trace(go.Scatter(x=plot_data.index, y=signal, name='Weighted Signal', line=dict(color='white')), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['price'], name='Price', line=dict(color='darkorange', dash='dot')), secondary_y=True, row=2, col=1)

    # Add zero line and markers
    fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)
    sign_change = np.sign(signal).diff().ne(0)
    crossing_points = plot_data[sign_change]
    if not crossing_points.empty and crossing_points.index[0] == plot_data.index[0]:
        crossing_points = crossing_points.iloc[1:]
    fig.add_trace(
        go.Scatter(
            x=crossing_points.index,
            y=np.zeros(len(crossing_points)),
            mode='markers',
            name='Zero Crossing',
            marker=dict(color='cyan', size=8, symbol='x'),
        ),
        row=2,
        col=1,
    )
    if not crossing_points.empty:
        label_idx = crossing_points.index[-max_zero_cross_labels:]
        y_min = float(signal.min()) if pd.notnull(signal.min()) else -1.0
        y_max = float(signal.max()) if pd.notnull(signal.max()) else 1.0
        margin = max(1.0, 0.1 * (y_max - y_min))
        label_y = min(max(0.0, y_min + margin), y_max - margin)
        for dt_idx in label_idx:
            fig.add_vline(x=dt_idx, line_dash="dot", line_color="cyan", opacity=0.35, row=2, col=1)
        fig.add_trace(
            go.Scatter(
                x=label_idx,
                y=[label_y] * len(label_idx),
                text=[ts.strftime("%Y-%m-%d") for ts in label_idx],
                mode="markers+text",
                textposition="top center",
                textfont=dict(color="cyan", size=11),
                marker=dict(color="cyan", size=7, symbol="x"),
                name="Zero Cross Dates",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # --- Layout Updates ---
    fig.update_layout(
        title_text=f"{ticker} Breakout ({lookback}D)", template="plotly_dark", height=1000, width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1.02),
        hovermode="x unified", margin=dict(t=100)
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="PnL", row=1, col=1)
    fig.update_yaxes(title_text="Signal Strength", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price", row=2, col=1, secondary_y=True, showgrid=False)
    
    return fig

def generate_matplotlib_chart(
    pnl_series: pd.Series,
    signal_df: pd.DataFrame,
    ticker: str,
    lookback: int = 90,
    max_zero_cross_labels: int = 8,
):
    """
    Generates and RETURNS a Matplotlib figure object with two charts,
    styled with a dark theme.
    """
    plt.style.use('dark_background')

    # --- Data Preparation and Alignment ---
    plot_data = signal_df.iloc[-lookback:].dropna(subset=['weighted_signal', 'price'])
    if plot_data.empty:
        print(f"No valid data to plot for {ticker} in the lookback window.")
        return None # Return None if there's no data
        
    pnl_to_plot = pnl_series.loc[plot_data.index]
    if not pnl_to_plot.empty:
        pnl_to_plot = pnl_to_plot - pnl_to_plot.iloc[0]

    # --- Create Figure and Subplots ---
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [1, 2]}
    )
    fig.suptitle(f"{ticker} Strategy Analysis", fontsize=16, color='white')

    # --- Chart 1: Cumulative PnL ---
    ax1.plot(pnl_to_plot.index, pnl_to_plot, label='Cumulative PnL', color='dodgerblue')
    ax1.set_title(f"Cumulative PnL (Last {len(pnl_to_plot)} periods)")
    ax1.set_ylabel("PnL")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    # --- Chart 2: Signal vs. Price ---
    signal = plot_data['weighted_signal']
    ax2.plot(plot_data.index, signal, label='Weighted Signal', color='white', zorder=10)
    ax2.set_ylabel("Signal Strength", color='white')
    ax2.tick_params(axis='y', labelcolor='white')

    ax2.axhline(0, color='grey', linestyle='--', linewidth=1.2, zorder=5)
    
    sign_change = np.sign(signal).diff().ne(0)
    crossing_points = plot_data[sign_change]
    if not crossing_points.empty and crossing_points.index[0] == plot_data.index[0]:
        crossing_points = crossing_points.iloc[1:]
    ax2.scatter(
        crossing_points.index,
        np.zeros(len(crossing_points)),
        color='cyan',
        s=50,
        zorder=20,
        label='Zero Crossing',
        marker='x',
    )
    if not crossing_points.empty:
        label_idx = crossing_points.index[-max_zero_cross_labels:]
        y_min = float(signal.min()) if pd.notnull(signal.min()) else -1.0
        y_max = float(signal.max()) if pd.notnull(signal.max()) else 1.0
        margin = max(1.0, 0.1 * (y_max - y_min))
        label_y = min(max(0.0, y_min + margin), y_max - margin)
        for dt_idx in label_idx:
            ax2.axvline(dt_idx, color='cyan', linestyle=':', linewidth=1.0, alpha=0.4, zorder=5)
            ax2.annotate(
                dt_idx.strftime("%Y-%m-%d"),
                xy=(dt_idx, label_y),
                xytext=(0, 6),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=8,
                color='cyan',
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'),
            )

    ax2_price = ax2.twinx()
    ax2_price.plot(plot_data.index, plot_data['price'], label='Price', color='darkorange',
                   linestyle='dotted', alpha=0.9)
    ax2_price.set_ylabel("Price", color='white')
    ax2_price.tick_params(axis='y', labelcolor='white')

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_price.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlabel("Date")
    fig.autofmt_xdate()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- FIX: Return the figure object instead of saving it ---
    return fig

######################## RUNNING CODE ####################################
top_100_mcap = [
    'BTCUSDT',
    'ETHUSDT',
    # 'ETHBTC',
    # 'BTCDOMUSDT'
    'SOLUSDT',
    'BNBUSDT',
    'XRPUSDT',
    'DOGEUSDT',
    'ADAUSDT',
    '1000SHIBUSDT',
    'AVAXUSDT',
    'TRXUSDT',
    'DOTUSDT',
    'LINKUSDT',
    'BCHUSDT',
    'TONUSDT',
    'NEARUSDT',
    'LTCUSDT',
    'POLUSDT',
    '1000PEPEUSDT',
    'ICPUSDT',
    'UNIUSDT',
    'ETCUSDT',
    'APTUSDT',
    'XLMUSDT',
    'HBARUSDT',
    'RENDERUSDT',
    'ATOMUSDT',
    'ARBUSDT',
    'IMXUSDT',
    'FILUSDT',
    'MKRUSDT',
    'INJUSDT',
    'SUIUSDT',
    'OPUSDT',
    'GRTUSDT',
    'TAOUSDT',
    'FETUSDT',
    # 'TFUELUSDT', # Note: TFUEL is part of the Theta ecosystem (THETA)
    'LDOUSDT',
    'SEIUSDT',
    'VETUSDT',
    'AAVEUSDT',
    'ALGOUSDT',
    'GALAUSDT',
    'JUPUSDT',
    'FLOWUSDT',
    'STRKUSDT',
    'QNTUSDT',
    'SUSDT',
    'THETAUSDT',
    'DYDXUSDT',
    'WIFUSDT',
    'SUSHIUSDT', # Note: Often referred to as SUSHI
    # 'BEAMUSDT',
    'RUNEUSDT',
    # 'AGIXUSDT',
    'EGLDUSDT',
    'PENDLEUSDT',
    'SANDUSDT',
    'MANAUSDT',
    'AXSUSDT',

    'XTZUSDT',
    # 'EOSUSDT',
    'PYTHUSDT',
    'CHZUSDT',
    '1000BONKUSDT',
    'NEOUSDT',
    # 'CFXUSDT',
    # 'KCSUSDT',
    'MINAUSDT',
    '1000SATSUSDT', # Note: This represents SATS
    'IOTAUSDT',
    # 'GNOUSDT',
    'WLDUSDT',

    # 'CVXUSDT',
    'FXSUSDT',
    # 'KLAYUSDT',
    # 'XECUSDT',
    'CRVUSDT',
    # 'WAVESUSDT',
    'COMPUSDT',
    'ZECUSDT',
    'ORDIUSDT',
    'DYMUSDT',
    'ROSEUSDT',
    '1000FLOKIUSDT',
    'KASUSDT',
    'CAKEUSDT',
    'CELOUSDT',
    'ARUSDT',
    'ALTUSDT',
    'GMTUSDT',
    'MASKUSDT',
    'ZILUSDT',
    'ENSUSDT',
    'ANKRUSDT',
    'IOTXUSDT',
    'WOOUSDT'
]

# Sector specific momentum
REV_GENERATING = [x + "USDT"  for x in ['MKR', 'AAVE', 'UNI']]
L1 = [x + "USDT"  for x in ['ETH', 'BNB', 'SOL', 'ADA', 'TRX', 'TON', 'AVAX', 'SUI', 'APT', 'SEI', 'XRP', 'DOT', 'LTC', 'FIL', 'NEAR', 'XLM', 'HBAR', 'ATOM', 'INJ', 'BCH']]
SOL_BETA = [x + "USDT"  for x in ['SOL', 'JUP', 'JTO', 'RAYSOL', 'PYTH', 'DRIFT']]
ETH_BETA = [x + "USDT"  for x in ['ETH', 'LDO', 'UNI','OP', 'AAVE', 'ENA', 'STRK', 'MKR','PENDLE', 'POL', 'ARB', 'ZK', 'FXS', 'EIGEN', 'IMX', 'SUSHI', 'DYDX', 'SNX', 'CRV', 'GNS', 'ENS']]
BNB_BETA = [x + "USDT"  for x in ['BNB', 'CAKE', 'BAKE']]
MEME = [x + "USDT"  for x in ['TRUMP', 'MELANIA', 'FARTCOIN', 'ZEREBRO', '1000PEPE', 'WIF', '1000BONK', 'DOGE', '1000SHIB', 'POPCAT', 'PNUT', '1000RATS', 'MOODENG', 'CHILLGUY','1000SATS']]
HIGH_FDV_GARBO = [x + "USDT"  for x in ['SUI', 'SEI', 'APT', 'OM', 'IP', 'MOVE', 'WLD']]
AI = [x + "USDT"  for x in ['GRASS', 'KAITO','TAO', 'WLD', 'NEAR', 'RENDER', 'AR', 'ARKM', 'AI16Z', 'FET']]
DINOCOINS = [x + "USDT"  for x in ['ADA', 'XRP', 'XLM','BCH', 'DOT']]
COMPLETE_DOGHSIT = [x + "USDT"  for x in ['BB']]


# ==============================================================================
# TELEGRAM COMMAND HANDLERS
# ==============================================================================

# ------------------------------------------------------------------------------
# Shared helpers (keep handler behavior stable, reduce duplication)
# ------------------------------------------------------------------------------

_momentum_store = MarketDataStore(path=settings.momentum_price_data_path, freq=settings.momentum_data_frequency)


def _refresh_momentum_prices(df: pd.DataFrame, freq: str, path: os.PathLike) -> pd.DataFrame:
    return momo.update_historical_data(df, freq=freq, out_path=str(path), target_tickers=sorted(df.columns))


def _load_momentum_context():
    """
    Loads momentum price data + params, ensuring price data is up to date.
    Returns the same logical inputs as `momo.load_data()`, but avoids repeated disk reads.
    """
    optimized_parameters, momentum_section, params_source = _load_latest_strategy_params("momentum")
    if not optimized_parameters:
        raise FileNotFoundError("Momentum params not found in data_store or configured path.")
    tickers = sorted(optimized_parameters.keys())

    def _updater(df: pd.DataFrame, freq: str, path: os.PathLike) -> pd.DataFrame:
        return momo.update_historical_data(df, freq=freq, out_path=str(path), target_tickers=tickers)

    refreshed = _momentum_store.refresh(updater=_updater)
    tickers_price_data = refreshed.data.drop_duplicates()

    print(f"Using momentum params from {params_source}")
    ticker_symbol_map = {ticker: ticker for ticker in optimized_parameters.keys()}
    successfully_fetched_tickers_list = tickers_price_data.columns.tolist()
    runtime_config = _momentum_runtime_config(momentum_section)
    if settings.momentum_data_frequency == "auto":
        runtime_config["data_frequency"] = infer_frequency(tickers_price_data.index)
    data_frequency = runtime_config["data_frequency"]
    return (
        tickers,
        tickers_price_data,
        successfully_fetched_tickers_list,
        data_frequency,
        optimized_parameters,
        ticker_symbol_map,
        runtime_config,
    )


def _strip_usdt_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Name"] = out["Name"].astype(str).str.removesuffix("USDT")
    return out


def _top_len(n_side: int, tickers_len: int, *, large_frac: float, small_frac: float) -> int:
    if n_side > 0.5 * tickers_len:
        return int(n_side * large_frac)
    return int(n_side * small_frac)


def _split_ls(df: pd.DataFrame, *, side_col: str, long_value: str, short_value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    shorts = df[df[side_col] == short_value]
    longs = df[df[side_col] == long_value]
    return longs, shorts


def _safe_format_multi_text(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_title: str = "LONG SIGNAL",
    right_title: str = "SHORT SIGNAL",
) -> str:
    if (left_df is None or left_df.empty) and (right_df is None or right_df.empty):
        return "No signals available."

    columns: list[str] = []
    if left_df is not None and not left_df.empty:
        columns = list(left_df.columns)
    elif right_df is not None and not right_df.empty:
        columns = list(right_df.columns)
    if not columns:
        columns = ["Name", "Signal", "SR"]

    def _placeholder(label: str) -> pd.DataFrame:
        row = {col: "n/a" for col in columns}
        row[columns[0]] = label
        return pd.DataFrame([row], columns=columns)

    if left_df is None or left_df.empty:
        left_df = _placeholder("NO LONGS")
    if right_df is None or right_df.empty:
        right_df = _placeholder("NO SHORTS")

    return momo.format_multi_text(left_df, right_df, left_title, right_title)


def _plain_pre_text(formatted_text: str) -> str:
    return (
        formatted_text.replace("</pre>  <pre>", "\n\n")
        .replace("</pre><pre>", "\n\n")
        .replace("<pre>", "")
        .replace("</pre>", "")
    )


def _split_lines_for_telegram(text: str, max_chars: int = 3800) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines():
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        if line_len > max_chars:
            for start in range(0, len(line), max_chars):
                chunks.append(line[start : start + max_chars])
            continue
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks or [""]


async def _reply_table(update: Update, table_text: str) -> None:
    if len(table_text) <= 3900:
        await update.message.reply_text(text=table_text, parse_mode="HTML")
        return
    plain_text = _plain_pre_text(table_text)
    for chunk in _split_lines_for_telegram(plain_text):
        await update.message.reply_text(text=f"<pre>{html.escape(chunk)}</pre>", parse_mode="HTML")


def _get_momo_df(*, universe: list[str] | None = None) -> tuple[list[str], pd.DataFrame]:
    tickers, tickers_price_data, _, data_frequency, optimized_parameters, ticker_symbol_map, runtime_config = _load_momentum_context()
    missing_params = sorted(set(tickers_price_data.columns) - set(optimized_parameters.keys()))
    if missing_params:
        print(f"{len(missing_params)} stored momentum tickers have no optimized params yet; reoptimize to activate them.")
    momo_df = cta.carver_gen_signal(
        price_frame=tickers_price_data,
        tickers=list(optimized_parameters.keys()),
        ticker_dict=ticker_symbol_map,
        optimized_inputs_dict=optimized_parameters,
        ewmac_factors=runtime_config["ewmac_factors"],
        lookback=None,
        pnl_dict_gen=False,
        diversification_multiplier=runtime_config["diversification_multiplier"],
        forecast_scalars=runtime_config["forecast_scalars"],
        cap_final_forecast=runtime_config["cap_final_forecast"],
        data_frequency=data_frequency,
        portfolio_value=settings.portfolio_value,
        target_vol_annual=settings.target_vol_annual,
        max_leverage=settings.max_leverage,
    )
    if universe is not None:
        momo_df = momo_df.loc[momo_df["Name"].isin(universe)].copy()
    return tickers, momo_df


def _get_breakout_df(
    *,
    universe: list[str] | None = None,
    pairs: bool = False,
    opt_params_override: dict | None = None,
) -> pd.DataFrame:
    price_data, opt_params, ticker_map, freq, breakout_rules, runtime_config = load_data_breakout()
    params = opt_params_override if opt_params_override is not None else opt_params
    missing_params = sorted(set(price_data.columns) - set(params.keys()))
    if missing_params and not pairs:
        print(f"{len(missing_params)} stored breakout tickers have no optimized params yet; reoptimize to activate them.")
    breakout_df = (
        run_breakout_curr_pairs(price_data, freq, params, ticker_map, breakout_rules, runtime_config)
        if pairs
        else run_breakout_curr_sig(price_data, freq, params, ticker_map, breakout_rules, runtime_config)
    )
    if universe is not None:
        breakout_df = breakout_df.loc[breakout_df["Name"].isin(universe)].copy()
    return breakout_df


# --- Momentum Handlers (Unchanged) ---
# Get L/S Signals for past 7 days
async def curr_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.signals_command(update, context, sort_by="sr")
        return
    await update.message.reply_text("Generating Trend signals for ALL...")
    print('STARTING.....')
    print('GETTING MOMS.....')
    tickers, momo_df = _get_momo_df()
    momo_df = _strip_usdt_name(momo_df)

    longs, shorts = _split_ls(momo_df, side_col="Position", long_value="LONG", short_value="SHORT")
    shorts = shorts.sort_values(by='SR', ascending=False)
    longs = longs.sort_values(by='SR', ascending=False)

    # Get top 25% of the signal
    s_len = _top_len(len(shorts), len(tickers), large_frac=0.25, small_frac=0.5)
    l_len = _top_len(len(longs), len(tickers), large_frac=0.25, small_frac=0.5)

    shorts_top = shorts.iloc[: s_len, :]
    longs_top =  longs.iloc[:l_len, :]
    table_text = _safe_format_multi_text(longs_top, shorts_top)
    print(len(table_text))    
    await _reply_table(update, table_text)

async def curr_mom_str(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.signals_command(update, context, sort_by="strength")
        return
    await update.message.reply_text("Generating Trend signals for ALL...")
    print('STARTING.....')
    print('GETTING MOMS.....')
    tickers, momo_df = _get_momo_df()
    momo_df = _strip_usdt_name(momo_df)

    longs, shorts = _split_ls(momo_df, side_col="Position", long_value="LONG", short_value="SHORT")
    shorts = shorts.sort_values(by='SR', ascending=False)
    longs = longs.sort_values(by='SR', ascending=False)

    # Get top 25% of the signal
    s_len = _top_len(len(shorts), len(tickers), large_frac=0.25, small_frac=0.5)
    l_len = _top_len(len(longs), len(tickers), large_frac=0.25, small_frac=0.5)

    shorts_top = shorts.iloc[: s_len, :]
    shorts_top = shorts_top.sort_values(by='Signal', ascending=True)
    
    longs_top =  longs.iloc[:l_len, :]
    longs_top = longs_top.sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs_top, shorts_top)
    print(len(table_text))
    await _reply_table(update, table_text)

async def l1_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.signals_command(update, context, sort_by="strength", sector="l1")
        return
    await update.message.reply_text("Generating Trend signals for L1...")
    print('STARTING.....')
    print('GETTING MOMS.....')
    _, momo_df = _get_momo_df(universe=L1)
    momo_df = _strip_usdt_name(momo_df)

    longs, shorts = _split_ls(momo_df, side_col="Position", long_value="LONG", short_value="SHORT")
    shorts = shorts.sort_values(by='SR', ascending=False)
    longs = longs.sort_values(by='SR', ascending=False)

    shorts = shorts.sort_values(by='Signal', ascending=True)
    longs = longs.sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts)
    print(len(table_text))
    await _reply_table(update, table_text)

async def eth_beta_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.signals_command(update, context, sort_by="strength", sector="eth_beta")
        return
    await update.message.reply_text("Generating Trend signals for ETH BETA...")
    print('STARTING.....')
    print('GETTING MOMS.....')
    _, momo_df = _get_momo_df(universe=ETH_BETA)
    momo_df = _strip_usdt_name(momo_df)

    longs, shorts = _split_ls(momo_df, side_col="Position", long_value="LONG", short_value="SHORT")
    shorts = shorts.sort_values(by='SR', ascending=False)
    longs = longs.sort_values(by='SR', ascending=False)

    shorts = shorts.sort_values(by='Signal', ascending=True)
    longs = longs.sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts)
    print(len(table_text))
    await _reply_table(update, table_text)

async def sol_beta_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.signals_command(update, context, sort_by="strength", sector="sol_beta")
        return
    await update.message.reply_text("Generating Trend signals for SOL BETA...")
    print('STARTING.....')
    print('GETTING MOMS.....')
    _, momo_df = _get_momo_df(universe=SOL_BETA)
    momo_df = _strip_usdt_name(momo_df)

    longs, shorts = _split_ls(momo_df, side_col="Position", long_value="LONG", short_value="SHORT")
    shorts = shorts.sort_values(by='SR', ascending=False)
    longs = longs.sort_values(by='SR', ascending=False)

    shorts = shorts.sort_values(by='Signal', ascending=True)
    longs = longs.sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts)
    print(len(table_text))
    await _reply_table(update, table_text)

async def meme_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.signals_command(update, context, sort_by="strength", sector="meme")
        return
    await update.message.reply_text("Generating Trend signals for memes...")
    print('STARTING.....')
    print('GETTING MOMS.....')
    _, momo_df = _get_momo_df(universe=MEME)
    momo_df = _strip_usdt_name(momo_df)

    longs, shorts = _split_ls(momo_df, side_col="Position", long_value="LONG", short_value="SHORT")
    shorts = shorts.sort_values(by='SR', ascending=False)
    longs = longs.sort_values(by='SR', ascending=False)

    shorts = shorts.sort_values(by='Signal', ascending=True)
    longs = longs.sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts)
    print(len(table_text))
    await _reply_table(update, table_text)

# Get L/S Signals for past 7 days
async def curr_mom_top100(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.signals_command(update, context, sort_by="sr")
        return
    await update.message.reply_text("Generating Trend signals for top 100 (sorted by SR)...")
    print('STARTING.....')
    print('GETTING MOMS.....')
    tickers, momo_df = _get_momo_df(universe=top_100_mcap)
    momo_df = _strip_usdt_name(momo_df)

    longs, shorts = _split_ls(momo_df, side_col="Position", long_value="LONG", short_value="SHORT")
    shorts = shorts.sort_values(by='SR', ascending=False)
    longs = longs.sort_values(by='SR', ascending=False)

    # Get top 25% of the signal
    s_len = _top_len(len(shorts), len(tickers), large_frac=0.3, small_frac=0.5)
    l_len = _top_len(len(longs), len(tickers), large_frac=0.3, small_frac=0.5)

    shorts_top = shorts.iloc[: s_len, :]
    longs_top =  longs.iloc[:l_len, :]
    table_text = _safe_format_multi_text(longs_top, shorts_top)
    print(len(table_text))    
    await _reply_table(update, table_text)

async def curr_mom_str_top100(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.signals_command(update, context, sort_by="strength")
        return
    await update.message.reply_text("Generating Trend signals for top 100 (sorted by Strength)...")
    print('STARTING.....')
    print('GETTING MOMS.....')
    tickers, momo_df = _get_momo_df(universe=top_100_mcap)
    momo_df = _strip_usdt_name(momo_df)

    longs, shorts = _split_ls(momo_df, side_col="Position", long_value="LONG", short_value="SHORT")
    shorts = shorts.sort_values(by='SR', ascending=False)
    longs = longs.sort_values(by='SR', ascending=False)

    # Get top 25% of the signal
    s_len = _top_len(len(shorts), len(tickers), large_frac=0.25, small_frac=0.5)
    l_len = _top_len(len(longs), len(tickers), large_frac=0.25, small_frac=0.5)


    shorts_top = shorts.iloc[: s_len, :]
    shorts_top = shorts_top.sort_values(by='Signal', ascending=True)
    
    longs_top =  longs.iloc[:l_len, :]
    longs_top = longs_top.sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs_top, shorts_top)
    print(len(table_text))
    await _reply_table(update, table_text)

# Get Curr Position Momentum
async def port_mommy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.account_momentum_command(update, context)
        return
    await update.message.reply_text("Generating Position Trend Signals ...")
    print('GETTING MOMS.....')
    _tickers, momo_df = _get_momo_df()
    momo_df = momo.get_position_trend(momo_df)
    shorts = momo_df[momo_df['PORT DIR'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['PORT DIR'] == 'LONG'].sort_values(by='SR', ascending=False)

    table_text = _safe_format_multi_text(longs, shorts)
    print(len(table_text))    
    await _reply_table(update, table_text)

from telegram import Update
from telegram.ext import ContextTypes

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /chart command. Generates and sends PnL and Signal charts for a given ticker.
    Usage: /chart BTCUSDT
    """
    if not xsec_telegram.wants_legacy(context.args):
        await xsec_telegram.ticker_chart_command(update, context)
        return
    try:
        period, days = xsec_telegram.parse_lookback(context.args, default="all")
        ticker_args = [
            arg for arg in context.args
            if str(arg).lower() != "legacy"
            and xsec_telegram.lookback_token(str(arg)) is None
        ]
        # Check if the user provided a ticker
        if not ticker_args:
            await update.message.reply_text("Please provide a ticker. Usage: /mc BTC [lookback] legacy")
            return

        ticker = ticker_args[0].upper()
        if not ticker.endswith("USDT"):
            ticker += "USDT"
        await update.message.reply_text(f"Generating charts for {ticker}...")

        # --- 1. Load Data ---
        # This part should match your existing bot structure
        tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map, runtime_config = _load_momentum_context()

        
        # For demonstration, we'll assume these are loaded
        if ticker not in optimized_parameters or ticker not in tickers_price_data.columns:
            await update.message.reply_text(f"Sorry, I don't have data or optimized parameters for {ticker}.")
            return
            
        # --- 2. Generate PnL and Signal Data ---
        # We need both dicts, so we call the signal generator with pnl_dict_gen=True
        pnl_dict, signal_dict = cta.carver_gen_signal(
            price_frame=tickers_price_data,
            tickers=[ticker], # Only process the requested ticker
            ticker_dict=ticker_symbol_map,
            optimized_inputs_dict=optimized_parameters,
            ewmac_factors=runtime_config["ewmac_factors"],
            lookback=None,
            pnl_dict_gen=True,
            diversification_multiplier=runtime_config["diversification_multiplier"],
            forecast_scalars=runtime_config["forecast_scalars"],
            cap_final_forecast=runtime_config["cap_final_forecast"],
            data_frequency=data_frequency,
            portfolio_value=settings.portfolio_value,
            target_vol_annual=settings.target_vol_annual,
            max_leverage=settings.max_leverage,
        )

        if not pnl_dict or not signal_dict or not signal_dict[ticker]:
            await update.message.reply_text(f"Could not generate signal/PnL for {ticker}.")
            return

        # --- 3. Prepare Data and Generate Chart Image ---
        pnl_series = pnl_dict[ticker][3].fillna(0.0).cumsum() # Use the 4th element (base_ccy_returns)
        signal_df = signal_dict[ticker][0]
        
        # Add price data to the signal DataFrame for plotting
        signal_df['price'] = tickers_price_data[ticker].reindex(signal_df.index, method='ffill')
        if days is not None:
            cutoff = signal_df.index.max() - pd.Timedelta(days=days - 1)
            signal_df = signal_df.loc[signal_df.index >= cutoff]
            pnl_series = pnl_series.loc[pnl_series.index >= cutoff]
        if not pnl_series.empty:
            pnl_series = pnl_series - pnl_series.iloc[0]
        
        # Build a robust path relative to the script's location
        charts_folder = os.path.join(BASE_DIR, "momo_charts")
        chart_filename = os.path.join(charts_folder, f"{ticker}_chart.png")
        
        # Optional: Create the directory if it doesn't exist
        os.makedirs(charts_folder, exist_ok=True)
        momo.generate_chart_image(
            pnl_series,
            signal_df,
            ticker,
            lookback=max(len(signal_df), 1),
            output_filename=chart_filename,
        )
        
        # --- 4. Send the Chart Image ---
        await update.message.reply_photo(photo=open(chart_filename, 'rb'))

    except Exception as e:
        print(f"Error in /chart command: {e}")
        await update.message.reply_text("An error occurred while generating the chart.")



# --- NEW: Breakout Handlers ---
### --- NEW: BREAKOUT HANDLERS --- ###
async def breakout_sig_sr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals (sorted by SR)...")
    breakout_df = _get_breakout_df()
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='SR', ascending=False)
    table_text = _safe_format_multi_text(longs.head(15), shorts.head(15), "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

async def breakout_sig_str(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals (sorted by Strength)...")
    breakout_df = _get_breakout_df()
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='Signal', ascending=True)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs.head(15), shorts.head(15), "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

async def l1_breakout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals for L1...")
    breakout_df = _get_breakout_df(universe=L1)
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='Signal', ascending=True)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts, "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

async def eth_beta_breakout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals for ETH BETA...")
    breakout_df = _get_breakout_df(universe=ETH_BETA)
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='Signal', ascending=True)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts, "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

async def sol_beta_breakout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals for SOL BETA...")
    breakout_df = _get_breakout_df(universe=SOL_BETA)
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='Signal', ascending=True)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts, "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

async def meme_breakout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals for MEMES...")
    breakout_df = _get_breakout_df(universe=MEME)
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='Signal', ascending=True)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts, "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

async def breakout_top100(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals for top 100 (sorted by SR)...")
    breakout_df = _get_breakout_df(universe=top_100_mcap)
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='SR', ascending=False)
    table_text = _safe_format_multi_text(longs.head(30), shorts.head(30), "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

async def breakout_str_top100(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals for top 100 (sorted by Strength)...")
    breakout_df = _get_breakout_df(universe=top_100_mcap)
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='Signal', ascending=True)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs.head(30), shorts.head(30), "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

async def breakout_str_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Breakout signals for pairs (sorted by Strength)...")
    if not settings.breakout_pairs_params_path or not settings.breakout_pairs_params_path.exists():
        await update.message.reply_text(
            "Pairs params not configured. Set MOMO_BREAKOUT_PAIRS_PARAMS_PATH (see .env.example)."
        )
        return
    opt_pair_ams = pd.read_pickle(settings.breakout_pairs_params_path)
    breakout_df = _get_breakout_df(pairs=True, opt_params_override=opt_pair_ams)
    # breakout_df = breakout_df.loc[breakout_df['Name'].isin(top_100_mcap)]
    breakout_df['Name'] = breakout_df['Name'].str.removesuffix('USDT')
    shorts = breakout_df[breakout_df['Direction'] == 'SHORT'].sort_values(by='Signal', ascending=True)
    longs = breakout_df[breakout_df['Direction'] == 'LONG'].sort_values(by='Signal', ascending=False)
    table_text = _safe_format_multi_text(longs.head(30), shorts.head(30), "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)


async def port_breakout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating Position Breakout Signals...")
    breakout_df = _get_breakout_df()
    print(breakout_df.columns)
    breakout_df = get_position_break(breakout_df) # Assuming this function is generic enough
    
    shorts = breakout_df[breakout_df['PORT DIR'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = breakout_df[breakout_df['PORT DIR'] == 'LONG'].sort_values(by='SR', ascending=False)
    table_text = _safe_format_multi_text(longs, shorts, "BREAKOUT LONGS", "BREAKOUT SHORTS")
    await _reply_table(update, table_text)

# --- NEW: Charting Command for Breakout Strategy ---
async def chart_breakout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /chart_breakout command. Generates charts for the breakout strategy.
    Usage: /chart_breakout BTCUSDT (or /chart_breakout BTC)
    """
    try:
        period, days = xsec_telegram.parse_lookback(context.args, default="60d")
        ticker_args = [
            arg for arg in context.args
            if xsec_telegram.lookback_token(str(arg)) is None
        ]
        if not ticker_args:
            await update.message.reply_text("Please provide a ticker. Usage: /bc BTC [lookback]")
            return
        ticker = ticker_args[0].upper()
        if not ticker.endswith("USDT"):
            ticker = f"{ticker}USDT"
        await update.message.reply_text(f"Generating Breakout charts for {ticker}...")

        price_data, opt_params, ticker_map, freq, breakout_rules, runtime_config = load_data_breakout()

        if ticker not in opt_params or ticker not in price_data.columns:
            await update.message.reply_text(f"Sorry, I don't have data or parameters for {ticker}.")
            return
            
        pnl_dict, signal_dict, _ = cta.carver_gen_signal_unified(
            price_frame=price_data, tickers=[ticker], ticker_dict=ticker_map,
            optimized_inputs_dict=opt_params, rule_variations=breakout_rules,
            strategy_type='breakout', data_frequency=freq, pnl_dict_gen=True,
            portfolio_value=settings.portfolio_value,
            target_vol_annual=settings.target_vol_annual,
            max_leverage=settings.max_leverage,
            vol_lookback_for_position_sizing=runtime_config.get("vol_lookback_for_position_sizing", 180),
            forecast_scalars=runtime_config.get("forecast_scalars"),
            diversification_multiplier=runtime_config.get("diversification_multiplier"),
            cap_final_forecast=runtime_config.get("cap_final_forecast", True),
        )

        pnl_series = pnl_dict[ticker]['pnl_data'][3].fillna(0.0).cumsum()
        signal_df = signal_dict[ticker]
        signal_df['price'] = price_data[ticker].reindex(signal_df.index, method='ffill')
        if days is not None:
            cutoff = signal_df.index.max() - pd.Timedelta(days=days - 1)
            signal_df = signal_df.loc[signal_df.index >= cutoff]
            pnl_series = pnl_series.loc[pnl_series.index >= cutoff]
        
        # fig = generate_plotly_chart(pnl_series, signal_df, ticker)
        fig = generate_matplotlib_chart(
            pnl_series=pnl_series,
            signal_df=signal_df,
            ticker=ticker,
            lookback=max(len(signal_df), 1),
        )


        # --- FIX: Save the Plotly figure as a PNG image ---
        charts_folder = os.path.join(BASE_DIR, "charts")
        os.makedirs(charts_folder, exist_ok=True)
        chart_filename = os.path.join(charts_folder, f"{ticker}_chart.png")
        
        # You need the 'kaleido' package installed for this to work: pip install kaleido
        # fig.write_image(chart_filename,  width=900, height=1000, scale=2)
        fig.savefig(chart_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig) # IMPORTANT: Close the figure to prevent memory leaks in your bot

        
        # --- Send the saved PNG image file ---
        await update.message.reply_photo(photo=open(chart_filename, 'rb'))
    except Exception as e:
        print(f"Error in /chart_breakout command: {e}")
        await update.message.reply_text("An error occurred while generating the breakout chart.")


async def _handle_command_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled Telegram command error", exc_info=context.error)
    message = getattr(update, "effective_message", None) if update is not None else None
    if message is None:
        return
    try:
        await message.reply_text("Command failed before output. Check the bot logs for details.")
    except Exception:
        logger.exception("Could not send Telegram command error message.")


def _store_interval(store: MarketDataStore) -> str:
    if store.freq != "auto":
        return store.freq
    df = store.load()
    return infer_frequency(df.index)


async def _start_market_data_streams(application: Application) -> None:
    try:
        await application.bot.set_my_commands(
            [
                BotCommand("mom_sig_str", "XSec momentum by signal strength"),
                BotCommand("mom_sig_sr", "Momentum by standalone Sharpe"),
                BotCommand("mc", "Ticker return, signal and rank charts"),
                BotCommand("portfolio", "Target weights, notionals and quantities"),
                BotCommand("rebalance", "Read-only model rebalance sheet"),
                BotCommand("risk", "Exposure and realized-volatility analytics"),
                BotCommand("performance", "Portfolio returns vs an asset"),
                BotCommand("distribution", "Cross-sectional signal percentiles"),
                BotCommand("changes", "Largest daily signal and target changes"),
                BotCommand("portfolio_momo", "Assess current Binance positions"),
                BotCommand("model", "Pinned model parameters and lineage"),
                BotCommand("health", "Snapshot freshness and universe status"),
                BotCommand("manual", "Command and lookback guide"),
            ]
        )
    except Exception:
        logger.exception("Could not publish Telegram command descriptions.")

    if settings.market_data_mode != "websocket_closed_candles":
        return

    tasks: list[asyncio.Task] = []
    try:
        momentum_params, _, _ = _load_latest_strategy_params("momentum")
        if momentum_params:
            tasks.append(
                asyncio.create_task(
                    stream_closed_klines(
                        momentum_params.keys(),
                        _store_interval(_momentum_store),
                        _momentum_store,
                        shard_size=settings.binance_ws_shard_size,
                    )
                )
            )
    except Exception:
        logger.exception("Could not start momentum Binance WebSocket stream.")

    try:
        breakout_params, _, _ = _load_latest_strategy_params("breakout")
        if breakout_params and _breakout_store is not None:
            tasks.append(
                asyncio.create_task(
                    stream_closed_klines(
                        breakout_params.keys(),
                        _store_interval(_breakout_store),
                        _breakout_store,
                        shard_size=settings.binance_ws_shard_size,
                    )
                )
            )
    except Exception:
        logger.exception("Could not start breakout Binance WebSocket stream.")

    application.bot_data["market_data_stream_tasks"] = tasks
    if tasks:
        logger.info("Started %d Binance closed-candle WebSocket stream task(s).", len(tasks))


async def _stop_market_data_streams(application: Application) -> None:
    tasks = application.bot_data.get("market_data_stream_tasks", [])
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Stopped Binance closed-candle WebSocket stream tasks.")


def _parse_bare_command(text: str | None) -> tuple[str, list[str]] | None:
    if not text:
        return None
    parts = text.strip().split()
    if not parts or parts[0].startswith("/"):
        return None
    return parts[0].lower(), parts[1:]


async def _dispatch_bare_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message is None:
        return
    parsed = _parse_bare_command(message.text)
    if parsed is None:
        return
    command, args = parsed
    handler = context.application.bot_data.get("bare_command_handlers", {}).get(command)
    if handler is None:
        return
    context.args = args
    await handler(update, context)


# ==============================================================================
# MAIN BOT APPLICATION
# ==============================================================================
def main():
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set (see .env.example).")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    application_builder = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(_start_market_data_streams)
        .post_shutdown(_stop_market_data_streams)
    )
    if not settings.telegram_verify_ssl:
        request_kwargs = {"httpx_kwargs": {"verify": False}}
        application_builder = application_builder.request(
            HTTPXRequest(**request_kwargs)
        ).get_updates_request(HTTPXRequest(**request_kwargs))
    application = application_builder.build()
    application.add_error_handler(_handle_command_error)
    
    # --- Momentum Handlers (Unchanged) ---
    mom_handler =  CommandHandler('mom_sig_sr', curr_mom)
    mom_handler_str =  CommandHandler('mom_sig_str', curr_mom_str)
    mom_handler_100 =  CommandHandler('mom_sr_100', curr_mom_top100)
    mom_handler_str_100 =  CommandHandler('mom_str_100', curr_mom_str_top100)
    l1_mom_handler = CommandHandler('mom_l1', l1_mom)
    eth_beta_mom_handler = CommandHandler('mom_eth_beta', eth_beta_mom)
    sol_beta_mom_handler = CommandHandler('mom_sol_beta', sol_beta_mom)
    meme_mom_handler = CommandHandler('mom_meme', meme_mom)
    # short_mom_handler =  CommandHandler('mom_sig_short', short_mom)
    port_mom_handler = CommandHandler('portfolio_momo', port_mommy)
    chart_handler = CommandHandler('mc', chart_command)

    application.add_handler(mom_handler)
    application.add_handler(mom_handler_str)
    application.add_handler(mom_handler_100)
    application.add_handler(mom_handler_str_100)
    application.add_handler(l1_mom_handler)
    application.add_handler(eth_beta_mom_handler)
    application.add_handler(sol_beta_mom_handler)
    application.add_handler(meme_mom_handler)
    # application.add_handler(short_mom_handler)
    application.add_handler(port_mom_handler)
    application.add_handler(chart_handler)
    application.add_handler(CommandHandler('model', xsec_telegram.model_command))
    application.add_handler(CommandHandler('portfolio', xsec_telegram.portfolio_command))
    application.add_handler(CommandHandler('rebalance', xsec_telegram.rebalance_command))
    application.add_handler(CommandHandler('risk', xsec_telegram.risk_command))
    application.add_handler(CommandHandler('performance', xsec_telegram.performance_command))
    application.add_handler(CommandHandler('distribution', xsec_telegram.distribution_command))
    application.add_handler(CommandHandler('changes', xsec_telegram.changes_command))
    application.add_handler(CommandHandler('health', xsec_telegram.health_command))
    application.add_handler(CommandHandler('manual', xsec_telegram.manual_command))
    application.add_handler(CommandHandler('help', xsec_telegram.manual_command))


    # --- Breakout Handlers ---
    breakout_params, _, _ = _load_latest_strategy_params("breakout")
    breakout_ready = bool(settings.breakout_price_data_path and settings.breakout_price_data_path.exists() and breakout_params)
    # Only register breakout commands if configured; otherwise keep bot startup healthy.
    if breakout_ready:
        application.add_handler(CommandHandler('breakout_sig_sr', breakout_sig_sr))
        application.add_handler(CommandHandler('breakout_sig_str', breakout_sig_str))
        application.add_handler(CommandHandler('breakout_l1', l1_breakout))
        application.add_handler(CommandHandler('breakout_eth_beta', eth_beta_breakout))
        application.add_handler(CommandHandler('breakout_sol_beta', sol_beta_breakout))
        application.add_handler(CommandHandler('breakout_meme', meme_breakout))
        application.add_handler(CommandHandler('breakout_sr_100', breakout_top100))
        application.add_handler(CommandHandler('breakout_str_100', breakout_str_top100))
        application.add_handler(CommandHandler('breakout_pairs', breakout_str_pairs))
        application.add_handler(CommandHandler('portfolio_breakout', port_breakout))
        application.add_handler(CommandHandler('chart_breakout', chart_breakout))
        application.add_handler(CommandHandler('bc', chart_breakout))
    else:
        print("Breakout commands disabled (missing breakout data or params).")

    bare_command_handlers = {
        "mom_sig_sr": curr_mom,
        "mom_sig_str": curr_mom_str,
        "mom_sr_100": curr_mom_top100,
        "mom_str_100": curr_mom_str_top100,
        "mom_l1": l1_mom,
        "mom_eth_beta": eth_beta_mom,
        "mom_sol_beta": sol_beta_mom,
        "mom_meme": meme_mom,
        "portfolio_momo": port_mommy,
        "mc": chart_command,
        "model": xsec_telegram.model_command,
        "portfolio": xsec_telegram.portfolio_command,
        "rebalance": xsec_telegram.rebalance_command,
        "risk": xsec_telegram.risk_command,
        "performance": xsec_telegram.performance_command,
        "distribution": xsec_telegram.distribution_command,
        "changes": xsec_telegram.changes_command,
        "health": xsec_telegram.health_command,
        "manual": xsec_telegram.manual_command,
        "help": xsec_telegram.manual_command,
    }
    if breakout_ready:
        bare_command_handlers.update(
            {
                "breakout_sig_sr": breakout_sig_sr,
                "breakout_sig_str": breakout_sig_str,
                "breakout_l1": l1_breakout,
                "breakout_eth_beta": eth_beta_breakout,
                "breakout_sol_beta": sol_beta_breakout,
                "breakout_meme": meme_breakout,
                "breakout_sr_100": breakout_top100,
                "breakout_str_100": breakout_str_top100,
                "breakout_pairs": breakout_str_pairs,
                "portfolio_breakout": port_breakout,
                "chart_breakout": chart_breakout,
                "bc": chart_breakout,
            }
        )
    application.bot_data["bare_command_handlers"] = bare_command_handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _dispatch_bare_command))

    print("Bot is running with both Momentum and Breakout commands...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
