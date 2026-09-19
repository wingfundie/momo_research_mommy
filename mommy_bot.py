# display(HTML("<style>.container { width:100% !important; }</style>"))
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, timezone
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, Application
from telegram.ext import CallbackContext
from telegram.request import HTTPXRequest
import asyncio 
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
import urllib3
from pathlib import Path
from binance.exceptions import BinanceAPIException
import portfolio_strategy as cta
import trading_bot as pos
import os
from momo_bot.config import settings
from momo_bot.binance_rate_limit import (
    BinanceRestCircuitOpen,
    get_rest_guard,
    is_rate_limit_error,
)
from momo_bot.candles import check_freshness, expected_last_complete_open_time, normalize_timestamp
from momo_bot.data_validation import (
    infer_frequency,
    missing_coverage_dates,
    resolve_coverage_starts,
    validate_price_coverage,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)
if not settings.binance_verify_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Paths now come from config (defaults to ./data_store)
PRICE_DATA_FILE_PATH = str(settings.momentum_price_data_path)
OPTIMIZED_PARAMS_PATH = str(settings.momentum_params_path)

# Ticker Pulls
def get_all_usdt_perpetual_futures_tickers(api_key=None, api_secret=None):
    """
    Fetches a list of all actively trading USDT-margined perpetual futures tickers
    from Binance.

    Args:
        api_key (str, optional): Your Binance API key.
        api_secret (str, optional): Your Binance API secret.

    Returns:
        list: A list of ticker symbols (e.g., ['BTCUSDT', 'ETHUSDT', ...])
              or an empty list if an error occurs.
    """
    return cta.get_all_usdt_perpetual_futures_tickers(api_key=api_key, api_secret=api_secret)

# Define a function to format the DataFrame as a monospaced table
def format_as_monospaced_text(df):
    # Calculate the maximum width of each column
    col_widths = [max(map(len, df[col].astype(str))) for col in df.columns]
    # Create the header row
    header_row = " | ".join(f"{col:{w}}" for col, w in zip(df.columns, col_widths))
    # Create the separator row
    separator_row = "-+-".join("-" * w for w in col_widths)
    # Create the data rows
    data_rows = [" | ".join(f"{str(value):<{w}}" for value, w in zip(row, col_widths)) for row in df.values]
    # Combine all rows into a single string with line breaks
    table = "\n".join([header_row, separator_row] + data_rows)
    return f"<pre>{table}</pre>"


# Format Multiple DFS
def format_multi_text(left_df, right_df, left_title="LONG SIGNAL", right_title="SHORT SIGNAL"):
    # Calculate the maximum width of each column for both DataFrames
    col_widths_left = [max(map(len, left_df[col].astype(str))) for col in left_df.columns]
    col_widths_right = [max(map(len, right_df[col].astype(str))) for col in right_df.columns]
    
    # Create the header row for both DataFrames
    header_row_left = " | ".join(f"{col:{w}}" for col, w in zip(left_df.columns, col_widths_left))
    header_row_right = " | ".join(f"{col:{w}}" for col, w in zip(right_df.columns, col_widths_right))
    
    # Create the separator row for both DataFrames
    separator_row_left = "-+-".join("-" * w for w in col_widths_left)
    separator_row_right = "-+-".join("-" * w for w in col_widths_right)
    
    # Create the data rows for both DataFrames
    data_rows_left = [" | ".join(f"{str(value):<{w}}" for value, w in zip(row, col_widths_left)) for row in left_df.values]
    data_rows_right = [" | ".join(f"{str(value):<{w}}" for value, w in zip(row, col_widths_right)) for row in right_df.values]
    
    # Combine all rows into single strings with line breaks for both DataFrames
    table_left = "\n".join([left_title, header_row_left, separator_row_left] + data_rows_left)
    table_right = "\n".join([right_title, header_row_right, separator_row_right] + data_rows_right)
    
    # Combine both tables with a space and return as HTML pre-formatted text
    formatted_text = f"<pre>{table_left}</pre>  <pre>{table_right}</pre>"
    return formatted_text


# Crypto Tickers
def new_binance_tickers(start_date, tickers, freq = '1d'):
    # DATES
    end_date = dt.datetime.now().strftime("%Y-%m-%d")

    store = pd.DataFrame()

    for ticker in tickers:
        try:
            
            temp = cta.gen_lookback_df(ticker, start_date, end_date, freq=freq).loc[:, ['Close']]
            temp.columns = [ticker]
            store = pd.concat([store, temp], axis=1, join='outer')
            print(ticker)
        except BinanceAPIException as e:
            if e.code == -1121:
                print(f"Invalid symbol: {ticker}")
                continue  # Skip to the next ticker
            else:
                raise 

    tickers_store = store.loc[:, "BTCUSDT":]

    return tickers_store


def update_historical_data(
    existing_df: pd.DataFrame,
    freq: str = '4h',
    *,
    out_path: str | os.PathLike | None = None,
    target_tickers: list[str] | None = None,
    start_date_for_new: str | None = None,
    keep_inactive_existing: bool | None = None,
    request_sleep_seconds: float = 0.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    require_complete: bool = False,
    signal_tickers: list[str] | None = None,
    metadata_extra: dict[str, object] | None = None,
    expected_start_by_ticker: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """
    Updates historical prices through the latest completed candle and expands the
    stored universe with any new active Binance USDT perpetual symbols.
    """
    if not isinstance(existing_df.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame's index must be a DatetimeIndex.")

    if keep_inactive_existing is None:
        keep_inactive_existing = settings.keep_inactive_tickers
    if start_date_for_new is None:
        start_date_for_new = settings.new_ticker_start_date

    updated_df = existing_df.copy()
    updated_df.index = pd.DatetimeIndex([normalize_timestamp(ts) for ts in updated_df.index])
    updated_df = updated_df.loc[~updated_df.index.duplicated(keep="last")].sort_index()
    if target_tickers is None:
        target_tickers = cta.get_all_usdt_perpetual_futures_tickers(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
        )
    active_tickers = sorted(set(target_tickers or []))
    existing_tickers = set(updated_df.columns.tolist())
    active_set = set(active_tickers)
    new_tickers = sorted(active_set - existing_tickers)
    inactive_existing = sorted(existing_tickers - active_set) if active_set else []
    existing_active = sorted(existing_tickers & active_set) if active_set else sorted(existing_tickers)

    if new_tickers:
        print(f"Adding {len(new_tickers)} new active Binance tickers to storage: {', '.join(new_tickers[:20])}{'...' if len(new_tickers) > 20 else ''}")
    if inactive_existing:
        action = "keeping" if keep_inactive_existing else "dropping"
        print(f"{action.title()} {len(inactive_existing)} inactive existing tickers.")

    tickers_list = existing_active + new_tickers
    if not tickers_list:
        print("Warning: no tickers available to refresh. Returning existing data.")
        return updated_df

    all_new_data: list[pd.DataFrame] = []
    successful_fetch_starts: dict[str, pd.Timestamp] = {}
    failed_tickers: list[str] = []
    required_ts = expected_last_complete_open_time(freq=freq)
    expected_starts = {
        ticker: normalize_timestamp(timestamp)
        for ticker, timestamp in (expected_start_by_ticker or {}).items()
    }
    fetch_starts = dict(expected_starts)
    if out_path:
        existing_metadata_path = Path(out_path).with_suffix(".meta.json")
        if existing_metadata_path.exists():
            try:
                existing_metadata = json.loads(existing_metadata_path.read_text(encoding="utf-8"))
                stored_starts = existing_metadata.get("expected_start_by_ticker", {})
                if isinstance(stored_starts, dict):
                    fetch_starts.update(
                        {
                            ticker: normalize_timestamp(timestamp)
                            for ticker, timestamp in stored_starts.items()
                            if ticker in active_tickers
                        }
                    )
            except (OSError, ValueError, TypeError):
                pass

    def _first_missing_stored_timestamp(
        series: pd.Series,
        expected_start: pd.Timestamp | None,
    ) -> pd.Timestamp | None:
        valid_index = pd.DatetimeIndex(series.dropna().index)
        if expected_start is not None:
            valid_index = valid_index[valid_index >= expected_start]
            if not len(valid_index) or valid_index.min() > expected_start:
                return expected_start
        if len(valid_index) < 2:
            return None

        valid_index = pd.DatetimeIndex([normalize_timestamp(ts) for ts in valid_index]).sort_values()
        first_valid = valid_index.min()
        last_valid = valid_index.max()
        scan_end = min(last_valid, required_ts)
        if first_valid >= scan_end:
            return None

        expected_delta = pd.Timedelta(freq)
        diffs = valid_index.to_series().diff().dropna()
        gap_diffs = diffs[diffs > expected_delta]
        if gap_diffs.empty:
            return None

        gap_end = gap_diffs.index.min()
        gap_end_pos = valid_index.get_loc(gap_end)
        if isinstance(gap_end_pos, slice) or gap_end_pos == 0:
            return None
        gap_start = valid_index[gap_end_pos - 1] + expected_delta
        return normalize_timestamp(gap_start)

    for ticker in tqdm.tqdm(tickers_list, desc="Updating Tickers", file=sys.stdout, leave=True):
        try:
            is_existing = ticker in updated_df.columns
            last_valid_timestamp = updated_df[ticker].last_valid_index() if is_existing else None

            if is_existing and pd.isna(last_valid_timestamp):
                last_valid_timestamp = None

            expected_start = fetch_starts.get(ticker)
            missing_start_timestamp = (
                _first_missing_stored_timestamp(updated_df[ticker], expected_start)
                if is_existing
                else None
            )

            if missing_start_timestamp is not None:
                start_timestamp = missing_start_timestamp
                print(f"Backfilling internal gap for {ticker} from {start_timestamp:%Y-%m-%d %H:%M:%S}")
            elif last_valid_timestamp is not None:
                last_valid_timestamp = normalize_timestamp(last_valid_timestamp)
                if last_valid_timestamp >= required_ts:
                    continue
                start_timestamp = last_valid_timestamp
            else:
                start_timestamp = (
                    expected_start
                    if expected_start is not None
                    else pd.to_datetime(start_date_for_new)
                )

            start_date_str = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")

            new_data_chunk = pd.DataFrame()
            for attempt in range(1, max(1, int(max_retries)) + 1):
                try:
                    new_data_chunk = cta.get_fresh_lookback_df(ticker, start_date_str, freq=freq)
                    break
                except BinanceRestCircuitOpen:
                    raise
                except BinanceAPIException as exc:
                    if is_rate_limit_error(exc):
                        get_rest_guard().record_rate_limit(exc, endpoint="update_historical_data")
                        raise BinanceRestCircuitOpen(str(exc)) from exc
                    if getattr(exc, 'code', None) == -1121 or attempt >= max_retries:
                        raise
                    delay = retry_backoff_seconds * (2 ** (attempt - 1))
                    print(f"Retrying {ticker} after Binance error ({attempt}/{max_retries}) in {delay:.1f}s")
                    time.sleep(delay)

                except Exception:
                    if attempt >= max_retries:
                        raise
                    delay = retry_backoff_seconds * (2 ** (attempt - 1))
                    print(f"Retrying {ticker} after request error ({attempt}/{max_retries}) in {delay:.1f}s")
                    time.sleep(delay)

            successful_fetch_starts[ticker] = normalize_timestamp(start_timestamp)
            
            if not new_data_chunk.empty and 'Close' in new_data_chunk.columns:
                new_data_chunk.index = pd.DatetimeIndex([normalize_timestamp(ts) for ts in new_data_chunk.index])
                new_data_chunk = new_data_chunk.loc[~new_data_chunk.index.duplicated(keep="last")].sort_index()
                if last_valid_timestamp is not None and missing_start_timestamp is None:
                    new_data_chunk = new_data_chunk[new_data_chunk.index > last_valid_timestamp]
                else:
                    new_data_chunk = new_data_chunk[new_data_chunk.index >= start_timestamp]
                new_data_chunk = new_data_chunk[new_data_chunk.index <= required_ts]
                
                if not new_data_chunk.empty:
                    close_prices = new_data_chunk[['Close']].rename(columns={'Close': ticker})
                    all_new_data.append(close_prices)

        except BinanceRestCircuitOpen as e:
            print(f"Binance REST circuit is open while updating {ticker}: {e}")
            raise
        except BinanceAPIException as e:
            failed_tickers.append(ticker)
            if hasattr(e, 'code') and e.code == -1121:
                print(f"Invalid symbol or no new data for: {ticker}. Skipping.")
            else:
                print(f"API Error updating {ticker}: {e}. Skipping.")
        except Exception as e:
            failed_tickers.append(ticker)
            print(f"An unexpected error occurred while updating {ticker}: {e}. Skipping.")
        finally:
            if request_sleep_seconds > 0:
                time.sleep(request_sleep_seconds)

    if all_new_data:
        print("\nCombining all new data chunks...")
        new_data_master_df = pd.concat(all_new_data, axis=1)
        new_data_master_df.index = pd.DatetimeIndex([normalize_timestamp(ts) for ts in new_data_master_df.index])
        new_data_master_df = new_data_master_df.loc[~new_data_master_df.index.duplicated(keep="last")].sort_index()

        print("Merging new data block into existing DataFrame...")
        updated_df = updated_df.combine_first(new_data_master_df)

    if not updated_df.empty:
        updated_df = updated_df.sort_index(ascending=True)
        updated_df = updated_df.loc[:required_ts]
        if active_set:
            final_columns = sorted(existing_tickers | active_set) if keep_inactive_existing else active_tickers
            updated_df = updated_df.reindex(columns=final_columns)

    if out_path is None:
        out_path = PRICE_DATA_FILE_PATH

    coverage_starts = resolve_coverage_starts(
        updated_df,
        required_tickers=active_tickers,
        onboard_date_by_ticker=expected_starts,
    )
    missing_dates = missing_coverage_dates(
        updated_df,
        required_tickers=active_tickers,
        freq=freq,
        required_timestamp=required_ts,
        expected_start_by_ticker=coverage_starts,
    )
    confirmed_unavailable = {
        ticker: [
            timestamp
            for timestamp in ticker_missing
            if ticker in successful_fetch_starts
            and timestamp >= successful_fetch_starts[ticker]
        ]
        for ticker, ticker_missing in missing_dates.items()
    }
    confirmed_unavailable = {
        ticker: dates for ticker, dates in confirmed_unavailable.items() if dates
    }
    coverage = validate_price_coverage(
        updated_df,
        required_tickers=active_tickers,
        freq=freq,
        required_timestamp=required_ts,
        expected_start_by_ticker=coverage_starts,
        allowed_missing_by_ticker=confirmed_unavailable,
    )
    if require_complete and not coverage.is_publishable:
        missing_preview = ", ".join(coverage.missing_latest_tickers[:10])
        raise RuntimeError(
            "Price refresh failed coverage validation; keeping the last good file. "
            f"fresh={coverage.fresh_ticker_count}/{coverage.required_ticker_count}, "
            f"unresolved_internal_cells={coverage.unresolved_internal_cells}, "
            f"missing_latest={missing_preview or 'none'}"
        )

    if out_path:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        try:
            updated_df.to_pickle(temp_path)
            os.replace(temp_path, output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        metadata = {
            "published_at_utc": pd.Timestamp.utcnow().isoformat(),
            "price_path": str(output_path.resolve()),
            "frequency": freq,
            "latest_completed_candle": required_ts.isoformat(),
            "active_ticker_count": len(active_tickers),
            "signal_ticker_count": len(set(signal_tickers or [])),
            "onboard_date_by_ticker": {
                ticker: timestamp.isoformat()
                for ticker, timestamp in sorted(expected_starts.items())
            },
            "expected_start_by_ticker": {
                ticker: timestamp.isoformat()
                for ticker, timestamp in sorted(coverage_starts.items())
            },
            "confirmed_unavailable_by_ticker": {
                ticker: [timestamp.isoformat() for timestamp in dates]
                for ticker, dates in sorted(confirmed_unavailable.items())
            },
            "failed_tickers": sorted(set(failed_tickers)),
            "coverage": coverage.to_dict(),
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        metadata_path = output_path.with_suffix(".meta.json")
        metadata_temp_path = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
        try:
            metadata_temp_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            os.replace(metadata_temp_path, metadata_path)
        finally:
            if metadata_temp_path.exists():
                metadata_temp_path.unlink()
    return updated_df




# Running the process
def load_data():
    tickers_price_data = pd.read_pickle(PRICE_DATA_FILE_PATH)
    successfully_fetched_tickers_list = tickers_price_data.columns.tolist()
    data_frequency = infer_frequency(tickers_price_data.index) if settings.momentum_data_frequency == "auto" else settings.momentum_data_frequency
    optimized_parameters = pd.read_pickle(OPTIMIZED_PARAMS_PATH)
    tickers = sorted(optimized_parameters.keys())
    ticker_symbol_map = {ticker: ticker for ticker in optimized_parameters.keys()}
    return tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map

import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D # Needed for custom legend

def generate_chart_image(
    pnl_series: pd.Series,
    signal_df: pd.DataFrame,
    ticker: str,
    lookback: int = 210,
    output_filename: str = "chart.png"
):
    """
    Generates a PNG image file with two charts: PnL and Signal vs. Price.
    The signal line is purple, with markers indicating zero crossings.
    """
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 10),
        sharex=False,
        gridspec_kw={'height_ratios': [1, 2]} # Give more space to the second chart
    )
    
    # --- Chart 1: Cumulative PnL (Unchanged) ---
    ax1.plot(pnl_series.index, pnl_series, label='Cumulative PnL', color='dodgerblue')
    ax1.set_title(f"{ticker} Cumulative PnL Over Time", fontsize=14)
    ax1.set_ylabel("PnL", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Chart 2: Signal vs. Price ---
    plot_data = signal_df.iloc[-lookback:].dropna(subset=['weighted signal', 'price'])

    # --- FIX: Revert to a single purple line for the signal ---
    signal = plot_data['weighted signal']
    ax2.plot(plot_data.index, signal, label='Weighted Signal', color='purple', zorder=3)

    # --- FIX: Add a prominent zero line ---
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.2, zorder=2)
    
    # --- FIX: Find and plot markers for zero crossings ---
    # Find where the sign of the signal changes
    sign_change = np.sign(signal).diff().ne(0)
    crossing_points = plot_data[sign_change]
    
    # We only want to plot markers where the crossing actually happens, not on the first day
    if not crossing_points.empty and crossing_points.index[0] == plot_data.index[0]:
         crossing_points = crossing_points.iloc[1:]

    # Plot markers at these points on the zero line
    ax2.scatter(
        crossing_points.index, 
        np.zeros(len(crossing_points)), 
        color='blue', 
        s=50,  # size of the marker
        zorder=5, # plot markers on top of everything else
        label='Zero Crossing'
    )
    # --- End of Fixes ---

    ax2.set_title(f"{ticker} Signal vs. Price (Last {lookback} periods)", fontsize=14)
    ax2.set_ylabel("Signal Strength", color='black', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Create a secondary y-axis for the price
    ax2_price = ax2.twinx()
    ax2_price.plot(plot_data.index, plot_data['price'], label='Price', color='darkorange', alpha=0.8)
    ax2_price.set_ylabel("Price", color='black', fontsize=12)
    ax2_price.tick_params(axis='y', labelcolor='black')

    # Combine legends from both axes for the second chart
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_price.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Improve x-axis date formatting
    fig.autofmt_xdate()
    
    plt.tight_layout(pad=2.0)
    
    plt.savefig(output_filename)
    plt.close(fig) # Close the figure to free up memory

# =================== END: Replace your old load_data function with this ===================
# Signal Generation
def get_curr_sigs(new_data, 
                  optimized_parameters, 
                  ticker_symbol_map):
    
    # Get current signals
    cur_sig = cta.gen_signal_ls(new_data,
                                list(optimized_parameters.keys()), # Use keys from optimized_parameters
                                ticker_symbol_map,
                                optimized_parameters,
                                lookback= None,
                                pnl_dict_gen= False)
    
    cur_sig.sort_values(by='SR', ascending=False, inplace=True)
    return cur_sig

# PNL Generation
def get_pnl_dict(new_data, 
                  optimized_parameters, 
                  ticker_symbol_map,
                  sig = False):
    pnl_dict, signal_dict = cta.gen_signal_ls(new_data,
                                        list(optimized_parameters.keys()), # Use keys from optimized_parameters
                                        ticker_symbol_map,
                                        optimized_parameters,
                                        lookback= None,
                                        pnl_dict_gen= True)
    if sig:
        return pnl_dict, signal_dict
    else:
        return pnl_dict


def carv_curr_sigs(new_data,
                  optimized_parameters,
                  ticker_symbol_map,
                  data_frequency: str = '4h'):
    """
    This function now calls the new 'carver_gen_signal' to get the
    DataFrame of current signals and SR.
    """
    # Call the new function. Its parameters map directly from the old call.
    # It returns the summary DataFrame when pnl_dict_gen is False.
    cur_sig = cta.carver_gen_signal(
        price_frame=new_data,
        tickers=list(optimized_parameters.keys()),
        ticker_dict=ticker_symbol_map,
        optimized_inputs_dict=optimized_parameters,
        lookback=None,
        pnl_dict_gen=False,
        data_frequency=data_frequency,
        portfolio_value=settings.portfolio_value,
        target_vol_annual=settings.target_vol_annual,
        max_leverage=settings.max_leverage,
        # Optional parameters like ewmac_factors, diversification_multiplier, etc.,
        # will use the defaults defined in carver_gen_signal.
    )

    cur_sig.sort_values(by='SR', ascending=False, inplace=True)
    return cur_sig


def carv_pnl_dict(new_data,
                 optimized_parameters,
                 ticker_symbol_map,
                 data_frequency: str = '4h',
                 sig=False):
    """
    This function now calls the new 'carver_gen_signal' to generate
    the PnL dictionary.
    """
    # Call the new function. When pnl_dict_gen is True, it returns two values.
    pnl_dict, signal_dict = cta.carver_gen_signal(
        price_frame=new_data,
        tickers=list(optimized_parameters.keys()),
        ticker_dict=ticker_symbol_map,
        optimized_inputs_dict=optimized_parameters,
        lookback=None,
        pnl_dict_gen=True,
        data_frequency=data_frequency,
        portfolio_value=settings.portfolio_value,
        target_vol_annual=settings.target_vol_annual,
        max_leverage=settings.max_leverage,
    )

    # Your original function returned only the pnl_dict
    if sig:
        return pnl_dict, signal_dict
    else: 
        return signal_dict

# Get current MOMO Signals
def run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map, carver = True):
    """
    Generates current momentum signals, automatically updating historical data if it's stale.
    """
    # --- Start of new logic ---

    if data_frequency == "auto":
        data_frequency = infer_frequency(tickers_price_data.index)
    should_update = True
    if not tickers_price_data.empty and isinstance(tickers_price_data.index, pd.DatetimeIndex):
        freshness = check_freshness(tickers_price_data.index, freq=data_frequency)
        if freshness.is_fresh:
            should_update = False
            print("✅ Data is up to date (UTC completed candle check).")
            print(f"   Latest data timestamp: {freshness.latest.strftime('%Y-%m-%d %H:%M:%S') if freshness.latest else 'None'}")
            print(f"   Required timestamp:    {freshness.required.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("⚠️ Data is stale. Update required (UTC completed candle check).")
            print(f"   Latest data timestamp: {freshness.latest.strftime('%Y-%m-%d %H:%M:%S') if freshness.latest else 'None'}")
            print(f"   Required timestamp:    {freshness.required.strftime('%Y-%m-%d %H:%M:%S')}")

    if should_update:
        print(f"Data is stale or empty. Updating historical data...")
        try:
            new_data = update_historical_data(
                tickers_price_data,
                freq=data_frequency,
                out_path=PRICE_DATA_FILE_PATH,
                target_tickers=list(optimized_parameters.keys()),
            )
        except BinanceRestCircuitOpen as exc:
            print(f"Skipping momentum refresh because Binance REST is cooling down: {exc}")
            new_data = tickers_price_data
        print(f'UPDATE: {new_data.index.max()}')
        #new_data.to_pickle(PRICE_DATA_FILE_PATH)
    else:
        new_data = tickers_price_data

    # Ensure we only use fully completed candles for signal generation.
    if not new_data.empty and isinstance(new_data.index, pd.DatetimeIndex):
        required = check_freshness(new_data.index, freq=data_frequency).required
        new_data = new_data.loc[:required]
    if carver:
        curr_sig = carv_curr_sigs(
            new_data.drop_duplicates(),
            optimized_parameters,
            ticker_symbol_map,
            data_frequency=data_frequency,
        )
    else:
        curr_sig = get_curr_sigs(new_data.drop_duplicates(), optimized_parameters, ticker_symbol_map)
    return curr_sig

# Getting current signals for current positioning
def get_position_trend(cur_sig):
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
    print(f'MY POSIT: {my_posit.columns}')
    my_posit.columns = ['COIN', 'MOMO DIR', 'SIGNAL', 'SR', 'PORT DIR', 'NTL']
    my_posit['COIN'] = my_posit['COIN'].str.removesuffix('USDT')
    my_posit['NTL']= (my_posit['NTL'].astype(float))/1000
    my_posit['NTL'] = my_posit['NTL'].apply(lambda x: f"{x:.2f}k")
    my_posit['TREND=FRIEND'] = np.where(my_posit['MOMO DIR'] == my_posit['PORT DIR'], 'YES', 'NO')
    return my_posit


def color_tx(s):
    a = s.copy()
    rng = a.max() - a.min()

    res_b = []
    for i, val in a.items():
        if abs(val) > 50: #a.min() + rng/2:
            res_b.append('color: white')
        else:
            res_b.append('color: black')

    return res_b

def color_tx2(s):
    a = s.copy()
    rng = a.max() - a.min()

    res_b = []
    for i, val in a.items():
        if abs(val) > 0.5: #a.min() + rng/2:
            res_b.append('color: white')
        else:
            res_b.append('color: black')

    return res_b

def b_g(s, cmap, low=0, high=0):
    norm = colors.Normalize(-100,100)
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]

def b_g2(s, cmap, low=0, high=0):
    norm = colors.Normalize(-1,1)
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]

def get_colored_df(df):
    df = df.style.apply(b_g, cmap=ListedColormap(sns.color_palette("RdBu",10).as_hex()), subset=['Signal'])\
        .apply(b_g2, cmap=ListedColormap(sns.color_palette("RdBu",10).as_hex()), subset=['SR'])\
        .apply(color_tx, subset=['Signal'])\
        .apply(color_tx2, subset=['SR'])\
        .format(lambda x: '{:.2f}'.format(x), subset=['SR'])\
        .format(lambda x: '{:.1f}%'.format(x) , subset=['Signal'])\
        .hide()
    return df

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



# Get L/S Signals for past 7 days
async def curr_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    momo_df['Name'] = momo_df['Name'].str.removesuffix('USDT')
    shorts = momo_df[momo_df['Position'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['Position'] == 'LONG'].sort_values(by='SR', ascending=False)

    # Get top 25% of the signal
    if len(shorts)> 0.5 * len(tickers):
        s_len = int(len(shorts)* 0.25)
    else:
        s_len = int(len(shorts)* 0.5)
    
    if len(longs) > 0.5 * len(tickers):
        l_len = int(len(longs)* 0.25)
    else:
        l_len = int(len(longs) * 0.5)

    shorts_top = shorts.iloc[: s_len, :]
    longs_top =  longs.iloc[:l_len, :]
    table_text = format_multi_text(longs_top, shorts_top)
    print(len(table_text))    
    await update.message.reply_text(text= table_text, parse_mode='HTML' )

async def curr_mom_str(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    momo_df['Name'] = momo_df['Name'].str.removesuffix('USDT')
    shorts = momo_df[momo_df['Position'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['Position'] == 'LONG'].sort_values(by='SR', ascending=False)

    # Get top 25% of the signal
    if len(shorts)> 0.5 * len(tickers):
        s_len = int(len(shorts)* 0.25)
    else:
        s_len = int(len(shorts) * 0.5)
    
    if len(longs) > 0.5 * len(tickers):
        l_len = int(len(longs)* 0.25)
    else:
        l_len = int(len(longs) * 0.5)

    shorts_top = shorts.iloc[: s_len, :]
    shorts_top = shorts_top.sort_values(by='Signal', ascending=True)
    
    longs_top =  longs.iloc[:l_len, :]
    longs_top = longs_top.sort_values(by='Signal', ascending=False)
    table_text = format_multi_text(longs_top, shorts_top)
    print(len(table_text))
    await update.message.reply_text(text= table_text, parse_mode='HTML')

async def l1_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    momo_df = momo_df.loc[momo_df['Name'].isin(L1)]
    momo_df['Name'] = momo_df['Name'].str.removesuffix('USDT')
    shorts = momo_df[momo_df['Position'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['Position'] == 'LONG'].sort_values(by='SR', ascending=False)

    shorts = shorts.sort_values(by='Signal', ascending=True)
    longs = longs.sort_values(by='Signal', ascending=False)
    table_text = format_multi_text(longs, shorts)
    print(len(table_text))
    await update.message.reply_text(text= table_text, parse_mode='HTML')

async def eth_beta_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    momo_df = momo_df.loc[momo_df['Name'].isin(ETH_BETA)]
    momo_df['Name'] = momo_df['Name'].str.removesuffix('USDT')
    shorts = momo_df[momo_df['Position'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['Position'] == 'LONG'].sort_values(by='SR', ascending=False)

    shorts = shorts.sort_values(by='Signal', ascending=True)
    longs = longs.sort_values(by='Signal', ascending=False)
    table_text = format_multi_text(longs, shorts)
    print(len(table_text))
    await update.message.reply_text(text= table_text, parse_mode='HTML')

async def sol_beta_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    momo_df = momo_df.loc[momo_df['Name'].isin(SOL_BETA)]
    momo_df['Name'] = momo_df['Name'].str.removesuffix('USDT')
    shorts = momo_df[momo_df['Position'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['Position'] == 'LONG'].sort_values(by='SR', ascending=False)

    shorts = shorts.sort_values(by='Signal', ascending=True)
    longs = longs.sort_values(by='Signal', ascending=False)
    table_text = format_multi_text(longs, shorts)
    print(len(table_text))
    await update.message.reply_text(text= table_text, parse_mode='HTML')

async def meme_mom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    momo_df = momo_df.loc[momo_df['Name'].isin(MEME)]
    momo_df['Name'] = momo_df['Name'].str.removesuffix('USDT')
    shorts = momo_df[momo_df['Position'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['Position'] == 'LONG'].sort_values(by='SR', ascending=False)

    shorts = shorts.sort_values(by='Signal', ascending=True)
    longs = longs.sort_values(by='Signal', ascending=False)
    table_text = format_multi_text(longs, shorts)
    print(len(table_text))
    await update.message.reply_text(text= table_text, parse_mode='HTML')

# Get L/S Signals for past 7 days
async def curr_mom_top100(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    
    momo_df = momo_df.loc[momo_df['Name'].isin(top_100_mcap)]
    momo_df['Name'] = momo_df['Name'].str.removesuffix('USDT')
    shorts = momo_df[momo_df['Position'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['Position'] == 'LONG'].sort_values(by='SR', ascending=False)

    # Get top 25% of the signal
    if len(shorts)> 0.5 * len(tickers):
        s_len = int(len(shorts)* 0.3)
    else:
        s_len = int(len(shorts)* 0.5)
    
    if len(longs) > 0.5 * len(tickers):
        l_len = int(len(longs)* 0.3)
    else:
        l_len = int(len(longs) * 0.5)

    shorts_top = shorts.iloc[: s_len, :]
    longs_top =  longs.iloc[:l_len, :]
    table_text = format_multi_text(longs_top, shorts_top)
    print(len(table_text))    
    await update.message.reply_text(text= table_text, parse_mode='HTML' )

async def curr_mom_str_top100(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    momo_df = momo_df.loc[momo_df['Name'].isin(top_100_mcap)]
    momo_df['Name'] = momo_df['Name'].str.removesuffix('USDT')
    shorts = momo_df[momo_df['Position'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['Position'] == 'LONG'].sort_values(by='SR', ascending=False)

    # Get top 25% of the signal
    if len(shorts)> 0.5 * len(tickers):
        s_len = int(len(shorts)* 0.25)
    else:
        s_len = int(len(shorts) * 0.5)
    
    if len(longs) > 0.5 * len(tickers):
        l_len = int(len(longs)* 0.25)
    else:
        l_len = int(len(longs) * 0.5)


    shorts_top = shorts.iloc[: s_len, :]
    shorts_top = shorts_top.sort_values(by='Signal', ascending=True)
    
    longs_top =  longs.iloc[:l_len, :]
    longs_top = longs_top.sort_values(by='Signal', ascending=False)
    table_text = format_multi_text(longs_top, shorts_top)
    print(len(table_text))
    await update.message.reply_text(text= table_text, parse_mode='HTML')

# Get Curr Position Momentum
async def port_mommy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    print('STARTING.....')
    # Loading data
    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

    # The params needed that we have now are price_data, tickers, optimized_inputs_dict
    print('GETTING MOMS.....')
    momo_df = run_momo_curr_sig(tickers_price_data, data_frequency, optimized_parameters, ticker_symbol_map)
    momo_df = get_position_trend(momo_df)
    shorts = momo_df[momo_df['PORT DIR'] == 'SHORT'].sort_values(by='SR', ascending=False)
    longs = momo_df[momo_df['PORT DIR'] == 'LONG'].sort_values(by='SR', ascending=False)

    table_text = format_multi_text(longs, shorts)
    print(len(table_text))    
    await update.message.reply_text(text= table_text, parse_mode='HTML' )

from telegram import Update
from telegram.ext import ContextTypes

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /chart command. Generates and sends PnL and Signal charts for a given ticker.
    Usage: /chart BTCUSDT
    """
    try:
        # Check if the user provided a ticker
        if not context.args:
            await update.message.reply_text("Please provide a ticker. Usage: /chart BTCUSDT")
            return

        ticker = context.args[0].upper()
        await update.message.reply_text(f"Generating charts for {ticker}...")

        # --- 1. Load Data ---
        # This part should match your existing bot structure
        tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()

        
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
            lookback=None,
            pnl_dict_gen=True,
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
        
        # Build a robust path relative to the script's location
        charts_folder = os.path.join(BASE_DIR, "momo_charts")
        chart_filename = os.path.join(charts_folder, f"{ticker}_chart.png")
        
        # Optional: Create the directory if it doesn't exist
        os.makedirs(charts_folder, exist_ok=True)
        generate_chart_image(pnl_series, signal_df, ticker, output_filename=chart_filename)
        
        # --- 4. Send the Chart Image ---
        await update.message.reply_photo(photo=open(chart_filename, 'rb'))

    except Exception as e:
        print(f"Error in /chart command: {e}")
        await update.message.reply_text("An error occurred while generating the chart.")


async def _handle_command_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled Telegram command error", exc_info=context.error)
    message = getattr(update, "effective_message", None) if update is not None else None
    if message is None:
        return
    try:
        await message.reply_text("Command failed before output. Check the bot logs for details.")
    except Exception:
        logger.exception("Could not send Telegram command error message.")


def main():
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set (see .env.example).")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    application_builder = ApplicationBuilder().token(settings.telegram_bot_token)
    if not settings.telegram_verify_ssl:
        request_kwargs = {"httpx_kwargs": {"verify": False}}
        application_builder = application_builder.request(
            HTTPXRequest(**request_kwargs)
        ).get_updates_request(HTTPXRequest(**request_kwargs))
    application = application_builder.build()
    application.add_error_handler(_handle_command_error)
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
    application.add_handler(meme_mom_handler)  # application.add_handler(short_mom_handler)
    application.add_handler(port_mom_handler)
    application.add_handler(chart_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
