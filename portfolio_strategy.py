from IPython.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
import pprint
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, timezone
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, Application
from telegram.ext import CallbackContext
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
from binance.exceptions import BinanceAPIException
from binance.client import Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import logging # Import the logging module

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Initialize Logger ---
# This should be done at the module level
logger = logging.getLogger(__name__)
# Basic configuration for the logger if the script is run directly.
# If this module is imported, the importing script should configure logging.
if not logging.getLogger().hasHandlers(): # Avoid adding multiple handlers if imported
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

import optuna

from momo_bot import binance_data as _binance_data
from momo_bot.strategies.scalars import (
    CARVER_BREAKOUT_FORECAST_SCALARS,
    CARVER_EWMAC_FORECAST_SCALARS,
)
from momo_bot.strategies.forecasts import (
    calc_breakout_forecast as _calc_breakout_forecast,
    calc_ewma_forecast as _calc_ewma_forecast,
    get_signal as _get_signal,
)

# sys.path.append(r'C:\Users\ezra.soong\OneDrive - L3 Management Pte Ltd\Desktop\py')
# import cftc_helpers as cft
# # import win32com.client as win32
# sys.path.append(r'C:\Users\ezra.soong\OneDrive - L3 Management Pte Ltd\python')
# #import qcp
# import timeseries_help as tsh


### Crypto Functions ###
def get_prev_date(date, lb):
    # Parse the date string into a datetime object
    date_obj = dt.datetime.strptime(date, "%Y-%m-%d")

    # Calculate the date 30 days before
    date = date_obj - timedelta(days=lb)

    # Format the date in the desired format
    date = date.strftime("%Y-%m-%d")

    return date

def gen_lookback_df(ticker, start_date, end_date, freq='4h', client=None):
    return _binance_data.gen_lookback_df(ticker, start_date, end_date, freq=freq, client=client)

def get_utc_string(dt_object):
    return _binance_data.get_utc_string(dt_object)

def get_fresh_lookback_df(ticker, start_date, freq='4h', client=None):
    return _binance_data.get_fresh_lookback_df(ticker, start_date, freq=freq, client=client)


def get_utc(date):
    return _binance_data.get_utc(date)

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
    return _binance_data.get_all_usdt_perpetual_futures_tickers(api_key=api_key, api_secret=api_secret)


def get_usdt_perpetual_futures_onboard_dates(api_key=None, api_secret=None):
    return _binance_data.get_usdt_perpetual_futures_onboard_dates(
        api_key=api_key,
        api_secret=api_secret,
    )

# Crypto Tickers
def gen_binance_tickers(freq = '1d'):
    from momo_bot.exchange import get_binance_client

    cl = get_binance_client()
    tickers = cl.get_all_tickers()
    tickers = [item['symbol'] for item in tickers]
    tickers_try = [x for x in tickers if 'USDT' in x]

    # DATES
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    start_date = get_prev_date(end_date, 365* 3)

    store = pd.DataFrame()

    for ticker in tickers_try:
        try:
            
            temp = gen_lookback_df(ticker, start_date, end_date, freq=freq).loc[:, ['Close']]
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
    tickers_try = list(tickers_store.columns)
    len(tickers_try)
    da_dict = {coin: coin for coin in tickers_try}

    return tickers_store, tickers_try, da_dict 


#### MOMENTUM STRAT ####
DEFAULT_BDAYS_IN_YEAR = 365
DEFAULT_WEEKS_IN_YEAR = 52
DEFAULT_MONTHS_IN_YEAR = 12

DEFAULT_PORTFOLIO_VALUE = 100000.0
DEFAULT_TARGET_VOL_ANNUAL = 0.2
DEFAULT_CONTRACT_SIZE = DEFAULT_PORTFOLIO_VALUE
ewmac_factors = [(2,8), (4,16), (8,32), (16,64), (32,128)] #, (64,256)]

def _periods_per_year(data_frequency: str) -> int:
    return {'1d': 365, '4h': 365 * 6, '1h': 365 * 24}.get(data_frequency, 365)


def _annualized_sharpe(returns: pd.Series, data_frequency: str) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    std = returns.std()
    if std == 0 or np.isnan(std):
        return np.nan
    periods_per_year = _periods_per_year(data_frequency)
    return returns.mean() / std * np.sqrt(periods_per_year)


def _position_size_from_forecast(
    price: pd.Series,
    forecast: pd.Series,
    *,
    portfolio_value: float,
    target_vol_annual: float,
    data_frequency: str,
    vol_lookback: int,
    max_leverage: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    if price.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    forecast = pd.Series(forecast).reindex(price.index).fillna(0.0)
    periods_per_year = _periods_per_year(data_frequency)
    per_period_risk = portfolio_value * target_vol_annual / np.sqrt(periods_per_year)

    actual_lookback = min(vol_lookback, max(2, len(price.dropna())))
    vol = price.diff().rolling(window=actual_lookback, min_periods=max(2, actual_lookback // 2)).std()
    vol = vol.replace(0, np.nan).bfill().ffill().fillna(1e-8)

    positions = (forecast / 10.0) * (per_period_risk / vol)
    positions = positions.ffill().fillna(0.0)
    positions_usd = positions * price

    if max_leverage is not None:
        cap = max_leverage * portfolio_value
        positions_usd = positions_usd.clip(-cap, cap)
        positions = positions_usd / price.replace(0, np.nan)
        positions = positions.ffill().fillna(0.0)

    return positions, positions_usd

class strategy_statistics:
    
    DEFAULT_BDAYS_IN_YEAR = 365
    DEFAULT_WEEKS_IN_YEAR = 52
    DEFAULT_MONTHS_IN_YEAR = 12

    
    def __init__(self, returns_df, frequency='D'):
        self._returns_df = pd.Series(returns_df)[-DEFAULT_BDAYS_IN_YEAR:]
        
        try:
            returns_scalar= dict(D=DEFAULT_BDAYS_IN_YEAR, W=DEFAULT_WEEKS_IN_YEAR, M=DEFAULT_MONTHS_IN_YEAR, Y=1)[frequency]
            vol_scalar = dict(D=DEFAULT_BDAYS_IN_YEAR**0.5, W=DEFAULT_BDAYS_IN_YEAR**0.5, M=DEFAULT_MONTHS_IN_YEAR**0.5, Y=1)[frequency]
        except KeyError:
            raise Exception("Not a frequency {}".format(frequency))
            
        setattr(self, "frequency", frequency)
        setattr(self, "_returns_scalar", returns_scalar)
        setattr(self, "_vol_scalar", vol_scalar)
        setattr(self, "_returns_df", returns_df)
    
    def as_ts(self):
        return pd.Series(self._returns_df)
    
    def mean(self):
        return(float(self.as_ts().mean()))
               
    def std(self):
        return(float(self.as_ts().std()))
    
    def ann_mean(self):
        avg = self.mean()
        return avg * self._returns_scalar
    
    def ann_std(self):
        period_std = self.std()
        return period_std * self._vol_scalar
    
    def sharpe(self):
        mean_return = self.ann_mean()
        vol = self.ann_std()
        try:
            sharpe = mean_return/vol
        except ZeroDivisionError:
            sharpe = np.nan
        return sharpe
    
    def vals(self):
        x = [z for z in self.as_ts() if not np.isnan(z)]
        return x
    
    def min(self):
        return np.nanmin(self.vals())
    
    def max(self):
        return np.max(self.vals())
    
    def median(self):
        return np.median(self.vals())
    
    def rolling_ann_std(self, window=40):
        y = self.as_ts().rolling(window, min_periods=4, center=True).std().to_frame()
        return y * self._vol_scalar

    def t_test(self):
        return ttest_1samp(self.vals(), 0.0)

    def t_stat(self):
        return float(self.t_test()[0])

    def p_value(self):
        return float(self.t_test()[1])
    
    def skew(self):
        return skew(self.vals())
    
    def curve(self):
        if hasattr(self, "_curve"):
            return self.curve
        else:
            curve = self._returns_df.cumsum()
            setattr(self, "_curve", curve)
            return curve
        
    def drawdown(self):
        x = self.curve()
        return drawdown_func(x)
    
    def worst_drawdown(self):
        dd = self.drawdown()
        return np.nanmin(dd.values)
    
    def stats(self):
        
        stats_list = [ "min","max","median",  "mean","std","ann_mean", "ann_std", "sharpe", "t_stat","p_value", "drawdown"] #"drawdown"
        build_stats = []
        
        for stat_name in stats_list:
            stat_method = getattr(self, stat_name)
            ans = stat_method
            build_stats.append((stat_name, "{0:.4g}".format(ans)))
            
        return [build_stats]
    

def get_rsi(price):
    delta = price.diff()[1:]
    up, down = delta.copy(), delta.copy()
    up[up<0] = 0
    down[down>0] = 0
    
    rolling_up = up.rolling(14).mean()
    rolling_down = down.rolling(14).mean()
    
    rs = rolling_up/ (rolling_down * -1)
    rsi = 100 - (100/(1+rs))
    
    return rsi

def get_pnl(
    price,
    trades = None,
    positions = None,
    roundpositions = False,
    delayfill = True,
    get_daily_returns_volatility = None,
    forecast= None,
    fx = None, 
    contract_size = None,
    value_of_price_point=1.0,
):
    """
    If trades are not provided, work out using positions (default)
    
    If delayfill is True, assume we get filled at the next price after the trade
    
    If fx is not provided, assume fx is 1.0 and work out p&l in ccy of instrument (default)
    
    """
    if price is None:
        raise Exception("can't work out p&l without price")
        
    if fx is None:
        use_fx = pd.Series([1.0]* len(price.index), index=price.index)
    else:
        use_fx = fx.reindex(price.index, method='ffill')
    
    #price = price.fillna(method = 'ffill').dropna()

    rsi = get_rsi(price)
        
    if trades is None:
        prices_to_use = price.copy()
        #if positions is None:
        #    positions = get_positions(price, forecast, get_daily_returns_volatility, contract_size, )

        if roundpositions:
            use_positions = positions.round()
        else:
            use_positions = positions.copy()

        if delayfill:
            use_positions = use_positions.shift(1)
        
        #use_positions = np.where((rsi>30)&(rsi<70), use_positions, 'nan' )
        cum_trades = use_positions.ffill()
        trades_to_use = cum_trades.diff()        
    else:
        prices_to_use = price.copy()
        cum_trades = trades.ffill()
        trades_to_use = cum_trades.diff()
    
    price_returns = prices_to_use.ffill().diff()
    instr_ccy_returns = cum_trades.shift(1) * price_returns * value_of_price_point
    
    instr_ccy_returns = instr_ccy_returns.cumsum().ffill().reindex(price.index).diff()
    base_ccy_returns = instr_ccy_returns * use_fx
    
    return(cum_trades, trades_to_use, instr_ccy_returns, base_ccy_returns, use_fx, value_of_price_point)


def calc_ewma_forecast(price, Lfast, Lslow=None, vol_lookback: int = 365):
    return _calc_ewma_forecast(price, Lfast=Lfast, Lslow=Lslow, vol_lookback=vol_lookback)

def get_signal(price, forecast, get_daily_returns_volatility, instrument_multiplier, **kwargs):
    return _get_signal(price, forecast, get_daily_returns_volatility, instrument_multiplier, **kwargs)

#Breakout Forecast
def calc_breakout_forecast(price_series: pd.Series, horizon: int) -> pd.Series:
    return _calc_breakout_forecast(price_series, horizon=horizon)


# GENERALIZED OPTIMIZATION FUNCTION
def optimize_function_gen(inputs, price_data, ticker):
    # x1, x2, x3, x4, x5, x6 = inputs
    x1, x2, x3, x4, x5 = inputs


    ewmac_factors = [(2,8), (4,16), (8,32), (16,64), (32,128)] #, (64,256)]
    normalising_factors = {
        '(2,8)' : x1,
        '(4,16)' : x2,
        '(8,32)' : x3,
        '(16,64)' : x4,
        '(32,128)' : x5,
        #'(64,256)' : x6,
    }
    
    #takes ticker input
    price = price_data[ticker]
    price = price.ffill()#.dropna()
    
    signal_df = pd.DataFrame()
    
    for i,j in ewmac_factors:
    
        forecast = calc_ewma_forecast(price, Lfast=i, Lslow=j)
        signal = get_signal(price, forecast, None, normalising_factors['({},{})'.format(i,j)])
        signal_df['({},{})'.format(i,j)] = signal
    
    signal_df['weighted signal'] = signal_df.mean(axis=1) * 1.12 #diversification_mult
    
    positions, _ = _position_size_from_forecast(
        price,
        signal_df['weighted signal'],
        portfolio_value=DEFAULT_CONTRACT_SIZE,
        target_vol_annual=DEFAULT_TARGET_VOL_ANNUAL,
        data_frequency='1d',
        vol_lookback=365,
    )
    pnl = get_pnl(price, trades=None, positions=positions)
        
    breakdown = signal_df.tail(1)
    signal = round(signal_df['weighted signal'].iloc[-1] * 5, 2) 
    position = 'LONG' if signal >0 else 'SHORT'
    
    returns = pnl[3] / DEFAULT_CONTRACT_SIZE
    sr = round(_annualized_sharpe(returns, '1d'), 2)
    
    return -sr

def optimize(price_frame, tickers, input_ranges):
    optimized_inputs_dict = {}
    with tqdm.tqdm(total=len(tickers), file=sys.stdout, position=0, leave=True) as pbar:
        for ticker in tickers:
            try:
                result = minimize(optimize_function_gen, np.zeros(5), bounds=input_ranges, method='Powell', args=( price_frame, ticker,))
                optimized_inputs = result.x
                optimized_output = -result.fun

                optimized_inputs_dict[ticker] = {'inputs': optimized_inputs, 'sr': optimized_output}
                print(f'Optimized for {ticker}')

            except IndexError as e:
                print(f"IndexError: {ticker}")
                # Optionally, you can also log or handle the error in some way here

            finally:
                # Update progress bar whether an error occurred or not
                pbar.update(1)

    return optimized_inputs_dict


# --- Optuna Objective Function (Modified) ---
def _optuna_objective_generalized(trial, 
                                  price_data_ticker, 
                                  ewmac_factors, 
                                  input_ranges_for_x, 
                                  vol_lookback_for_pos_sizing=365, 
                                  DEFAULT_CONTRACT_SIZE=DEFAULT_PORTFOLIO_VALUE):
    """
    Objective function for Optuna to optimize normalizing factors (x1 to x5) and weights.
    """
    if len(ewmac_factors) != len(input_ranges_for_x):
        raise ValueError("Mismatch between number of EWMAC factors and input ranges for x_i.")

    # 1. Suggest normalizing factors (x1 to xn)
    normalizing_factors_values = []
    for i in range(len(ewmac_factors)):
        min_val, max_val = input_ranges_for_x[i]
        normalizing_factors_values.append(trial.suggest_float(f'x{i+1}', min_val, max_val))

    # 2. Suggest raw weights (w1 to wn) in [0,1] and normalize to sum=1.0
    raw_weights = []
    for i in range(len(ewmac_factors)):
        raw_weights.append(trial.suggest_float(f'w{i+1}', 0.0, 1.0))
    
    total_raw = sum(raw_weights)
    if total_raw < 1e-8:  # Avoid division by zero
        weights = [1.0 / len(ewmac_factors)] * len(ewmac_factors)
    else:
        weights = [w / total_raw for w in raw_weights]  # Normalized weights

    # Reconstruct normalising_factors dictionary
    normalising_factors_dict = {}
    for idx, (i, j) in enumerate(ewmac_factors):
        normalising_factors_dict[f'({i},{j})'] = normalizing_factors_values[idx]

    price = price_data_ticker.ffill()
    if price.empty:
        return np.inf

    signal_df = pd.DataFrame(index=price.index)

    # Calculate signal components
    for factor_pair_str, norm_value in normalising_factors_dict.items():
        i, j = map(int, factor_pair_str.strip('()').split(','))
        forecast = calc_ewma_forecast(price, Lfast=i, Lslow=j)
        signal_component = get_signal(price, forecast, None, norm_value)
        signal_df[factor_pair_str] = signal_component.reindex(price.index)

    # 3. Calculate WEIGHTED SIGNAL (using normalized weights)
    weighted_signal = pd.Series(0, index=signal_df.index)
    for i, col in enumerate(signal_df.columns):
        weighted_signal += signal_df[col] * weights[i]
    weighted_signal *= 1.12  # Apply diversification multiplier
    signal_df['weighted_signal'] = weighted_signal

    # Position sizing and P&L calculation (unchanged)
    actual_vol_lookback = min(vol_lookback_for_pos_sizing, len(price.diff().dropna()))
    if actual_vol_lookback < 2:
        return np.inf

    positions, _ = _position_size_from_forecast(
        price,
        signal_df['weighted_signal'],
        portfolio_value=DEFAULT_CONTRACT_SIZE,
        target_vol_annual=DEFAULT_TARGET_VOL_ANNUAL,
        data_frequency='1d',
        vol_lookback=actual_vol_lookback,
    )
    pnl_series = get_pnl(price, trades=None, positions=positions)
    if pnl_series[3].empty or pnl_series[3].isnull().all() or len(pnl_series[3].dropna()) < 20:
        return np.inf

    returns = pnl_series[3] / DEFAULT_CONTRACT_SIZE
    sr = _annualized_sharpe(returns, '1d')
    return -sr if pd.notnull(sr) and not np.isinf(sr) else np.inf


# --- Main Optuna Loop (Modified) ---
def optimize_ewmac_factors_with_optuna(
    price_frame, 
    tickers, 
    ewmac_factors_list, 
    input_ranges_for_x_factors,
    vol_lookback_for_position_sizing=365,
    n_trials_per_ticker=500,
    study_name_prefix="ewmac_factor_opt"
):
    if len(ewmac_factors_list) != len(input_ranges_for_x_factors):
        raise ValueError("Mismatch between EWMAC factors and input ranges.")

    optimized_inputs_dict = {}
    print('NEW ONE')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with tqdm.tqdm(total=len(tickers), desc="Optimizing Tickers") as pbar:
        for ticker in tickers:
            pbar.set_description(f"Optimizing {ticker}")
            price_data_ticker = price_frame[ticker]

            # Skip if insufficient data
            min_data_required = max(f[1] for f in ewmac_factors_list) + vol_lookback_for_position_sizing
            if price_data_ticker.isnull().all() or len(price_data_ticker.dropna()) < min_data_required:
                optimized_inputs_dict[ticker] = {
                    'normalizing_factors': [np.nan] * len(ewmac_factors_list),
                    'weights': [np.nan] * len(ewmac_factors_list),
                    'sr': np.nan,
                    'status': 'skipped_insufficient_data'
                }
                pbar.update(1)
                continue
            
            study_name = f"{study_name_prefix}_{ticker}"
            study = optuna.create_study(study_name=study_name, direction='minimize')

            try:
                study.optimize(
                    lambda trial: _optuna_objective_generalized(
                        trial, 
                        price_data_ticker, 
                        ewmac_factors_list, 
                        input_ranges_for_x_factors,
                        vol_lookback_for_position_sizing
                    ),
                    n_trials=n_trials_per_ticker
                )

                if study.best_trial and study.best_trial.value != np.inf:
                    # Extract best normalizing factors
                    best_x = [study.best_trial.params[f'x{i+1}'] for i in range(len(ewmac_factors_list))]
                    
                    # Extract and NORMALIZE best weights
                    raw_weights = [study.best_trial.params[f'w{i+1}'] for i in range(len(ewmac_factors_list))]
                    total_raw = sum(raw_weights)
                    if total_raw < 1e-8:
                        best_weights = [1.0/len(ewmac_factors_list)] * len(ewmac_factors_list)
                    else:
                        best_weights = [w/total_raw for w in raw_weights]
                    
                    optimized_inputs_dict[ticker] = {
                        'normalizing_factors': best_x,
                        'weights': best_weights,  # Sums to 1.0
                        'sr': -study.best_trial.value,
                        'status': 'success'
                    }
                else:
                    optimized_inputs_dict[ticker] = {
                        'normalizing_factors': [np.nan] * len(ewmac_factors_list),
                        'weights': [np.nan] * len(ewmac_factors_list),
                        'sr': np.nan,
                        'status': 'failed_no_valid_solution'
                    }

            except Exception as e:
                optimized_inputs_dict[ticker] = {
                    'normalizing_factors': [np.nan] * len(ewmac_factors_list),
                    'weights': [np.nan] * len(ewmac_factors_list),
                    'sr': np.nan,
                    'status': f'error_{type(e).__name__}'
                }
            finally:
                pbar.update(1)
                
    return optimized_inputs_dict

# --- New Optuna Objective Function ---
def _objective_optimize_weights(
    trial,
    price_data_ticker,
    ewmac_factors_list,
    forecast_scalars,  # We now use the fixed scalars from the book
    vol_lookback_for_pos_sizing,
    DEFAULT_CONTRACT_SIZE,
    diversification_multiplier,
    data_frequency: str,
    target_vol_annual: float,
    max_leverage: float | None,
    cap_final_forecast: bool = False,
):
    """
    Optuna objective function that optimizes ONLY the weights for combining EWMAC rules.
    It uses pre-defined, fixed forecast scalars as per Carver's methodology.
    """
    # 1. Suggest raw weights to be optimized. Optuna will find the best combination.
    raw_weights = [
        trial.suggest_float(f'w{i+1}', 1e-6, 1.0) for i in range(len(ewmac_factors_list))
    ]
    # Normalize the trial weights so they sum to 1.0
    total_raw_weight = sum(raw_weights)
    if total_raw_weight < 1e-8:
        weights = [1.0 / len(ewmac_factors_list)] * len(ewmac_factors_list)
    else:
        weights = [w / total_raw_weight for w in raw_weights]

    # --- Backtest Logic ---
    price = price_data_ticker.ffill()
    if price.empty:
        return np.inf # Return a large value to indicate a failed trial

    signal_df = pd.DataFrame(index=price.index)

    # Calculate individual signal components using FIXED forecast scalars
    for idx, (i, j) in enumerate(ewmac_factors_list):
        # This is where your signal logic goes.
        # The key change is using forecast_scalars[idx] instead of an optimized 'x' value.
        # Example placeholder logic:
        raw_forecast = calc_ewma_forecast(price, Lfast=i, Lslow=j, vol_lookback=vol_lookback_for_pos_sizing)
        # This forecast should be volatility-standardized before applying the scalar
        # For simplicity, we assume calc_ewma_forecast returns the final standardized forecast
        scaled_forecast = (raw_forecast * forecast_scalars[idx]).clip(-20, 20)
        signal_df[f'({i},{j})'] = scaled_forecast.reindex(price.index)

    # 2. Calculate the WEIGHTED SIGNAL using the optimized weights
    weighted_signal = pd.Series(0.0, index=signal_df.index)
    for i, col in enumerate(signal_df.columns):
        weighted_signal += signal_df[col].fillna(0) * weights[i]

    # 3. Apply the Diversification Multiplier
    # NOTE: Carver's full methodology calculates this based on weights and signal correlations.
    # A fixed multiplier is a simplification.
    weighted_signal *= diversification_multiplier
    if cap_final_forecast:
        weighted_signal = weighted_signal.clip(-20, 20)
    
    # 4. Position sizing and P&L calculation
    actual_vol_lookback = min(vol_lookback_for_pos_sizing, len(price.diff().dropna()))
    if actual_vol_lookback < 2:
        return np.inf

    positions, _ = _position_size_from_forecast(
        price,
        weighted_signal,
        portfolio_value=DEFAULT_CONTRACT_SIZE,
        target_vol_annual=target_vol_annual,
        data_frequency=data_frequency,
        vol_lookback=actual_vol_lookback,
        max_leverage=max_leverage,
    )
    pnl_series = get_pnl(price, trades=None, positions=positions) # Your function
    if pnl_series[3].empty or len(pnl_series[3].dropna()) < 20:
        return np.inf

    # We want to MINIMIZE the NEGATIVE of the Sharpe Ratio
    returns = pnl_series[3] / DEFAULT_CONTRACT_SIZE
    sr = _annualized_sharpe(returns, data_frequency)
    return -sr if pd.notnull(sr) and not np.isinf(sr) else np.inf


# --- New Main Trend Strategy Optimization---

def optimize_ewmac_weights_carver_method(
    price_frame,
    tickers,
    ewmac_factors_list,
    vol_lookback_for_position_sizing=365,
    n_trials_per_ticker=500,
    study_name_prefix="ewmac_weight_opt_carver",
    DEFAULT_CONTRACT_SIZE=DEFAULT_PORTFOLIO_VALUE,
    diversification_multiplier=1.12, # This is a simplification; see book Ch. 8
    data_frequency: str = '4h',
    target_vol_annual: float = DEFAULT_TARGET_VOL_ANNUAL,
    max_leverage: float | None = None,
    adj = 0
):
    # CARVER_FORECAST_SCALARS = {
    # (2, 8): 10.6,
    # (4, 16): 7.5,
    # (8, 32): 5.3,
    # (16, 64): 3.75,
    # (32, 128): 2.65,
    # (64, 256): 1.87,
    # }
    # Adjustment for 4hr intervals
    CARVER_FORECAST_SCALARS = {(12, 48): 10.6,
        (24, 96): 7.5,
        (48, 192): 5.3,
        (96, 384): 3.75,
        (192, 768): 2.65,
        (384, 1536): 1.87}
    # Retrieve the fixed forecast scalars from the book's table
    try:
        forecast_scalars = [CARVER_FORECAST_SCALARS[tuple(f)] for f in ewmac_factors_list]
    except KeyError as e:
        raise ValueError(f"EWMAC factor pair {e} not found in CARVER_FORECAST_SCALARS. "
                         "Please use one of the pairs from the book: {list(CARVER_FORECAST_SCALARS.keys())}")

    optimized_results_dict = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with tqdm.tqdm(total=len(tickers), desc="Optimizing Tickers") as pbar:
        for ticker in tickers:
            pbar.set_description(f"Optimizing {ticker}")
            price_data_ticker = price_frame[ticker]

            # Skip if insufficient data
            min_data_required = max(f[1] for f in ewmac_factors_list) + vol_lookback_for_position_sizing
            if price_data_ticker.isnull().all() or len(price_data_ticker.dropna()) < min_data_required:
                optimized_results_dict[ticker] = {
                    'weights': [np.nan] * len(ewmac_factors_list), 'sr': np.nan, 'status': 'skipped_insufficient_data'
                }
                pbar.update(1)
                continue

            study = optuna.create_study(study_name=f"{study_name_prefix}_{ticker}", direction='minimize')

            try:
                study.optimize(
                    lambda trial: _objective_optimize_weights(
                        trial,
                        price_data_ticker,
                        ewmac_factors_list,
                        forecast_scalars,
                        vol_lookback_for_position_sizing,
                        DEFAULT_CONTRACT_SIZE,
                        diversification_multiplier,
                        data_frequency,
                        target_vol_annual,
                        max_leverage,
                    ),
                    n_trials=n_trials_per_ticker
                )

                if study.best_trial and study.best_trial.value != np.inf:
                    # Extract and NORMALIZE the best weights to sum to 1.0
                    raw_weights = [study.best_trial.params[f'w{i+1}'] for i in range(len(ewmac_factors_list))]
                    total_raw = sum(raw_weights)
                    if total_raw < 1e-8:
                        best_weights = [1.0 / len(ewmac_factors_list)] * len(ewmac_factors_list)
                    else:
                        best_weights = [w / total_raw for w in raw_weights]

                    optimized_results_dict[ticker] = {
                        'weights': best_weights,
                        'sr': -study.best_trial.value, # Convert back to positive SR
                        'status': 'success'
                    }
                else:
                    optimized_results_dict[ticker] = {
                        'weights': [np.nan] * len(ewmac_factors_list), 'sr': np.nan, 'status': 'failed_no_valid_solution'
                    }

            except Exception as e:
                optimized_results_dict[ticker] = {
                    'weights': [np.nan] * len(ewmac_factors_list), 'sr': np.nan, 'status': f'error_{type(e).__name__}'
                }
            finally:
                pbar.update(1)

    return optimized_results_dict



# --- Fixed Scalars for Breakout Strategy (from Table 91) ---
def _objective_optimize_breakout_weights(
    trial,
    price_data_ticker,
    breakout_horizons,
    forecast_scalars,
    vol_lookback_for_pos_sizing,
    DEFAULT_CONTRACT_SIZE,
    diversification_multiplier,
    data_frequency: str,
    target_vol_annual: float,
    max_leverage: float | None,
    cap_final_forecast: bool = False,
):
    """
    Optuna objective function that optimizes the weights for combining
    different breakout rule variations.
    """
    # Suggest raw weights to be optimized
    raw_weights = [
        trial.suggest_float(f'w{i+1}', 1e-6, 1.0) for i in range(len(breakout_horizons))
    ]
    total_raw_weight = sum(raw_weights)
    if total_raw_weight < 1e-8:
        weights = [1.0 / len(breakout_horizons)] * len(breakout_horizons)
    else:
        weights = [w / total_raw_weight for w in raw_weights]

    price = price_data_ticker.ffill()
    if price.empty: return np.inf

    signal_df = pd.DataFrame(index=price.index)

    # Calculate individual breakout signals using FIXED scalars
    for idx, horizon in enumerate(breakout_horizons):
        # Use the new breakout calculation function
        raw_forecast = calc_breakout_forecast(price, horizon=horizon)
        
        # Scale and cap the forecast
        scaled_forecast = raw_forecast * forecast_scalars[idx]
        capped_forecast = scaled_forecast.clip(-20, 20)
        signal_df[f'Breakout_{horizon}'] = capped_forecast.reindex(price.index)

    # Combine signals using the optimized weights
    weighted_signal = np.dot(signal_df.fillna(0), weights)
    weighted_signal = pd.Series(weighted_signal, index=signal_df.index) * diversification_multiplier
    if cap_final_forecast:
        weighted_signal = weighted_signal.clip(-20, 20)
    
    # Position sizing and PnL calculation
    positions, _ = _position_size_from_forecast(
        price,
        weighted_signal,
        portfolio_value=DEFAULT_CONTRACT_SIZE,
        target_vol_annual=target_vol_annual,
        data_frequency=data_frequency,
        vol_lookback=vol_lookback_for_pos_sizing,
        max_leverage=max_leverage,
    )
    pnl_series = get_pnl(price, trades=None, positions=positions)[3] # Get base_ccy_returns

    if pnl_series.empty or len(pnl_series.dropna()) < 20:
        return np.inf

    returns = pnl_series / DEFAULT_CONTRACT_SIZE
    sr = _annualized_sharpe(returns, data_frequency)
    return -sr if pd.notnull(sr) and not np.isinf(sr) else np.inf


def optimize_breakout_weights_carver_method(
    price_frame,
    tickers,
    breakout_horizons,
    vol_lookback_for_position_sizing=365,
    n_trials_per_ticker=100,
    study_name_prefix="breakout_weight_opt_carver",
    DEFAULT_CONTRACT_SIZE=DEFAULT_PORTFOLIO_VALUE,
    diversification_multiplier=1.24, # From Table 96 for 80, 160, 320 day breakouts
    forecast_scalars: dict[int, float] | None = None,
    data_frequency: str = '1d',
    target_vol_annual: float = DEFAULT_TARGET_VOL_ANNUAL,
    max_leverage: float | None = None,
    cap_final_forecast: bool = False,
):
    """
    Main Optuna optimization loop for the breakout strategy.
    """
    CARVER_BREAKOUT_SCALARS = CARVER_BREAKOUT_FORECAST_SCALARS
    scalars_map = CARVER_BREAKOUT_SCALARS if forecast_scalars is None else forecast_scalars
    try:
        forecast_scalars = [scalars_map[h] for h in breakout_horizons]
    except KeyError as e:
        raise ValueError(f"Breakout horizon {e} not found in CARVER_BREAKOUT_SCALARS.")

    optimized_results_dict = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with tqdm.tqdm(total=len(tickers), desc="Optimizing Breakout Tickers") as pbar:
        for ticker in tickers:
            pbar.set_description(f"Optimizing {ticker}")
            # --- The rest of the loop is identical to the EWMAC version ---
            price_data_ticker = price_frame[ticker]
            min_data_required = max(breakout_horizons) + vol_lookback_for_position_sizing
            if price_data_ticker.isnull().all() or len(price_data_ticker.dropna()) < min_data_required:
                optimized_results_dict[ticker] = {'weights': [np.nan]*len(breakout_horizons), 'sr': np.nan, 'status': 'skipped_insufficient_data'}
                pbar.update(1)
                continue
            
            study = optuna.create_study(study_name=f"{study_name_prefix}_{ticker}", direction='minimize')
            try:
                study.optimize(
                    lambda trial: _objective_optimize_breakout_weights(
                        trial, price_data_ticker, breakout_horizons, forecast_scalars,
                        vol_lookback_for_position_sizing, DEFAULT_CONTRACT_SIZE, diversification_multiplier,
                        data_frequency, target_vol_annual, max_leverage,
                        cap_final_forecast=cap_final_forecast
                    ),
                    n_trials=n_trials_per_ticker
                )
                if study.best_trial and study.best_trial.value != np.inf:
                    raw_weights = [study.best_trial.params[f'w{i+1}'] for i in range(len(breakout_horizons))]
                    total_raw = sum(raw_weights)
                    best_weights = [w / total_raw for w in raw_weights] if total_raw > 1e-8 else [1.0 / len(breakout_horizons)] * len(breakout_horizons)
                    optimized_results_dict[ticker] = {'weights': best_weights, 'sr': -study.best_trial.value, 'status': 'success'}
                else:
                    optimized_results_dict[ticker] = {'weights': [np.nan]*len(breakout_horizons), 'sr': np.nan, 'status': 'failed_no_valid_solution'}
            except Exception as e:
                optimized_results_dict[ticker] = {'weights': [np.nan]*len(breakout_horizons), 'sr': np.nan, 'status': f'error_{type(e).__name__}'}
            finally:
                pbar.update(1)

    return optimized_results_dict




def get_pnl_walkforward(price, inputs, ticker):
    """
    Calculate P&L for walk-forward testing using optimized parameters.
    """
    x1, x2, x3, x4, x5, x6 = inputs
    normalising_factors = {
        str(ewmac_factors[0]): x1, str(ewmac_factors[1]): x2, str(ewmac_factors[2]): x3,
        str(ewmac_factors[3]): x4, str(ewmac_factors[4]): x5, str(ewmac_factors[5]): x6,
    }

    price = price[ticker].ffill().dropna()
    if price.empty or len(price) < max(f[1] for f in ewmac_factors) + 60:
        print(f"Not enough price data ({len(price)} points) for P&L calculation.")
        return None

    signal_df = pd.DataFrame(index=price.index)
    for i, j in ewmac_factors:
        forecast = calc_ewma_forecast(price, Lfast=i, Lslow=j)
        signal_contrib = get_signal(price, forecast, normalising_factors[f'({i},{j})'])
        signal_df[f'({i},{j})'] = signal_contrib

    signal_df['weighted_signal'] = signal_df.mean(axis=1, skipna=True) #* 1.12

    pos_vol_window = min(len(price)-1, 252)
    if pos_vol_window < 20: pos_vol_window = 20
    if len(price.diff().dropna()) < pos_vol_window:
        vol_for_pos = price.diff().expanding(min_periods=max(2, pos_vol_window //2)).std()
    else:
        vol_for_pos = price.diff().rolling(window=pos_vol_window, min_periods=max(2, pos_vol_window//2)).std()

    positions, _ = _position_size_from_forecast(
        price,
        signal_df['weighted_signal'],
        portfolio_value=DEFAULT_CONTRACT_SIZE,
        target_vol_annual=DEFAULT_TARGET_VOL_ANNUAL,
        data_frequency='1d',
        vol_lookback=pos_vol_window,
    )

    pnl_series = get_pnl(price, trades=None, positions=positions)
    return pnl_series


def gen_signal_ls(price_frame, tickers, ticker_dict, optimized_inputs_dict, lookback=7, pnl_dict_gen = False):
    email_df = pd.DataFrame(columns=['Name', 'Position', 'Signal', 'SR'])
    pnl_dict = {}
    signal_dict = {x: [] for x in tickers}

    for ticker in tickers:
        try:
            # Attempt to unpack inputs and calculate signals
            #x1, x2, x3, x4, x5, x6 = optimized_inputs_dict[ticker]['inputs']
            x1, x2, x3, x4, x5 = optimized_inputs_dict[ticker]['normalizing_factors']#['inputs']

            normalising_factors = {
                '(2,8)': x1,
                '(4,16)': x2,
                '(8,32)': x3,
                '(16,64)': x4,
                '(32,128)': x5,
                #'(64,256)': x6,
            }

            # Handle different types of lookback inputs
            if type(lookback) == int:
                price = price_frame[ticker][lookback * 6 :].ffill().dropna()
            else:
                price = price_frame[ticker]

            price = price.ffill().dropna()

            signal_df = pd.DataFrame()

            # Assume ewmac_factors is defined elsewhere in your code
            # ewmac_factors = [(2,8), (4,16), (8,32), (16,64), (32,128), (64,256)]
            ewmac_factors = [(2,8), (4,16), (8,32), (16,64), (32,128)]

            for i, j in ewmac_factors:
                forecast = calc_ewma_forecast(price, Lfast=i, Lslow=j)
                signal = get_signal(price, forecast, None, normalising_factors['({},{})'.format(i,j)])
                signal_df['({},{})'.format(i,j)] = signal


            signal_df['weighted signal'] = signal_df.mean(axis=1) * 1.12  # diversification multiplier
            # Keep both spellings for downstream compatibility (some code expects 'weighted signal',
            # other code expects 'weighted_signal').
            signal_df['weighted signal'] = signal_df['weighted signal'].clip(-20, 20)
            signal_df['weighted_signal'] = signal_df['weighted signal']
            
            # Set this later based on your strategy
            vol_param = 500
            positions, _ = _position_size_from_forecast(
                price,
                signal_df['weighted signal'],
                portfolio_value=DEFAULT_CONTRACT_SIZE,
                target_vol_annual=DEFAULT_TARGET_VOL_ANNUAL,
                data_frequency='1d',
                vol_lookback=vol_param,
            )
            pnl = get_pnl(price, trades=None, positions=positions)
            pnl_dict[ticker] = pnl
            signal_dict[ticker].append(signal_df)

            breakdown = signal_df.tail(1)
            signal = round(signal_df['weighted signal'].iloc[-1] , 2)
            position = 'LONG' if signal > 0 else 'SHORT'

            returns = pnl[3] / DEFAULT_CONTRACT_SIZE
            sr = round(_annualized_sharpe(returns, '1d'), 2)

            add_in = {'Name': ticker_dict[ticker], 'Position': position, 'Signal': signal, 'SR': sr}
            email_df = pd.concat([email_df, pd.DataFrame(add_in, index=[0])], ignore_index=True)

        except KeyError as e:
            print(f"KeyError for ticker {ticker}: {e}")
            # Optionally log the error or handle it in a specific way
            continue  # This ensures the next iteration of the loop is processed

    if pnl_dict_gen:
        print(email_df)
        return pnl_dict, signal_dict

    return email_df


def generate_signals_for_grid(price_frame, tickers, x_i_params, ewmac_factors_list):
    """
    Generates combined signals for all tickers given fixed x_i and EWMAC factors.
    Returns a DataFrame of weighted signals.
    """
    all_signals_df = pd.DataFrame(index=price_frame.index)

    for ticker in tickers:
        if ticker not in price_frame.columns or ticker not in x_i_params:
            continue
            
        price = price_frame[ticker].ffill()
        if price.empty:
            continue

        inputs = x_i_params[ticker].get('inputs', [])
        if not inputs or len(inputs) != len(ewmac_factors_list):
            # logger.warning(f"Missing or mismatched x_i params for {ticker}. Skipping.")
            continue

        normalising_factors_dict = {
            f'({i},{j})': inputs[idx] 
            for idx, (i, j) in enumerate(ewmac_factors_list)
        }

        signal_df = pd.DataFrame(index=price.index)
        for factor_pair_str, norm_value in normalising_factors_dict.items():
            i_fast, j_slow = map(int, factor_pair_str.strip('()').split(','))
            forecast = calc_ewma_forecast(price, Lfast=i_fast, Lslow=j_slow)
            signal_df[factor_pair_str] = get_signal(price, forecast, None, norm_value)
        
        all_signals_df[ticker] = signal_df.mean(axis=1, skipna=True) * 1.12 # Weighted signal

    return all_signals_df.fillna(0)

# --- Adjusted Function to Generate Signals and PnL ---
def carver_gen_signal(
    price_frame,
    tickers,
    ticker_dict,
    optimized_inputs_dict,
    ewmac_factors=[(2,8), (4,16), (8,32), (16,64), (32,128)],
    lookback=7,
    pnl_dict_gen=False,
    diversification_multiplier=1.12, # From Carver's framework: 1.12
    forecast_scalars: dict[tuple[int, int], float] | None = None,
    cap_final_forecast: bool = True,
    data_frequency: str = '4h',
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
    target_vol_annual: float = DEFAULT_TARGET_VOL_ANNUAL,
    max_leverage: float | None = None,
):
    """
    Generates signals and PnL based on optimized weights from the Carver-aligned Optuna function.
    """
    email_df = pd.DataFrame(columns=['Name', 'Position', 'Signal', 'SR'])
    pnl_dict = {}
    signal_dict = {x: [] for x in tickers}

    for ticker in tickers:
        try:
            # --- FIX 1: Correctly retrieve the optimized weights ---
            # The optimization output gives us the weights to combine signals.
            if optimized_inputs_dict[ticker]['status'] != 'success':
                continue
            
            optimized_weights = optimized_inputs_dict[ticker]['weights']

            # Handle different types of lookback inputs
            if isinstance(lookback, int):
                price = price_frame[ticker].iloc[lookback * -6:].ffill()
            else:
                price = price_frame[ticker].ffill()

            if price.dropna().empty:
                continue

            signal_df = pd.DataFrame(index=price.index)

            # --- FIX 2: Calculate individual signals using FIXED scalars ---
            # Each signal component is calculated independently using its pre-defined scalar.
            for i, j in ewmac_factors:
                # Calculate the raw, volatility-adjusted forecast
                raw_forecast = calc_ewma_forecast(price, Lfast=i, Lslow=j)
                
                # Get the correct, non-optimized scalar for this rule
                try:
                    scalar_map = forecast_scalars or CARVER_EWMAC_FORECAST_SCALARS
                    fixed_scalar = scalar_map[(i, j)]
                except KeyError:
                    raise ValueError(f"EWMAC pair ({i},{j}) not found in CARVER_FORECAST_SCALARS.")

                # get_signal now correctly scales and caps the forecast
                signal = get_signal(price, raw_forecast, None, fixed_scalar)
                signal_df[f'({i},{j})'] = signal

            # --- FIX 3: Combine signals using the OPTIMIZED weights ---
            # Instead of .mean(), we compute the weighted average using the optimized weights.
            # np.dot is an efficient way to calculate the weighted sum.
            weighted_signal = np.dot(signal_df.fillna(0), optimized_weights)
            
            # Apply the diversification multiplier as per Carver's framework
            weighted_signal = pd.Series(weighted_signal, index=signal_df.index) * diversification_multiplier
            final_capped_signal = weighted_signal.clip(-20, 20) if cap_final_forecast else weighted_signal
            signal_df['weighted_signal'] = final_capped_signal
            # Keep legacy column name used by charting / older code paths.
            signal_df['weighted signal'] = signal_df['weighted_signal']
            
            # --- Position sizing and PnL calculation (risk-targeted) ---
            vol_param = 500 # Assuming this is your intended lookback for volatility
            positions, _ = _position_size_from_forecast(
                price,
                signal_df['weighted_signal'],
                portfolio_value=portfolio_value,
                target_vol_annual=target_vol_annual,
                data_frequency=data_frequency,
                vol_lookback=vol_param,
                max_leverage=max_leverage,
            )
            pnl = get_pnl(price, trades=None, positions=positions) # Use your get_pnl function
            pnl_dict[ticker] = pnl
            signal_dict[ticker].append(signal_df)

            # --- Reporting logic (remains the same) ---
            signal = round(signal_df['weighted_signal'].iloc[-1], 2)
            position = 'LONG' if signal > 0 else 'SHORT'
            returns = pnl[3] / portfolio_value
            sr = round(_annualized_sharpe(returns, data_frequency), 2)

            add_in = {'Name': ticker_dict.get(ticker, ticker), 'Position': position, 'Signal': signal, 'SR': sr}
            email_df = pd.concat([email_df, pd.DataFrame(add_in, index=[0])], ignore_index=True)

        except (KeyError, IndexError) as e:
            print(f"Skipping ticker {ticker} due to error: {e}")
            continue

    if pnl_dict_gen:
        print(email_df)
        return pnl_dict, signal_dict

    return email_df

# Generate overall 
# Generate overall 
def carver_gen_signal_unified(
    price_frame,
    tickers,
    ticker_dict,
    optimized_inputs_dict,
    rule_variations,
    strategy_type: str,
    data_frequency: str = '4h', # Added to handle annualization correctly
    lookback=None,
    pnl_dict_gen=False,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
    target_vol_annual: float = DEFAULT_TARGET_VOL_ANNUAL,
    max_leverage: float | None = None,
    vol_lookback_for_position_sizing=360, # Made this consistent
    forecast_scalars: dict | None = None,
    diversification_multiplier: float | None = None,
    cap_final_forecast: bool = True,
):
    """
    Generates signals and PnL, fixing lookahead bias and SR annualization.
    Now returns position size and direction over time.
    """
    CARVER_EWMAC_SCALARS = CARVER_EWMAC_FORECAST_SCALARS
    CARVER_BREAKOUT_SCALARS = CARVER_BREAKOUT_FORECAST_SCALARS
    
    # --- New Return Dictionaries ---
    email_df = pd.DataFrame(columns=['Name', 'Position Size', 'Direction', 'Signal', 'SR'])
    pnl_dict = {}
    signal_dict = {x: [] for x in tickers}

    # --- FIX: Define annualization factor based on data frequency ---
    periods_per_year = _periods_per_year(data_frequency)
    annualization_factor = np.sqrt(periods_per_year)
    logger.info(f"Using annualization factor of {annualization_factor:.2f} for '{data_frequency}' data.")

    for ticker in tickers:
        try:
            if optimized_inputs_dict.get(ticker, {}).get('status') != 'success':
                continue
            
            optimized_weights = optimized_inputs_dict[ticker]['weights']
            price = price_frame[ticker].ffill()

            # Logic to ensure enough data for the longest lookback
            min_periods_needed = max([v[1] if isinstance(v, tuple) else v for v in rule_variations]) + vol_lookback_for_position_sizing
            if price.dropna().empty or len(price.dropna()) < min_periods_needed:
                 logger.warning(f"Skipping {ticker}: Not enough data ({len(price.dropna())}) for lookbacks.")
                 continue

            signal_df = pd.DataFrame(index=price.index)

            # --- Signal Calculation (Unchanged logic, but now validated) ---
            if strategy_type.lower() == 'ewmac':
                for i, j in rule_variations:
                    raw_forecast = calc_ewma_forecast(price, Lfast=i, Lslow=j, vol_lookback=vol_lookback_for_position_sizing)
                    scalar_map = forecast_scalars or CARVER_EWMAC_SCALARS
                    fixed_scalar = scalar_map.get((i, j))
                    signal = get_signal(price, raw_forecast, None, fixed_scalar)
                    signal_df[f'EWMAC_({i},{j})'] = signal
            elif strategy_type.lower() == 'breakout':
                for horizon in rule_variations:
                    raw_forecast = calc_breakout_forecast(price, horizon=horizon)
                    scalar_map = forecast_scalars or CARVER_BREAKOUT_SCALARS
                    fixed_scalar = scalar_map.get(horizon)
                    signal = get_signal(price, raw_forecast, None, fixed_scalar)
                    signal_df[f'Breakout_{horizon}'] = signal
            else:
                raise ValueError(f"Unknown strategy_type: '{strategy_type}'.")

            div_multiplier = diversification_multiplier if diversification_multiplier is not None else 1.24
            weighted_signal = np.dot(signal_df.fillna(0), optimized_weights)
            weighted_signal = pd.Series(weighted_signal, index=signal_df.index) * div_multiplier

            final_capped_signal = weighted_signal.clip(-20, 20) if cap_final_forecast else weighted_signal
            signal_df['weighted_signal'] = final_capped_signal
            
            # --- Position Sizing (risk-targeted) ---
            positions, positions_usd = _position_size_from_forecast(
                price,
                weighted_signal,
                portfolio_value=portfolio_value,
                target_vol_annual=target_vol_annual,
                data_frequency=data_frequency,
                vol_lookback=vol_lookback_for_position_sizing,
                max_leverage=max_leverage,
            )
            
            pnl_results = get_pnl(price, trades=None, positions=positions)
            
            # --- New Return Structure ---
            pnl_dict[ticker] = {
                'pnl_data': pnl_results,
                'positions': positions, # Add the positions Series
                'positions_usd': positions_usd, # USD notional series (signed)
                'direction': np.sign(positions) # Add the direction Series (-1, 0, 1)
            }
            signal_dict[ticker] = signal_df # Changed from list to single DF

            # --- Reporting Logic ---
            pnl_series = pnl_results[3] # base_ccy_returns
            if pnl_series.dropna().empty: continue
            
            # --- LOG 2 & FIX: Validate and fix Sharpe Ratio calculation ---
            returns = pnl_series / portfolio_value
            returns_std = returns.std()
            if returns_std == 0 or np.isnan(returns_std):
                unannualized_sr = np.nan
                correctly_annualized_sr = np.nan
            else:
                unannualized_sr = returns.mean() / returns_std
                correctly_annualized_sr = unannualized_sr * annualization_factor
            logger.debug(f"[{ticker}] Unannualized SR: {unannualized_sr:.4f}, Correctly Annualized SR: {correctly_annualized_sr:.2f}")

            # --- New Reporting Structure ---
            # Report position size in USD notional (not coin units) for bot messages.
            final_signal = round(signal_df['weighted_signal'].iloc[-1], 2)
            last_notional = float(positions_usd.iloc[-1]) if len(positions_usd) else 0.0
            final_position_size = round(last_notional, 2)
            final_direction = 'LONG' if final_position_size > 0 else ('SHORT' if final_position_size < 0 else 'FLAT')
            
            add_in = {
                'Name': ticker_dict.get(ticker, ticker),
                'Position Size': final_position_size,
                'Direction': final_direction,
                'Signal': final_signal,
                'SR': round(correctly_annualized_sr, 2)
            }
            email_df = pd.concat([email_df, pd.DataFrame(add_in, index=[0])], ignore_index=True)

        except Exception as e:
            logger.error(f"Critical error processing ticker {ticker}: {e}", exc_info=True)
            continue

    if pnl_dict_gen:
        return pnl_dict, signal_dict, email_df

    return email_df



def run_simple_backtest_for_grid(price_frame, signals_df, vol_lookback, 
                                 n_longs=5, n_shorts=5, rebal_freq='D', fee_bps=5.0):
    """
    Runs a simplified backtest for a given set of signals and vol lookback.
    Returns performance metrics dictionary.
    """
    if signals_df.empty or price_frame.empty:
        return {'sharpe': 0, 'cagr': 0, 'max_drawdown': -1}

    # Determine rebalance dates
    rebalance_dates = pd.Series(index=price_frame.index, data=price_frame.index)
    rebalance_dates = rebalance_dates.resample(rebal_freq).first().dropna()
    rebalance_dates = rebalance_dates[rebalance_dates >= price_frame.index.min()]
    rebalance_dates = rebalance_dates[rebalance_dates <= price_frame.index.max()]
    rebalance_dates = rebalance_dates.intersection(signals_df.index) # Ensure rebal dates have signals

    if rebalance_dates.empty:
        return {'sharpe': 0, 'cagr': 0, 'max_drawdown': -1}

    portfolio_returns = pd.Series(index=price_frame.index, dtype=float).fillna(0)
    current_weights = pd.Series(dtype=float)

    for i, rebal_date in enumerate(rebalance_dates):
        start_date = rebal_date
        end_date = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else price_frame.index[-1]

        # 1. Determine Target Weights at rebal_date
        latest_signals = signals_df.loc[rebal_date].dropna()
        longs = latest_signals[latest_signals > 0].nlargest(n_longs)
        shorts = latest_signals[latest_signals < 0].nsmallest(n_shorts)
        
        target_tickers = pd.concat([longs, shorts])
        
        if target_tickers.empty:
            next_weights = pd.Series(dtype=float)
        else:
            # Equal weight for this simple backtest
            next_weights = pd.Series(0.0, index=target_tickers.index)
            next_weights.loc[longs.index] = 1.0 / (n_longs + n_shorts)
            next_weights.loc[shorts.index] = -1.0 / (n_longs + n_shorts)

            # --- Position Sizing with Volatility ---
            # This is where the `vol_lookback` comes into play.
            # We scale weights inversely to volatility.
            vol_df = price_frame[target_tickers.index].pct_change().rolling(window=vol_lookback).std()
            
            # Use volatility at rebal_date (or just before)
            if rebal_date in vol_df.index:
                asset_vol = vol_df.loc[rebal_date]
                asset_vol = asset_vol.replace(0, np.nan).fillna(method='bfill').fillna(0.0001) # Handle zero/NaN vol
                
                # Inverse vol weighting (simplified, needs normalization to target total weight)
                inv_vol = 1.0 / asset_vol
                total_inv_vol = inv_vol.sum()
                if total_inv_vol > 0:
                     # Re-scale weights based on inv_vol but keep long/short signs and counts
                     next_weights_vol_scaled = next_weights * inv_vol
                     # Normalize to keep total long = 0.5 and total short = -0.5 (example target leverage 1)
                     total_long_w = next_weights_vol_scaled[next_weights_vol_scaled > 0].sum()
                     total_short_w = next_weights_vol_scaled[next_weights_vol_scaled < 0].sum()
                     if total_long_w > 0: next_weights_vol_scaled[next_weights_vol_scaled > 0] /= (total_long_w * 2)
                     if total_short_w < 0: next_weights_vol_scaled[next_weights_vol_scaled < 0] /= (abs(total_short_w) * 2)
                     next_weights = next_weights_vol_scaled.fillna(0)

        # 2. Calculate Transaction Costs & Update Weights
        turnover = (next_weights.reindex(current_weights.index).fillna(0) - 
                    current_weights.reindex(next_weights.index).fillna(0)).abs().sum()
        tx_cost_pct = turnover * (fee_bps / 10000.0)
        current_weights = next_weights.copy()
        
        # 3. Calculate Returns for the Holding Period
        period_index = price_frame.loc[start_date:end_date].index
        
        if not current_weights.empty and len(period_index) > 1:
            asset_returns = price_frame[current_weights.index].pct_change().loc[period_index].fillna(0)
            # Align weights and returns (weights are fixed, returns change daily)
            # Weights are applied starting the day *after* rebalance date
            weighted_returns = asset_returns.dot(current_weights.reindex(asset_returns.columns).fillna(0))
            portfolio_returns.loc[period_index] = weighted_returns
            
            # Apply transaction cost at the start of the period
            portfolio_returns.loc[start_date] -= tx_cost_pct
            

    # Calculate Metrics using QuantStats
    portfolio_returns = portfolio_returns.fillna(0)
    if portfolio_returns.std() == 0:
        return {'sharpe': 0, 'cagr': 0, 'max_drawdown': -1}
        
    try:
        # Infer periods per year based on data frequency (assuming 4-hour)
        periods_per_year = 252 * 6 
        sharpe = qs.stats.sharpe(portfolio_returns, periods=periods_per_year, annualize=True)
        cagr = qs.stats.cagr(portfolio_returns, periods=periods_per_year)
        max_drawdown = qs.stats.max_drawdown(portfolio_returns)
        return {'sharpe': sharpe, 'cagr': cagr, 'max_drawdown': max_drawdown}
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        return {'sharpe': 0, 'cagr': 0, 'max_drawdown': -1}


def run_grid_search():
    """
    Main function to run the grid search sensitivity analysis.
    """
    logger.info("--- Starting Sensitivity Analysis Grid Search ---")

    # --- 1. Define Parameter Grids ---
    vol_lookbacks_to_test = [60, 120, 180, 240, 360]  # Example: 10, 20, 30, 40, 60 days in 4h bars
    ma_multipliers_to_test = [1.0, 2.0, 3.0, 4.0, 6.0] # Example: Original, 2x ... 6x (daily equiv)

    # Base EWMAC factors
    ewmac_base_factors = [(2,8), (4,16), (8,32), (16,64), (32,128)]

    # --- 2. Load Data ---
    # Replace with your actual data loading mechanism
    data_proc = DataProcessor()
    price_frame = data_proc.load_price_data("all_ticker_prices.pkl") # Assuming you have this file
    if price_frame.empty:
        logger.error("Price data not found or empty. Please ensure data is available.")
        # Optionally, add data fetching here
        return
    tickers = price_frame.columns.tolist()
    logger.info(f"Loaded price data for {len(tickers)} tickers. Shape: {price_frame.shape}")

    # --- 3. Load or Define Fixed x_i Parameters ---
    # Ideally, load these from an Optuna run. For this example, use placeholders.
    # You MUST replace this with your actual, meaningful x_i parameters.
    x_i_params_placeholder = {
        ticker: {'inputs': [10.0, 10.0, 20.0, 20.0, 30.0]} for ticker in tickers
    }
    logger.warning("USING PLACEHOLDER x_i parameters. Replace with your actual optimized values.")
    x_i_params_to_use = x_i_params_placeholder

    # --- 4. Run Grid Search ---
    results_list = []
    grid = list(itertools.product(vol_lookbacks_to_test, ma_multipliers_to_test))
    
    for vol_lookback, ma_multiplier in grid:
        logger.info(f"Testing: Vol Lookback = {vol_lookback}, MA Multiplier = {ma_multiplier:.1f}")

        # 1. Calculate current EWMAC factors
        current_ewmac_factors = [
            (int(round(f * ma_multiplier)), int(round(s * ma_multiplier))) 
            for f, s in ewmac_base_factors
        ]
        # Ensure fast < slow and > 1
        current_ewmac_factors = [(max(2, f), max(f + 2, s)) for f, s in current_ewmac_factors]

        # 2. Generate Signals
        signals_df = generate_signals_for_grid(price_frame, tickers, x_i_params_to_use, current_ewmac_factors)

        # 3. Run Backtest
        metrics = run_simple_backtest_for_grid(
            price_frame, 
            signals_df, 
            vol_lookback,
            n_longs=strategy_params.N_ASSETS_LONG,
            n_shorts=strategy_params.N_ASSETS_SHORT,
            rebal_freq='D', # Daily rebalancing
            fee_bps=5.0 # Example fee
        )
        
        results_list.append({
            'vol_lookback': vol_lookback,
            'ma_multiplier': ma_multiplier,
            'sharpe': metrics['sharpe'],
            'cagr': metrics['cagr'],
            'max_drawdown': metrics['max_drawdown']
        })

    # --- 5. Process and Display Results ---
    results_df = pd.DataFrame(results_list)
    print("\n--- Grid Search Results ---")
    print(results_df)

    # --- 6. Visualize Results (Heatmaps) ---
    if not results_df.empty:
        try:
            plt.figure(figsize=(18, 5))

            # Heatmap for Sharpe Ratio
            plt.subplot(1, 3, 1)
            sharpe_pivot = results_df.pivot(index='vol_lookback', columns='ma_multiplier', values='sharpe')
            sns.heatmap(sharpe_pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
            plt.title('Sharpe Ratio')

            # Heatmap for CAGR
            plt.subplot(1, 3, 2)
            cagr_pivot = results_df.pivot(index='vol_lookback', columns='ma_multiplier', values='cagr')
            sns.heatmap(cagr_pivot, annot=True, fmt=".2f", cmap="plasma", linewidths=.5)
            plt.title('CAGR (%)') # Assuming CAGR is returned as decimal

            # Heatmap for Max Drawdown
            plt.subplot(1, 3, 3)
            mdd_pivot = results_df.pivot(index='vol_lookback', columns='ma_multiplier', values='max_drawdown')
            sns.heatmap(mdd_pivot, annot=True, fmt=".2f", cmap="magma", linewidths=.5)
            plt.title('Max Drawdown (%)') # Assuming MDD is returned as negative decimal

            plt.tight_layout()
            plt.suptitle('Sensitivity Analysis Heatmaps', y=1.05, fontsize=16)
            plt.savefig("sensitivity_analysis.png")
            logger.info("Saved sensitivity analysis plot to sensitivity_analysis.png")
            plt.show() # Display the plot

        except Exception as e:
            logger.error(f"Error during plotting: {e}")

    logger.info("--- Sensitivity Analysis Finished ---")



# --- Display Functions (from paste.txt [1], slightly adapted for robustness) ---
def color_tx(s):
    return ['color: white' if abs(val) > 50 else 'color: black' for val in s]

def color_tx2(s):
    return ['color: white' if abs(val) > 0.5 else 'color: black' for val in s]

def b_g(s, cmap_name='RdBu', norm_min=-100, norm_max=100):
    cmap_obj = plt.cm.get_cmap(cmap_name) # Use plt.cm.get_cmap
    norm = colors.Normalize(vmin=norm_min, vmax=norm_max) # Use vmin, vmax
    hex_colors = [colors.rgb2hex(cmap_obj(x)) for x in norm(s.values)]
    return ['background-color: %s' % color for color in hex_colors]

def b_g2(s, cmap_name='RdBu', norm_min=-1, norm_max=1):
    cmap_obj = plt.cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=norm_min, vmax=norm_max)
    hex_colors = [colors.rgb2hex(cmap_obj(x)) for x in norm(s.values)]
    return ['background-color: %s' % color for color in hex_colors]

def get_colored_df(df_to_style):
    # Ensure DataFrame is not empty and has the required columns
    if df_to_style.empty or not all(col in df_to_style.columns for col in ['Signal', 'SR']):
        return df_to_style # Return original if styling cannot be applied
    
    # From original: cmap=ListedColormap(sns.color_palette("RdBu",10).as_hex())
    # Simpler: cmap='RdBu' (matplotlib will handle it)
    styled_df = df_to_style.style \
        .apply(b_g, cmap_name='RdBu', subset=['Signal']) \
        .apply(b_g2, cmap_name='RdBu', subset=['SR']) \
        .apply(color_tx, subset=['Signal']) \
        .apply(color_tx2, subset=['SR']) \
        .format({'SR': '{:.2f}', 'Signal': '{:.1f}%'})
    try: # hide_index() for older pandas, hide() for newer
        styled_df = styled_df.hide(axis="index")
    except AttributeError:
        try: styled_df = styled_df.hide_index()
        except AttributeError: pass
    return styled_df

# --- New function to fetch historical data for a list of tickers ---
def fetch_historical_data(tickers_list, freq='1d', years_of_data=3):
    end_date_str = dt.datetime.now().strftime("%Y-%m-%d")
    start_date_str = get_prev_date(end_date_str, 365 * years_of_data)
    
    all_prices_df = pd.DataFrame()
    successfully_fetched_tickers = []

    for ticker in tqdm.tqdm(tickers_list, desc="Fetching Historical Data", file=sys.stdout, leave=True):
        try:
            # print(f"Fetching {ticker} from {start_date_str} to {end_date_str}") # Debug
            ticker_data_df = gen_lookback_df(ticker, start_date_str, end_date_str, freq=freq)
            if not ticker_data_df.empty and 'Close' in ticker_data_df.columns:
                close_prices = ticker_data_df[['Close']].copy()
                close_prices.columns = [ticker]
                if all_prices_df.empty:
                    all_prices_df = close_prices
                else:
                    all_prices_df = pd.concat([all_prices_df, close_prices], axis=1, join='outer')
                successfully_fetched_tickers.append(ticker)
            else:
                # print(f"No data or 'Close' column missing for {ticker}. Skipping.") # Debug
                pass
        except BinanceAPIException as e:
            if e.code == -1121: # Invalid symbol
                print(f"Invalid symbol or no data for: {ticker}. Skipping. Error: {e}")
            else:
                print(f"API Error for {ticker}: {e}. Skipping.")
        except Exception as e:
            print(f"Error processing {ticker}: {e}. Skipping.")
            
    # Reindex to ensure consistent datetime index, fill gaps if any, though join='outer' handles alignment
    if not all_prices_df.empty:
         all_prices_df = all_prices_df.ffill() # Forward fill to handle NaNs from non-overlapping series

    return all_prices_df, successfully_fetched_tickers


def plot_single_adj(df, title, y_title, cols, y2=False, y2_col=None, y2_title=None, shaded=None, scatter=False, width=800, height=600):
    """
    Plots time-series data with options for secondary y-axis, shaded regions, and scatter plots.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        title (str): Title of the plot.
        y_title (str): Title for the primary y-axis.
        cols (list or str): Column name(s) to plot on the primary y-axis.
        y2 (bool): If True, enables the secondary y-axis. Defaults to False.
        y2_col (list or str): Column name(s) to plot on the secondary y-axis. Required if y2 is True.
        y2_title (str, optional): Title for the secondary y-axis. Used only if y2 is True. Defaults to None.
        shaded (list, optional): List of lists containing (start_date, end_date) tuples for shaded regions.
                                 If one list is provided, uses 'LightGreen'.
                                 If two lists [pos_regions, neg_regions] are provided, uses 'LightGreen' for pos, 'LightSalmon' for neg.
                                 Defaults to None.
        scatter (bool): If True, plots the first column in 'cols' as a scatter plot instead of lines. Defaults to False.
        width (int): Width of the plot in pixels. Defaults to 800.
        height (int): Height of the plot in pixels. Defaults to 600.
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    if not scatter:
        fig = make_subplots(specs=[[{"secondary_y": y2}]])

        # --- Primary Y-axis Data ---
        primary_cols = [cols] if isinstance(cols, str) else cols
        for col in primary_cols:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                hovertemplate='%{x|%Y-%m-%d}: %{y:,.2f}<extra></extra>' # Adjusted hover template format
            ), secondary_y=False) # Explicitly set primary axis

        # --- Secondary Y-axis Data ---
        if y2:
            if y2_col is None:
                 print("Warning: y2 is True but y2_col is not specified. Secondary axis will be empty.")
            else:
                secondary_cols = [y2_col] if isinstance(y2_col, str) else y2_col
                for var in secondary_cols:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[var],
                        mode='lines',
                        name=var,
                        hovertemplate='%{x|%Y-%m-%d}: %{y:,.2f}<extra></extra>', # Adjusted hover template format
                        # yaxis='y2' # Not strictly needed when using secondary_y=True
                    ), secondary_y=True)

        # --- Add Shaded Regions ---
        if shaded and isinstance(shaded, list):
            # Check if it's a list of lists (for pos/neg) or just one list of regions
            if shaded and isinstance(shaded[0], list): # Assumes [[pos], [neg]] structure
                 # Negative values (assuming index 1)
                if len(shaded) > 1:
                    for start_date, end_date in shaded[1]:
                        fig.add_vrect(
                            x0=start_date, x1=end_date,
                            fillcolor="LightSalmon", opacity=0.5,
                            layer="below", line_width=0,
                        )
                # Positive values (assuming index 0)
                for start_date, end_date in shaded[0]:
                    fig.add_vrect(
                        x0=start_date, x1=end_date,
                        fillcolor="LightGreen", opacity=0.5,
                        layer="below", line_width=0,
                    )
            else: # Assume a single list of regions (treat as positive)
                for start_date, end_date in shaded:
                    fig.add_vrect(
                        x0=start_date, x1=end_date,
                        fillcolor="LightGreen", opacity=0.5,
                        layer="below", line_width=0,
                    )

        # --- Update Layout ---
        layout_options = {
            'hoverlabel_bgcolor': '#DAEEED',
            'title': title,
            'title_font_size': 15,
            'title_font_color': "darkblue",
            'title_x': 0.5, # Center title
            'dragmode': 'zoom',
            'hovermode': "x unified",
            'legend': dict(orientation='h', xanchor="center", x=0.5, y=1), # Adjust legend position
            'yaxis': dict(title=y_title), # Primary y-axis title
            'width': width,
            'height': height,
            'margin': go.layout.Margin(l=40, r=40, b=40, t=80) # Adjusted top margin for title/legend
        }

        # Conditionally add secondary y-axis title
        if y2 and y2_title:
            layout_options['yaxis2'] = dict(title=y2_title)

        fig.update_layout(**layout_options)

        # --- Configure X-axis ---
        fig.update_xaxes(
            #tickformat="%b %Y",  # Show Month and Year for clarity
            #dtick="M1",       # One tick per month (adjust if needed)
            #ticklabelmode="period",
            showgrid=True,
            zeroline=True,
            showline=True,
            rangeslider_visible=True # Keep range slider
        )

    else: # Scatter plot case
        col = cols[0] if isinstance(cols, list) else cols
        fig = go.Figure(data=go.Scatter(
            x=df.index,
            y=df[col],
            mode='markers',
             hovertemplate='%{x|%Y-%m-%d}: %{y:,.2f}<extra></extra>' # Adjusted hover template format
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date', # Changed from 'date' for capitalization
            yaxis_title=y_title, # Use y_title parameter
            width=width,
            height=height,
            dragmode='zoom',
            hovermode="closest", # Changed hovermode for scatter
            title_x=0.5,
            margin=go.layout.Margin(l=40, r=40, b=40, t=80)
        )

        fig.update_xaxes(
            #tickformat="%b %Y",
            #dtick="M1",
            #ticklabelmode="period",
            showgrid=True
        )

    fig.show()

