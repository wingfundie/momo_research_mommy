from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, Application
import pandas as pd
import requests
import json
import numpy as np
from datetime import datetime, date, time, timedelta, timezone
#from datetime import datetime as dt
import time as Time
from bs4 import BeautifulSoup
from telegram.ext import CallbackContext
import cloudscraper
import asyncio 
from momo_bot.exchange import get_binance_client
from momo_bot.binance_rate_limit import get_rest_guard
import statsmodels.api as sm
import os

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
def format_multi_text(left_df, right_df, left_title="LONGS", right_title="SHORTS"):
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

# BETA TABLE
def format_multi_beta(left_df, right_df, left_title="LONGS", right_title="SHORTS", beta=0):
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
    
    # Calculate total width for the net beta exposure line
    total_width = max(len(table_left), len(table_right))
    net_beta_exposure_line = f"NET BETA EXPOSURE: {beta}".center(total_width)
    
    # Combine both tables with a space, the net beta exposure line, and bold the line
    formatted_text = f"<pre>{table_left}</pre>  <pre>{table_right}</pre>\n\n{net_beta_exposure_line}"
    return formatted_text



# WEb Scraper
def scrape_etf_flows():
    site = 'https://farside.co.uk/?p=997'
    yst = (datetime.today() - timedelta(days= 1)).strftime('%d %b %Y')
    test_date = (datetime.today()).strftime('%d %b %Y')
    #stored_data = pd.read_pickle(r"C:\Users\ezra.soong\OneDrive - L3 Management Pte Ltd\python\etf_data\btc_etf_flows.pkl")
    r = requests.get(site)
    soup = BeautifulSoup(r.content, 'lxml') 
    dates = soup.find_all('span', class_ ='tabletext')

    date_is_in =  any(yst in span.get_text() for span in dates)
    headers = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'GBTC', 'DEFI','Total']
    flow_store = []
    extract = False

    # Only start looping if the date is out already
    if date_is_in:
        for entry in dates:
            text = entry.get_text()
            stopper = "Total"
        # print(text)

            if text == yst:
                print("LMAO")
                extract = True
                continue      

            if text == stopper and extract:
                print(flow_store)
                break 

            if extract:
                flow_store.append(text)
                #print(flow_store)
                

        flow_frame = pd.DataFrame([flow_store], columns= headers).T.reset_index()
        flow_frame.columns = ['Date', f'{yst}']

      #  if flow_frame.equals(stored_data):
      #      return 404

       # flow_frame.to_pickle(r"C:\Users\ezra.soong\OneDrive - L3 Management Pte Ltd\python\etf_data\btc_etf_flows.pkl")

        # Format the DataFrame and print the result
        table = format_as_monospaced_text(flow_frame)
        return table
    else:
        return 404
    

# Get ETF Flows
async def etf_flows(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    flows = scrape_etf_flows()
    await update.message.reply_text(text= flows, parse_mode='HTML' )

# Get Futures positions
# Get Futures positions
def get_futures_positions(client=None):
    cl = client or get_binance_client()
    posit_info = get_rest_guard().call(
        "futures_position_information",
        5,
        cl.futures_position_information,
    )

    # Function to attempt to convert a value to float
    def float_con(value):
        try:
            return float(value)
        except ValueError:
            return value

    # Convert all values in all dictionaries to floats where possible
    posit_info = [{k: float_con(v) for k, v in d.items()} for d in posit_info]

    # Filter for open positions
    positions = [i['symbol'] for i in posit_info if float(i['positionAmt']) > 0 or float(i['positionAmt']) < 0]
    positions = [d for d in posit_info if d['symbol'] in positions]
    open_positions = [d for d in posit_info if float(d.get('positionAmt', '0')) != 0.0]
    return open_positions

# Round numeric
def round_numeric(df, decimal_places=2):
    """
    Round numeric values in a pandas DataFrame to a specified number of decimal places.
    
    Parameters:
    df (DataFrame): The pandas DataFrame.
    decimal_places (int): Number of decimal places to round to (default is 2).
    
    Returns:
    DataFrame: A new DataFrame with rounded numeric values.
    """
    rounded_df = df.map(lambda x: round(x, decimal_places) if pd.api.types.is_numeric_dtype(type(x)) else x)
    
    return rounded_df

# Generate long short frames
def gen_ls_frame(open_positions, posit_frame = False):
    print('RUNNING EXPO...........')
    # Getting all positions
    positions_df = pd.DataFrame(open_positions)
    positions_df = positions_df.apply(
        lambda col: pd.to_numeric(col) if col.dtype == object and pd.to_numeric(col, errors='coerce').notna().all() else col
    )

    # Get relevant columns
    profit_df = round_numeric(positions_df.loc[:, ['symbol', 'positionSide', 'notional', 'unRealizedProfit']])

    # Return profit_df
    if posit_frame:
        return profit_df
    
    # Create Long/Short Frames
    long_df = profit_df[profit_df['positionSide']== 'LONG']
    long_df = long_df.sort_values(by='notional', ascending=False)
    long_row = {
        'symbol': 'TOTAL LONG',
        'notional': f'{round(sum(long_df["notional"]),2)}',
        'positionSide': 'LONG',
        'unRealizedProfit': f'{round(sum(long_df["unRealizedProfit"]),2)}'
    }
    long_df = pd.concat([long_df, pd.DataFrame([long_row])])


    short_df = profit_df[profit_df['positionSide']== 'SHORT']
    short_df = short_df.sort_values(by='notional', ascending=True)
    short_row = {
        'symbol': 'TOTAL SHORT',
        'notional': f'{round(sum(short_df["notional"]), 2)}',
        'positionSide': 'SHORT',
        'unRealizedProfit': f'{round(sum(short_df["unRealizedProfit"]), 2)}'
    }
    short_df = pd.concat([short_df, pd.DataFrame([short_row])])

    long_df = long_df.loc[:, ['symbol', 'notional', 'unRealizedProfit']]
    short_df = short_df.loc[:, ['symbol', 'notional', 'unRealizedProfit']]
    
    return long_df, short_df

# Net Exposure
def net_exposure(profit_df):
    net_exposure = pd.DataFrame([[0, 0, 0, 0, 0]], columns= ['sign', 'long_expo', 'short_expo','net_expo', 'upnl'])
    net_exposure['long_expo'] =  round(sum(profit_df[profit_df['notional'] > 0]['notional']), 2)
    net_exposure['short_expo'] = round(sum(profit_df[profit_df['notional'] < 0]['notional']), 2)
    net_exposure['net_expo'] = round(sum(profit_df['notional']), 2)
    net_exposure['upnl'] = round(sum(profit_df['unRealizedProfit']), 2)
    net_exposure['sign'] = np.sign(net_exposure['net_expo'])
    return net_exposure

def get_utc(date):
    # Parse the date string into a datetime object
    date_obj = datetime.strptime(date, "%Y-%m-%d")

    # Convert datetime object to UTC timezone
    date_utc = date_obj.astimezone(timezone.utc)

    # Convert datetime object back to string in UTC format
    date_utc_str = date_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    return date_utc_str

def get_prev_date(date):
    # Parse the date string into a datetime object
    date_obj = datetime.strptime(date, "%Y-%m-%d")

    # Calculate the date 30 days before
    date = date_obj - timedelta(days=30)

    # Format the date in the desired format
    date = date.strftime("%Y-%m-%d")

    return date


def gen_lookback_df(ticker, start_date, end_date, freq = '4h', client=None):
    cl = client or get_binance_client()
    headers = ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    historicals = get_rest_guard().call(
        "futures_historical_klines",
        10,
        lambda: cl.futures_historical_klines(
            ticker,
            freq,
            start_str=get_utc(start_date),
            end_str=get_utc(end_date),
        ),
    )
    ticker_df = pd.DataFrame(historicals, columns= headers)
    ticker_df['time'] = pd.to_datetime(ticker_df['time'], unit = 'ms')
    ticker_df = ticker_df.set_index('time')
    ticker_df = ticker_df.astype('float')
    
    return ticker_df

# Calculate Beta
def calc_beta(df_market, df_coin):
    # Ensure dataframes have the same index
    btc_rets =  df_market.pct_change().dropna()
    df_rets = df_coin.pct_change().dropna()

    df_market, df_coin = btc_rets.align(df_rets, join='inner')

    # Add constant to independent variable (market returns)
    X = sm.add_constant(df_market.values)
    y = df_coin.values

    # Perform OLS regression
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract beta coefficient
    beta = results.params[1]

    return float(beta)

# GEnerate L/S Beta Port
def gen_ls_beta():
    # Input date string
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = get_prev_date(end_date)

    # Get BTC frame
    btc_frame = gen_lookback_df('BTCUSDT', start_date, end_date, freq= '4h').loc[:, ['Close']]

    # Get open positions
    op = get_futures_positions()
    

    # Exclude the last long short row
    long_df, short_df = gen_ls_frame(op)
    long_df = long_df.iloc[:-1]
    short_df = short_df.iloc[:-1]

    print('RUNNING EXPO...........')
    # Input either long or short df without the bottom frame
    def get_roll_beta(df, short = False):
        df['beta (30D)'] = 0
        df = df.set_index('symbol')
        print(df)

        # Calculate rolling beta
        for ticker in df.index:  
            print(ticker)
            coin = gen_lookback_df(ticker, start_date, end_date, freq= '4h').loc[:, ['Close']]
            #print(df)
            if short:
                df.loc[ticker, 'beta (30D)'] = round(-calc_beta(btc_frame, coin), 2)
            else:
                df.loc[ticker, 'beta (30D)'] = round(calc_beta(btc_frame, coin), 2)

        return df.reset_index()

    # Get Beta Frames
    short_beta = get_roll_beta(short_df, short= True)
    long_beta = get_roll_beta(long_df)
    print('error')
    """
    short_row = {
        'symbol': 'TOTAL SHORT BETA',
        'notional': f'{round(sum(short_beta["notional"].astype(float)),2)}',
        'unRealizedProfit': f'{round(sum(short_beta["unRealizedProfit"].astype(float)),2)}',
        'beta (30D)': f'{round(sum(short_beta["beta (30D)"].astype(float)),2)}'
    }
    short_beta = pd.concat([short_beta, pd.DataFrame([short_row])])
    print(short_beta)

    """
    # Calculate the total notional to use for calculating weights
    short_notional = short_beta['notional'].sum()
    
    # Calculate the total notional to use for calculating weights
    long_notional = long_beta['notional'].sum()

    tot_notional  = long_notional + abs(short_notional)
    short_multi = -short_notional /tot_notional
    long_multi = long_notional /tot_notional

    # Calculate the weighted beta for each position
    short_beta['w_beta'] = ((short_beta['notional'] / short_notional) * short_beta['beta (30D)'])* short_multi
    print(short_beta['w_beta'])
    short_beta['w_beta'] = pd.to_numeric(short_beta['w_beta'], errors='coerce').round(2)
    

    short_row = {
        'symbol': 'TOTAL SHORT BETA',
        'notional': round(short_notional, 2),
        'unRealizedProfit': round(short_beta['unRealizedProfit'].sum(), 2),  # This assumes you want to sum unrealized profit as well
        'beta (30D)': round(short_beta['beta (30D)'].sum(), 2),  # This is not the correct way to get portfolio beta, just summing individual betas
        'w_beta': round(short_beta['w_beta'].sum(), 2)
    }
    short_beta = pd.concat([short_beta, pd.DataFrame([short_row])])
    print(short_beta)
    short_beta['w_beta'] = round(short_beta['w_beta'].astype(float), 2)



    print('error')
    """
    long_row = {
        'symbol': 'TOTAL LONG BETA',
        'notional': f'{round(sum(long_beta["notional"].astype(float)),2)}',
        'unRealizedProfit': f'{round(sum(long_beta["unRealizedProfit"].astype(float)),2)}',
        'beta (30D)': f'{round(sum(long_beta["beta (30D)"].astype(float)),2)}'
    }
    long_beta = pd.concat([long_beta, pd.DataFrame([long_row])])
    print(long_beta)
    """

    
    # Calculate the weighted beta for each position
    long_beta['w_beta'] = ((long_beta['notional'] / long_notional) * long_beta['beta (30D)']) * long_multi 
    print(long_beta['w_beta'])
    long_beta['w_beta'] = pd.to_numeric(long_beta['w_beta'], errors='coerce').round(2).round(2)

    long_row = {
        'symbol': 'TOTAL LONG BETA',
        'notional': round(long_notional, 2),
        'unRealizedProfit': round(long_beta['unRealizedProfit'].sum(), 2),  # This assumes you want to sum unrealized profit as well
        'beta (30D)': round(long_beta['beta (30D)'].sum(), 2),  # This is not the correct way to get portfolio beta, just summing individual betas
        'w_beta': round(long_beta['w_beta'].sum(), 2)
    }
    long_beta = pd.concat([long_beta, pd.DataFrame([long_row])])
    long_beta['w_beta'] = round(long_beta['w_beta'].astype(float), 2)
    print(long_beta)

    long_beta.columns = ['coin', 'not', 'upnl', 'beta', 'w_beta']
    short_beta.columns = ['coin', 'not', 'upnl', 'beta', 'w_beta']
  
    return long_beta, short_beta

# Get ETF Flows
async def etf_flows(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    flows = scrape_etf_flows()
    await update.message.reply_text(text= flows, parse_mode='HTML' )

# Get Futures L/S Expo
async def curr_ls_expo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    open_positions = get_futures_positions()
    long_df, short_df = gen_ls_frame(open_positions)
    table_text = format_multi_text(long_df, short_df)
    await update.message.reply_text(text= table_text, parse_mode='HTML' )

# Get Futures Net Expo
async def curr_net_expo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    open_positions = get_futures_positions()
    positioning = gen_ls_frame(open_positions, posit_frame= True)
    net_expo = net_exposure(positioning)
    table_text = format_as_monospaced_text(net_expo)
    await update.message.reply_text(text= table_text, parse_mode='HTML')

# Get Futures L/S Beta Expo
async def curr_ls_beta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    long_df, short_df = gen_ls_beta()
    net_beta = round(long_df.iloc[-1, -1] + short_df.iloc[-1, -1], 2) 
    table_text = format_multi_beta(long_df, short_df, beta = net_beta)
    await update.message.reply_text(text= table_text, parse_mode='HTML' )


async def expo_net(context: ContextTypes.DEFAULT_TYPE):
    open_positions = get_futures_positions()
    positioning = gen_ls_frame(open_positions, posit_frame= True)
    net_expo = net_exposure(positioning)
    table_text = format_as_monospaced_text(net_expo)
    await context.bot.send_message(chat_id=context.job.chat_id, text=table_text, parse_mode='HTML')

async def beta_net(context: ContextTypes.DEFAULT_TYPE):
    long_df, short_df = gen_ls_beta()
    net_beta = round(long_df.iloc[-1, -1] + short_df.iloc[-1, -1], 2) 
    table_text = format_multi_beta(long_df, short_df, beta = net_beta)
    await context.bot.send_message(chat_id=context.job.chat_id, text=table_text, parse_mode='HTML')

# Sends port updates
async def port_timer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    name = update.effective_chat.full_name
    context.job_queue.run_repeating(beta_net, interval = 15 * 2, first = 5, chat_id=chat_id)

# Sends expo updates
async def expo_timer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    name = update.effective_chat.full_name
    context.job_queue.run_repeating(expo_net, interval = 15 * 2, first = 5, chat_id=chat_id)


