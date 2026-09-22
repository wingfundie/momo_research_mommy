# Telegram bot command manual

All commands work with or without a leading slash. These are equivalent:

```text
mc SOL 6m
/mc SOL 6m
```

Arguments can be entered in any order unless a ticker is required. For ticker commands, put the ticker first.

## Lookbacks

Historical commands accept flexible lookbacks from one day through ten years:

| Format | Meaning | Examples |
|---|---|---|
| `Nd` | Calendar days | `7d`, `45d`, `180d` |
| `Nw` | Seven-day weeks | `4w`, `12w` |
| `Nm` | Thirty-day months | `3m`, `6m`, `18m` |
| `Ny` | 365-day years | `1y`, `2y`, `5y` |
| `all` | All available history | `all` |

Lookbacks change the analysis or chart window. They do not change the pinned corrected-XSec20 parameters: EWMAC 2/8 through 32/128, IC20, vol60, 15% target volatility, 2x gross cap, 25% ticker cap, next-open activation, costs or funding.

## Momentum signals

| Command | Arguments | Result |
|---|---|---|
| `mom_sig_str` | `[N] [lookback]` | Current relative-strength ranking. The selected-window standalone Sharpe is included. |
| `mom_sig_sr` | `[N] [lookback]` | Long and short signals ranked by standalone annualized Sharpe over the chosen window. |
| `mom_l1` | `[N] [lookback]` | Layer-1 subset. |
| `mom_eth_beta` | `[N] [lookback]` | Ethereum-ecosystem subset. |
| `mom_sol_beta` | `[N] [lookback]` | Solana-ecosystem subset. |
| `mom_meme` | `[N] [lookback]` | Meme subset. |
| `mom_sr_100` | `[N] [lookback]` | Broad-universe Sharpe view. |
| `mom_str_100` | `[N] [lookback]` | Broad-universe strength view. |

`N` is the number of rows per side, from 1 to 25. It defaults to 10. Examples:

```text
mom_sig_sr 15 90d
mom_sig_str 10 6m
mom_l1 8 1y
```

Append `legacy` to use the previous per-ticker momentum engine:

```text
mom_sig_str 10 legacy
mc SOL 6m legacy
```

## Charts and analytics

| Command | Arguments | Result |
|---|---|---|
| `mc` | `TICKER [lookback]` | Standalone trend vs buy-and-hold, price/forecast, and XSec rank/held-weight charts. Default: `all`. |
| `distribution` | `[lookback]` | All ticker forecast paths, 25th/75th percentiles, interquartile band, universe mean and breadth. Default: `180d`. |
| `performance` | `[lookback] [ASSET]` | Portfolio return, drawdown and optional comparison with any Binance USDT asset. Default: `1y`. |
| `risk` | `[share|usd] [lookback]` | Long, short, gross and net exposure history. Default: `1y`. |
| `changes` | `[N] [lookback]` | Largest forecast, rank and target-weight changes from the selected historical date to the latest date. Default: `1d`. |

Examples:

```text
mc BTC 12w
distribution 45d
performance 2y BTC
performance all ETH
risk usd 6m
changes 20 14d
```

The benchmark is display-only. It never enters the model universe or changes signal ranks.

## Portfolio construction

| Command | Arguments | Result |
|---|---|---|
| `portfolio` | `[share|usd] [lookback]` | Current target weights, notionals and exchange-rounded quantities. A lookback adds exposure history. |
| `rebalance` | none | Read-only OPEN, CLOSE, FLIP, REDUCE and INCREASE changes from held model weights to targets. |
| `portfolio_momo` | none | Compares actual Binance positions with current trend and XSec directions. |

Sizing uses a fixed $100,000 model base. Quantities respect Binance step size, minimum quantity and minimum notional when exchange metadata is available. These commands do not place orders.

## Model and operations

| Command | Result |
|---|---|
| `model` | Pinned configuration, costs, validation, holdout and net-exposure diagnostics. |
| `health` | Snapshot freshness, data cutoff, universe date and refresh fallback status. |
| `manual` or `help` | In-bot version of this manual. |

## Breakout commands

The breakout engine remains separate from corrected XSec20 momentum:

```text
breakout_sig_sr
breakout_sig_str
breakout_l1
breakout_eth_beta
breakout_sol_beta
breakout_meme
breakout_sr_100
breakout_str_100
breakout_pairs
portfolio_breakout
bc BTC 90d
```

`bc TICKER [lookback]` controls the breakout chart window. Other breakout signal tables retain their existing optimized-strategy calculation and ranking behavior.

## Quick recipes

```text
# Ten strongest current relative signals with six-month standalone SR
mom_sig_str 10 6m

# Top 15 trend strategies by trailing-90-day standalone SR
mom_sig_sr 15 90d

# Current portfolio plus one year of exposure history in dollars
portfolio usd 1y

# Portfolio versus BTC over two years
performance 2y BTC

# How the universe signal distribution evolved over 12 weeks
distribution 12w

# SOL analytics over six months
mc SOL 6m

# Largest two-week target changes
changes 20 14d
```
