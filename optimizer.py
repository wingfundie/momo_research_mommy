import numpy as np
import pandas as pd
import carver_optimization as co
import pickle
# from backtesting import Backtest, Strategy
# from backtesting.lib import resample_apply
# import backtrader as bt
import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException
# file: run_lookback_sweep.py
import os
from typing import List, Dict, Any, Tuple
import json
# progress bar
try:
    from tqdm.auto import tqdm
except Exception:   # fallback no-op
    def tqdm(it, **kwargs): 
        return it


# sys.path.append(r'C:\Users\HomePC\Desktop\acausal capital\momentum_run\mom bot')
# import portfolio_strategy as cta

sys.path.append(r'C:\Users\HomePC\Desktop\acausal capital\momentum_run\momo_bot_local')
import portfolio_strategy as cta

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from momo_bot.config import settings
from momo_bot.exchange import get_binance_client



def optimize_for_lookback(
    lb: int,
    tickers: List[str],
    breakout_horizons: List[int],
    tickers_price_data: pd.DataFrame,   # wide price frame: one column per ticker
    pct_vol_target: float,
    n_trials_per_ticker: int,
    *,
    # Optional Optuna knobs (used if your optimizer exposes them)
    silent: bool = True,
    n_jobs: int | None = None,
    show_progress_bar: bool = False,
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    """
    Safe per-ticker optimization for a single vol lookback (lb).
    Skips tickers that error out and continues. Returns (lb, {ticker: result,...}).
    """

    # ---------- 0) Strong input coercion / sanity ----------
    try:
        lb = int(lb)
    except Exception:
        raise ValueError(f"vol_lookback_for_position_sizing (lb) must be int, got {type(lb)}")

    # horizons must be a LIST of ints (avoid 'int has no len()' surprises)
    if isinstance(breakout_horizons, (int, np.integer)):
        breakout_horizons = [int(breakout_horizons)]
    else:
        breakout_horizons = [int(h) for h in list(breakout_horizons)]

    if not isinstance(tickers_price_data, pd.DataFrame):
        # user said each data series is a column: coerce Series → DataFrame if needed
        if isinstance(tickers_price_data, pd.Series):
            tickers_price_data = tickers_price_data.to_frame()
        else:
            raise TypeError("tickers_price_data must be a pandas DataFrame (wide, one column per ticker).")

    # coerce numeric (bad strings → NaN) to avoid dtype edge cases
    tickers_price_data = tickers_price_data.apply(pd.to_numeric, errors="coerce")

    # ---------- 1) Pre-filter tickers (presence + minimum history) ----------
    print(f"[LB {lb}] Starting breakout optimization for {len(tickers)} tickers; horizons={breakout_horizons}")

    cols = set(map(str, tickers_price_data.columns))
    tickers = [str(t) for t in tickers]

    present = [t for t in tickers if t in cols]
    missing = [t for t in tickers if t not in cols]
    if missing:
        print(f"[LB {lb}] Skipping {len(missing)} missing tickers (not in price_frame), e.g. {missing[:8]}")

    # need at least the larger of (max horizon, lookback) plus a buffer for warmup
    min_needed = max(max(breakout_horizons), lb) + 25

    ok_len, too_short = [], []
    for t in present:
        n = tickers_price_data[t].dropna().shape[0]
        (ok_len if n >= min_needed else too_short).append(t if n >= min_needed else (t, n))

    if too_short:
        print(f"[LB {lb}] Skipping {len(too_short)} short series (<{min_needed} bars), e.g. {too_short[:8]}")

    tickers_to_use = ok_len
    if not tickers_to_use:
        print(f"[LB {lb}] No tickers left after pre-filtering.")
        return lb, {}

    # ---------- 2) Optimize per ticker, isolate errors ----------
    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}

    for i, t in enumerate(tickers_to_use, 1):
        try:
            # One-ticker WIDE DataFrame: keep shape [n_rows x 1]
            px_one = tickers_price_data.loc[:, [t]].copy()

            # Call your optimizer for a **single ticker**.
            outer_workers = min(6, os.cpu_count())  # how many processes you submit
            optuna_jobs   = max(1, os.cpu_count() // outer_workers)
            # If your co.* accepts Optuna quiet params, pass them; otherwise omit.
            kwargs = dict(
                price_frame=px_one,
                tickers=[t],
                breakout_horizons=breakout_horizons,
                vol_lookback_for_position_sizing=lb,
                pct_vol_target=float(pct_vol_target),
                n_trials_per_ticker=int(n_trials_per_ticker),
                n_jobs=optuna_jobs,  # how many trials to run in parallel
            )
            # optional quiet/parallel flags if available
            if "silent" in co.optimize_breakout_weights_carver.__code__.co_varnames:
                kwargs["silent"] = bool(silent)
            if "n_jobs" in co.optimize_breakout_weights_carver.__code__.co_varnames:
                kwargs["n_jobs"] = n_jobs
            if "show_progress_bar" in co.optimize_breakout_weights_carver.__code__.co_varnames:
                kwargs["show_progress_bar"] = bool(show_progress_bar)

            out = co.optimize_breakout_weights_carver(**kwargs)

            if out and t in out:
                results[t] = out[t]
                sr = out[t].get("sharpe")
                print(f"[LB {lb}] ({i}/{len(tickers_to_use)}) OK: {t}  Sharpe={sr:.3f}" if sr is not None else f"[LB {lb}] ({i}/{len(tickers_to_use)}) OK: {t}")
            else:
                failures[t] = "optimizer returned empty/none for this ticker"
                print(f"[LB {lb}] ({i}/{len(tickers_to_use)}) SKIP: {t}  (no result)")

        except Exception as e:
            # Typical causes: horizons not a list, NaNs everywhere, index not datetime, etc.
            failures[t] = f"{type(e).__name__}: {e}"
            print(f"[LB {lb}] ({i}/{len(tickers_to_use)}) ERROR: {t}  -> {e}")

    # ---------- 3) Save successes and log failures ----------
    horizons_tag = "_".join(str(h) for h in breakout_horizons)
    out_pkl = f"optimized_breakout_params_{horizons_tag}_vol{pct_vol_target}_{lb}.pkl"
    try:
        with open(out_pkl, "wb") as f:
            pickle.dump(results, f)
        print(f"[LB {lb}] Saved {len(results)} tickers to {out_pkl}")
    except Exception as e:
        print(f"[LB {lb}] Error saving pickle: {e}")

    if failures:
        err_json = f"failed_tickers_{horizons_tag}_vol{pct_vol_target}_{lb}.json"
        try:
            with open(err_json, "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=2, ensure_ascii=False)
            print(f"[LB {lb}] Logged {len(failures)} failures to {err_json}")
        except Exception as e:
            print(f"[LB {lb}] Error writing failure log: {e}")

    return lb, results



# --- Carver breakout scalar defaults (book-style table) ---
CARVER_BREAKOUT_SCALARS_DEFAULT: Dict[int, float] = {
    10: 0.60, 20: 0.67, 40: 0.70, 80: 0.73, 160: 0.74, 320: 0.74,
}

def optimize_for_lookback_carver(
    lb: int,
    tickers: List[str],
    breakout_horizons: List[int],
    tickers_price_data: pd.DataFrame,   # wide price frame: one column per ticker
    pct_vol_target: float,
    n_trials_per_ticker: int,
    *,
    # Optuna / logging knobs
    silent: bool = True,
    n_jobs: int | None = None,
    show_progress_bar: bool = False,
    # >>> NEW: Carver-scalar controls <<<
    use_carver_scalars: bool = False,
    carver_breakout_scalars: Dict[int, float] | None = None,
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    """
    Safe per-ticker optimization for a single vol lookback (lb).
    If `use_carver_scalars=True`, pass a fixed scalar per horizon (Carver table)
    via `carver_breakout_scalars` (defaults provided). Skips bad tickers and
    continues. Returns (lb, {ticker: result,...}).
    """

    # ---------- 0) Strong input coercion / sanity ----------
    try:
        lb = int(lb)
    except Exception:
        raise ValueError(f"vol_lookback_for_position_sizing (lb) must be int, got {type(lb)}")

    if isinstance(breakout_horizons, (int, np.integer)):
        breakout_horizons = [int(breakout_horizons)]
    else:
        breakout_horizons = [int(h) for h in list(breakout_horizons)]

    if not isinstance(tickers_price_data, pd.DataFrame):
        if isinstance(tickers_price_data, pd.Series):
            tickers_price_data = tickers_price_data.to_frame()
        else:
            raise TypeError("tickers_price_data must be a pandas DataFrame (wide, one column per ticker).")

    tickers_price_data = tickers_price_data.apply(pd.to_numeric, errors="coerce")

    # Carver scalars setup
    scalar_mode = "carver" if use_carver_scalars else "dynamic"
    scalar_table = (carver_breakout_scalars or CARVER_BREAKOUT_SCALARS_DEFAULT) if use_carver_scalars else None
    if use_carver_scalars:
        missing = [h for h in breakout_horizons if h not in scalar_table]
        if missing:
            print(f"[LB {lb}] WARNING: No scalar provided for horizons {missing}. Using 1.0 for those.")
            scalar_table = scalar_table.copy()
            for h in missing:
                scalar_table[h] = 1.0

    # ---------- 1) Pre-filter tickers ----------
    print(f"[LB {lb}] Starting breakout optimization for {len(tickers)} tickers; "
          f"horizons={breakout_horizons}; scalars={scalar_mode}")

    cols = set(map(str, tickers_price_data.columns))
    tickers = [str(t) for t in tickers]

    present = [t for t in tickers if t in cols]
    missing_cols = [t for t in tickers if t not in cols]
    if missing_cols:
        print(f"[LB {lb}] Skipping {len(missing_cols)} missing tickers (not in price_frame), e.g. {missing_cols[:8]}")

    min_needed = max(max(breakout_horizons), lb) + 25

    ok_len, too_short = [], []
    for t in present:
        n = tickers_price_data[t].dropna().shape[0]
        if n >= min_needed:
            ok_len.append(t)
        else:
            too_short.append((t, n))

    if too_short:
        print(f"[LB {lb}] Skipping {len(too_short)} short series (<{min_needed} bars), e.g. {too_short[:8]}")

    tickers_to_use = ok_len
    if not tickers_to_use:
        print(f"[LB {lb}] No tickers left after pre-filtering.")
        return lb, {}

    # ---------- 2) Optimize per ticker ----------
    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}

    for i, t in enumerate(tickers_to_use, 1):
        try:
            px_one = tickers_price_data.loc[:, [t]].copy()

            outer_workers = min(6, (os.cpu_count() or 1))
            optuna_jobs   = max(1, (os.cpu_count() or 1) // outer_workers)

            fn = co.optimize_breakout_weights_carver
            fn_params = getattr(fn, "__code__", None)
            fn_varnames = set(fn_params.co_varnames) if fn_params else set()

            kwargs = dict(
                price_frame=px_one,
                tickers=[t],
                breakout_horizons=breakout_horizons,
                vol_lookback_for_position_sizing=lb,
                pct_vol_target=float(pct_vol_target),
                n_trials_per_ticker=int(n_trials_per_ticker),
            )

            if "n_jobs" in fn_varnames and n_jobs is not None:
                kwargs["n_jobs"] = n_jobs
            elif "n_jobs" in fn_varnames:
                kwargs["n_jobs"] = optuna_jobs

            if "silent" in fn_varnames:
                kwargs["silent"] = bool(silent)
            if "show_progress_bar" in fn_varnames:
                kwargs["show_progress_bar"] = bool(show_progress_bar)

            # >>> pass Carver/dynamic scaling intent to the optimizer, if supported
            if use_carver_scalars:
                if "use_carver_scalars" in fn_varnames:
                    kwargs["use_carver_scalars"] = True
                if "carver_breakout_scalars" in fn_varnames:
                    kwargs["carver_breakout_scalars"] = scalar_table
                if "forecast_scalars" in fn_varnames and "carver_breakout_scalars" not in kwargs:
                    kwargs["forecast_scalars"] = scalar_table
                for k in ("use_dynamic_scaler", "dynamic_scaling", "scale_to_abs10"):
                    if k in fn_varnames:
                        kwargs[k] = False  # avoid double-scaling
            else:
                for k in ("use_dynamic_scaler", "dynamic_scaling", "scale_to_abs10"):
                    if k in fn_varnames:
                        kwargs[k] = True

            out = fn(**kwargs)

            if out and t in out:
                out[t].setdefault("scalar_mode", scalar_mode)
                if use_carver_scalars:
                    out[t].setdefault("scalars_used",
                                      {int(k): float(v) for k, v in (scalar_table or {}).items()})
                results[t] = out[t]
                sr = out[t].get("sharpe")
                print(f"[LB {lb}] ({i}/{len(tickers_to_use)}) OK: {t}  Sharpe={sr:.3f}"
                      if sr is not None else f"[LB {lb}] ({i}/{len(tickers_to_use)}) OK: {t}")
            else:
                failures[t] = "optimizer returned empty/none for this ticker"
                print(f"[LB {lb}] ({i}/{len(tickers_to_use)}) SKIP: {t}  (no result)")

        except Exception as e:
            failures[t] = f"{type(e).__name__}: {e}"
            print(f"[LB {lb}] ({i}/{len(tickers_to_use)}) ERROR: {t}  -> {e}")

    # ---------- 3) Save ----------
    horizons_tag = "_".join(str(h) for h in breakout_horizons)
    mode_tag     = scalar_mode  # 'carver' or 'dynamic'
    out_pkl = f"optimized_breakout_params_{horizons_tag}_vol{pct_vol_target}_{lb}_{mode_tag}.pkl"
    try:
        with open(out_pkl, "wb") as f:
            pickle.dump(results, f)
        print(f"[LB {lb}] Saved {len(results)} tickers to {out_pkl}")
    except Exception as e:
        print(f"[LB {lb}] Error saving pickle: {e}")

    if failures:
        err_json = f"failed_tickers_{horizons_tag}_vol{pct_vol_target}_{lb}_{mode_tag}.json"
        try:
            with open(err_json, "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=2, ensure_ascii=False)
            print(f"[LB {lb}] Logged {len(failures)} failures to {err_json}")
        except Exception as e:
            print(f"[LB {lb}] Error writing failure log: {e}")

    return lb, results

# Running the process
def load_data():
    _client = get_binance_client()

    tickers = cta.get_all_usdt_perpetual_futures_tickers(api_key=settings.binance_api_key, api_secret=settings.binance_api_secret)
    if not settings.breakout_price_data_path or not settings.breakout_price_data_path.exists():
        raise FileNotFoundError(
            "Breakout price data not found. Set MOMO_BREAKOUT_PRICE_DATA_PATH (see .env.example)."
        )
    # 1d data seems better for breakout
    tickers_price_data = pd.read_pickle(settings.breakout_price_data_path).drop_duplicates()
    successfully_fetched_tickers_list = tickers_price_data.columns.tolist()
    data_frequency = '4h'
    optimized_parameters = pd.read_pickle(settings.momentum_params_path)
    ticker_symbol_map = {ticker: ticker for ticker in optimized_parameters.keys()}
    return tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map


def main():
    adj = False
    factor = 6


    breakout_horizons = [10, 20, 40, 80, 160] # , 320]
    print(f'BREAKOUT HORIZONS USED: {breakout_horizons}')
    # Adjustment if needed
    if adj:
        breakout_horizons = [x*factor for x in breakout_horizons]

    tickers, tickers_price_data, successfully_fetched_tickers_list, data_frequency, optimized_parameters, ticker_symbol_map = load_data()
    lookbacks = [30, 60, 90,  120, 180, 200]
    # Build argument list
    args_list = [
        (lb, tickers, breakout_horizons, tickers_price_data, 0.80, 500)
        for lb in lookbacks
    ]

    # # Use an explicit spawn context on Windows
    # with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
    #     futures = [executor.submit(optimize_for_lookback, *args) for args in args_list]
    #     for fut in as_completed(futures):
    #         lb, res = fut.result()  # <-- we now return (lb, dict)
    #         if res:
    #             try:
    #                 best_list = [res[t]["sharpe"] for t in res]
    #             except Exception:
    #                 best_list = []
    #             print(f"Lookback {lb}: best sharpe(s) {best_list}")
    #         else:

    # Spawn-safe pool with a progress bar for the overall sweep
    # with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
    #     futures = [executor.submit(optimize_for_lookback, *args) for args in args_list]
    #     with tqdm(total=len(futures), desc="Lookback jobs", unit="job") as pbar:
    #         for fut in as_completed(futures):
    #             lb, res = fut.result()
    #             if res:
    #                 try:
    #                     best_list = [res[t]["sharpe"] for t in res]
    #                     print(f"Lookback {lb}: best sharpe(s) {best_list}")
    #                 except Exception:
    #                     print(f"Lookback {lb}: results saved (could not print sharpes).")
    #             else:
    #                 print(f"Lookback {lb}: no results")
    #             pbar.update(1)


    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
        futures = [
            executor.submit(
                optimize_for_lookback_carver,
                *args,
                use_carver_scalars=True,                 # <— choose mode here
                carver_breakout_scalars=None,            # or pass your custom dict
                silent=True,
                show_progress_bar=False
            )
            for args in args_list
        ]
        with tqdm(total=len(futures), desc="Lookback jobs", unit="job") as pbar:
            for fut in as_completed(futures):
                lb_done, res = fut.result()
                if res:
                    try:
                        best_list = [res[t]["sharpe"] for t in res]
                        print(f"Lookback {lb_done}: best sharpe(s) {best_list}")
                    except Exception:
                        print(f"Lookback {lb_done}: results saved (could not print sharpes).")
                else:
                    print(f"Lookback {lb_done}: no results")
                pbar.update(1)



if __name__ == "__main__":
    mp.freeze_support()  # important for Windows
    main()
