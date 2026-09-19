import os
import time

import numpy as np
import pandas as pd

import build_results_dashboard as dash_app


def _sample_bundle():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    static_a = pd.DataFrame(
        {
            "price": np.linspace(100, 110, 6),
            "returns": [0.0, 0.01, -0.02, 0.03, 0.01, -0.01],
            "equity": (1 + pd.Series([0.0, 0.01, -0.02, 0.03, 0.01, -0.01], index=idx)).cumprod(),
            "positions_usd": [0, 1000, 1500, 1000, 500, 0],
            "fees": [0, 1, 1, 1, 1, 1],
            "total_cost": [0, 2, 2, 2, 2, 2],
            "turnover_notional": [0, 100, 200, 100, 50, 0],
        },
        index=idx,
    )
    static_b = static_a.copy()
    static_b["returns"] = [-0.01, 0.02, 0.0, 0.01, -0.01, 0.01]
    static_b["equity"] = (1 + static_b["returns"]).cumprod()
    wfo_a = static_a.iloc[2:].copy()
    wfo_b = static_b.iloc[2:].copy()
    return {
        "meta": {"portfolio_value": 100_000.0, "timestamp": "test"},
        "breakout": {
            "first_oos_ts": idx[2],
            "data_frequency": "1d",
            "results": {
                "AAAUSDT": {
                    "params": {"status": "success"},
                    "metrics": {"sharpe": 1.0, "total_return": 0.05, "max_drawdown": -0.02},
                    "metrics_oos": {"sharpe": 0.5, "total_return": 0.02, "max_drawdown": -0.03},
                    "series": static_a,
                    "walkforward": {
                        "status": "success",
                        "n_windows": 2,
                        "first_oos_ts": idx[2],
                        "last_oos_ts": idx[-1],
                        "metrics_oos": {
                            "sharpe": 1.2,
                            "total_return": 0.04,
                            "max_drawdown": -0.02,
                            "cagr": 0.2,
                            "deflated_sharpe": 0.8,
                            "t_stat": 1.5,
                            "skew": 0.1,
                            "lower_tail": -0.01,
                            "upper_tail": 0.03,
                        },
                        "series": wfo_a,
                        "weights": pd.DataFrame({"10": [0.6, 0.4], "20": [0.4, 0.6]}, index=[idx[2], idx[4]]),
                    },
                },
                "BBBUSDT": {
                    "params": {"status": "success"},
                    "metrics": {"sharpe": 0.8, "total_return": 0.03, "max_drawdown": -0.01},
                    "metrics_oos": {"sharpe": 0.2, "total_return": 0.01, "max_drawdown": -0.02},
                    "series": static_b,
                    "walkforward": {
                        "status": "success",
                        "n_windows": 1,
                        "first_oos_ts": idx[2],
                        "last_oos_ts": idx[-1],
                        "metrics_oos": {"sharpe": -0.1, "total_return": -0.01, "max_drawdown": -0.03},
                        "series": wfo_b,
                        "weights": pd.DataFrame({"10": [1.0], "20": [0.0]}, index=[idx[2]]),
                    },
                },
                "NEWUSDT": {
                    "params": {"status": "skipped_insufficient_data"},
                    "metrics": {},
                    "metrics_oos": None,
                    "series": static_a.iloc[:1],
                    "walkforward": {"status": "skipped_insufficient_data", "n_windows": 0},
                },
            },
        },
    }


def test_find_latest_bundle_prefers_wfo_dir(tmp_path):
    data_dir = tmp_path / "data_store"
    wfo_dir = data_dir / "wfo_runs"
    wfo_dir.mkdir(parents=True)
    root_bundle = data_dir / "backtest_results_bundle_20990101_000000.pkl"
    wfo_bundle = wfo_dir / "backtest_results_bundle_20240101_000000.pkl"
    root_bundle.write_bytes(b"root")
    time.sleep(0.01)
    wfo_bundle.write_bytes(b"wfo")
    os.utime(root_bundle, (time.time() + 100, time.time() + 100))

    assert dash_app._find_latest_bundle(data_dir) == wfo_bundle


def test_universe_summary_includes_success_and_skipped_rows():
    summary = dash_app.build_universe_summary(_sample_bundle(), "breakout")

    assert len(summary) == 3
    assert set(summary["wfo_status"]) == {"success", "skipped_insufficient_data"}
    aaa = summary.set_index("ticker").loc["AAAUSDT"]
    assert aaa["wfo_sharpe"] == 1.2
    assert aaa["static_oos_sharpe"] == 0.5
    assert aaa["quality_flag"] == "low sample"


def test_universe_summary_coverage_uses_full_price_series_not_wfo_tail():
    bundle = _sample_bundle()
    idx = bundle["breakout"]["results"]["AAAUSDT"]["series"].index
    bundle["breakout"]["results"]["AAAUSDT"]["walkforward"]["series"] = bundle["breakout"]["results"]["AAAUSDT"]["series"].iloc[2:4]
    bundle["breakout"]["results"]["AAAUSDT"]["walkforward"]["last_oos_ts"] = idx[3]

    summary = dash_app.build_universe_summary(bundle, "breakout").set_index("ticker")

    assert summary.loc["AAAUSDT", "days_stale"] == 0.0
    assert "stale" not in summary.loc["AAAUSDT", "quality_flag"]


def test_extract_metric_prefers_wfo_for_wfo_source():
    payload = _sample_bundle()["breakout"]["results"]["AAAUSDT"]

    assert dash_app.extract_metric(payload, dash_app.SOURCE_WFO, "sharpe") == 1.2
    assert dash_app.extract_metric(payload, dash_app.SOURCE_STATIC_OOS, "sharpe") == 0.5
    assert dash_app.extract_metric(payload, dash_app.SOURCE_FULL, "sharpe") == 1.0


def test_cost_drag_uses_total_cost_over_portfolio_value():
    series = _sample_bundle()["breakout"]["results"]["AAAUSDT"]["walkforward"]["series"]

    assert dash_app.compute_cost_drag(series, 100_000.0) == series["total_cost"].sum() / 100_000.0


def test_equal_weight_basket_aligns_dates_and_ignores_missing():
    bundle = _sample_bundle()
    results = bundle["breakout"]["results"]

    basket = dash_app.build_equal_weight_portfolio(
        ["AAAUSDT", "BBBUSDT", "MISSINGUSDT"],
        dash_app.SOURCE_WFO,
        results,
        first_oos_ts=bundle["breakout"]["first_oos_ts"],
        portfolio_value=100_000.0,
        freq="1d",
    )

    assert len(basket["returns"]) == 4
    assert set(basket["contributions"].index) == {"AAAUSDT", "BBBUSDT"}
    assert pd.notna(basket["metrics"]["total_return"])


def test_drawdown_handles_empty_flat_returns_and_equity():
    assert dash_app.compute_drawdown(pd.Series(dtype=float)).empty
    flat = dash_app.compute_drawdown(pd.Series([0.0, 0.0, 0.0]))
    equity = dash_app.compute_drawdown(pd.Series([1.0, 1.2, 0.9]))

    assert flat.tolist() == [0.0, 0.0, 0.0]
    assert equity.iloc[-1] == 0.9 / 1.2 - 1.0


def test_drawdown_handles_equity_below_zero():
    equity = dash_app.compute_drawdown(pd.Series([1.0, 0.5, -0.2]))

    assert equity.iloc[-1] == -1.2


def test_calendarize_series_inserts_missing_daily_dates():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10"])
    series = pd.Series([1.0, 2.0, 3.0], index=idx)

    out = dash_app._calendarize_series(series, "1d")

    assert len(out) == 10
    assert out.isna().sum() == 7
    assert pd.isna(out.loc["2024-01-03"])


def test_detail_figure_does_not_connect_line_gaps():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10"])
    series = pd.DataFrame(
        {
            "price": [1.0, 1.1, 1.2],
            "signal": [0.0, 1.0, -1.0],
            "positions_usd": [0.0, 100.0, -100.0],
            "returns": [0.0, 0.01, -0.01],
        },
        index=idx,
    )

    fig = dash_app._build_detail_figure(series, "AAAUSDT", "breakout", "1d")

    line_traces = [trace for trace in fig.data if getattr(trace, "type", "") == "scatter"]
    assert line_traces
    assert all(trace.connectgaps is False for trace in line_traces)
    assert any(pd.isna(value) for value in line_traces[0].y)
