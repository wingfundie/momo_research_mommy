"""Causality and outcome checks for the standalone event study."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.momentum_turning_study import (
    ROOT, events_for, feature_frame, make_events, signal,
)
from momo_bot.strategies.forecasts import get_signal, calc_ewma_forecast
from momo_bot.strategies.scalars import CARVER_EWMAC_FORECAST_SCALARS


def test_signal_matches_production_formula_after_warmup():
    close = pd.Series(100 + np.arange(600)*.1 + np.sin(np.arange(600)/10))
    rules = list(CARVER_EWMAC_FORECAST_SCALARS)[:5]
    actual, _ = signal(close, [.2]*5, rules, CARVER_EWMAC_FORECAST_SCALARS, 1.12, True)
    components = [get_signal(close, calc_ewma_forecast(close, *pair), None,
                             CARVER_EWMAC_FORECAST_SCALARS[pair]) for pair in rules]
    expected = (pd.concat(components, axis=1).to_numpy() @ np.array([.2]*5)*1.12).clip(-20,20)
    np.testing.assert_allclose(actual.iloc[365:],expected[365:])
    assert actual.iloc[:365].isna().all()


def test_zero_hysteresis_does_not_retrigger_inside_band():
    s = pd.Series(np.r_[np.full(365,np.nan),[-3,-1,1,3,1,-1,-3]])
    comp = pd.concat([s]*5,axis=1)
    assert events_for(s,comp,'Zero 2') == [(368,1),(371,-1)]


def test_next_open_return_and_ambiguous_intrabar_order():
    ix = pd.date_range('2024-01-01',periods=410,tz='UTC')
    f = pd.DataFrame(dict(open=100.,high=102.,low=98.,close=100.,volume=10.,
                          quote_volume=1000.,taker_quote=500.),index=ix)
    # Close-time signal at index 370; gap to next open makes same-close entry incorrect.
    f.loc[ix[371],['open','high','low','close']] = [110.,120.,90.,115.]
    f=feature_frame(f,f)
    s=pd.Series(np.r_[np.full(365,np.nan),np.full(5,-3),np.full(40,3)],index=ix)
    rows=make_events('TEST','Test',f,s,pd.concat([s]*5,axis=1),'Zero 2')
    one=next(r for r in rows if r['horizon']==1)
    assert one['entry_time']==ix[371]
    assert np.isclose(one['gross'],115/110-1)
    assert one['barrier']=='ambiguous'
    assert np.isclose(one['net'],115/110-1-.0012)
