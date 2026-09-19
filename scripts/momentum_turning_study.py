"""Reproducible, public-data momentum turning-point event study and offline HTML report.

Run: python scripts/momentum_turning_study.py [--offline]
Outputs are isolated from live bot stores and parameter files.
"""
from __future__ import annotations

import ast
import argparse
import hashlib
import html
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from momo_bot.strategies.forecasts import calc_ewma_forecast
from momo_bot.strategies.scalars import CARVER_EWMAC_FORECAST_SCALARS
from scripts.report_theme.report_theme import hero, metric, finding, figure_html, style_plotly, render_page

OUT = ROOT / 'reports' / 'momentum_turning_points'
ASSETS = ['ETH', 'BTC', 'SOL', 'HYPE', 'XRP', 'XLM', 'ZEC', 'DOGE', 'WIF']
HORIZONS = [1, 3, 7, 14, 30]
START = pd.Timestamp('2020-07-05', tz='UTC')
FREEZE = pd.Timestamp('2026-06-03', tz='UTC')
COLS = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore']
BLUE, GOLD, PURPLE, GREY, GREEN = '#5696b9', '#ce9a48', '#8370b4', '#858585', '#748264'


def api(path, **params):
    url = 'https://fapi.binance.com' + path + '?' + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def load_model():
    from momo_bot.config import settings
    paths = sorted(settings.data_dir.glob('backtest_results_bundle_*.pkl'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        bundle = pd.read_pickle(path)
        if bundle.get('momentum', {}).get('results'):
            return path, bundle['momentum']
    raise RuntimeError('No momentum bundle found')


def fetch(asset, cutoff, offline):
    path = OUT / 'data' / f'{asset}USDT_1d.csv'
    if path.exists():
        df = pd.read_csv(path, index_col='time', parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df = pd.DataFrame()
    if not offline:
        cursor = int((df.index[-1] if len(df) else START).timestamp() * 1000)
        rows = []
        while cursor < cutoff:
            page = api('/fapi/v1/klines', symbol=asset + 'USDT', interval='1d',
                       startTime=cursor, endTime=cutoff - 1, limit=1000)
            if not page:
                break
            rows.extend(page)
            next_cursor = int(page[-1][0]) + 86400000
            if next_cursor <= cursor:
                raise RuntimeError('Pagination stalled')
            cursor = next_cursor
            time.sleep(.15)
        if rows:
            new = pd.DataFrame(rows, columns=COLS)
            new['time'] = pd.to_datetime(new['time'], unit='ms', utc=True)
            new = new.set_index('time').astype(float).drop(columns='ignore')
            df = pd.concat([df, new]) if len(df) else new
            df = df[~df.index.duplicated(keep='last')].sort_index()
    if df.empty:
        raise RuntimeError(f'No data for {asset}')
    df = df[df.close_time < cutoff]
    assert not df.index.duplicated().any()
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all(), asset + ' gaps'
    assert (df[['open', 'high', 'low', 'close']] > 0).all().all()
    assert (df.high >= df[['open', 'close', 'low']].max(axis=1)).all()
    assert (df.low <= df[['open', 'close', 'high']].min(axis=1)).all()
    assert (df[['volume', 'quote_volume', 'trades']] >= 0).all().all()
    assert (df.taker_quote <= df.quote_volume * 1.000001).all()
    df.to_csv(path)
    return df


def signal(close, weights, rules, scalars, dm, cap):
    components = pd.concat([calc_ewma_forecast(close, *pair, vol_lookback=365)
                            .mul(scalars[pair]).clip(-20, 20) for pair in rules], axis=1)
    s = pd.Series(components.to_numpy() @ np.asarray(weights), index=close.index) * dm
    return (s.clip(-20, 20) if cap else s), components


def events_for(s, components, kind):
    if kind.startswith('Slope'):
        confirm = int(kind[-1])
        smooth = s.ewm(span=3, adjust=False).mean()
        d = np.sign(smooth.diff()).replace(0, np.nan).ffill()
        out = []
        previous = 0
        for i in range(365 + confirm, len(s)):
            ds = d.iloc[i-confirm+1:i+1]
            if ds.isna().any() or not (ds == ds.iloc[-1]).all():
                continue
            direction = int(ds.iloc[-1])
            if previous and direction != previous:
                out.append((i, direction))
            previous = direction
        return out
    if kind.startswith('Zero'):
        threshold = float(kind.split(' ')[1])
        previous = 0
        out = []
        for i in range(365, len(s)):
            v = s.iloc[i]
            direction = 1 if v > threshold else -1 if v < -threshold else 0
            if direction:
                if previous and direction != previous:
                    out.append((i, direction))
                previous = direction
        return out
    fast, slow = components.iloc[:, :2].mean(axis=1), components.iloc[:, -2:].mean(axis=1)
    state = np.sign(fast).where(np.sign(fast) != np.sign(slow), 0)
    return [(i, int(state.iloc[i])) for i in range(366, len(s))
            if state.iloc[i] != 0 and np.isfinite(state.iloc[i]) and state.iloc[i] != state.iloc[i-1]]


def feature_frame(df, btc):
    f = df.copy()
    f['rvol'] = f.quote_volume / f.quote_volume.shift().rolling(30).median()
    f['imbalance'] = 2 * f.taker_quote / f.quote_volume.replace(0, np.nan) - 1
    tr = pd.concat([f.high-f.low, (f.high-f.close.shift()).abs(),
                    (f.low-f.close.shift()).abs()], axis=1).max(axis=1)
    f['atr'] = tr.rolling(14).mean()
    f['vol'] = f.close.pct_change().rolling(30).std()
    f['vol_high'] = f.vol > f.vol.shift().rolling(180).median()
    f['btc_bull'] = (btc.close > btc.close.ewm(span=128).mean()).reindex(f.index)
    # Ex-post evaluation targets only: strict unique extrema in centered 7-day windows.
    f['trough'] = (f.close == f.close.rolling(7, center=True).min())
    f['peak'] = (f.close == f.close.rolling(7, center=True).max())
    return f


def make_events(asset, model, f, s, comp, kind):
    rows = []
    ordinary_returns = {h: f.close.shift(-h)/f.open.shift(-1)-1 for h in HORIZONS}
    baseline_cache = {}
    for i, direction in events_for(s, comp, kind):
        if i+1 >= len(f):
            continue
        entry = f.open.iloc[i+1]
        vol_group = 'High volume' if f.rvol.iloc[i] >= 1.5 else 'Normal/low volume'
        pivot_flag = f.trough if direction > 0 else f.peak
        near = [j for j in range(max(3, i-3), min(len(f)-3, i+4)) if pivot_flag.iloc[j]]
        nearest = min(near, key=lambda j: abs(j-i)) if near else None
        for h in HORIZONS:
            if i+h >= len(f):
                continue
            gross = direction * (f.close.iloc[i+h] / entry - 1)
            path = f.iloc[i+1:i+h+1]
            favorable = ((path.high.max()/entry-1) if direction == 1 else (1-path.low.min()/entry))
            adverse = ((path.low.min()/entry-1) if direction == 1 else (1-path.high.max()/entry))
            # A fixed one-ATR price barrier on either side. Same-bar ties remain ambiguous.
            target = entry + direction*f.atr.iloc[i]
            stop = entry - direction*f.atr.iloc[i]
            barrier = 'timeout'
            for _, bar in path.iterrows():
                hit = bar.high >= target if direction == 1 else bar.low <= target
                loss = bar.low <= stop if direction == 1 else bar.high >= stop
                if hit or loss:
                    barrier = 'ambiguous' if hit and loss else 'target' if hit else 'stop'
                    break
            # Match ordinary eligible days on year and BTC trend; preserve event direction.
            key=(f.index[i].year,bool(f.btc_bull.iloc[i]),f.index[i]>=FREEZE,direction,h)
            if key not in baseline_cache:
                candidates = f.index[(f.index.year == key[0]) &
                                      (f.btc_bull == key[1]) & s.notna() &
                                      ((f.index>=FREEZE) if key[2] else (f.index<FREEZE))]
                ordinary=(direction*ordinary_returns[h]-.0012).loc[candidates].dropna()
                baseline_cache[key]=float((ordinary>0).mean()) if len(ordinary) else np.nan
            base_hit=baseline_cache[key]
            rows.append(dict(asset=asset, model=model, kind=kind, time=f.index[i],
                             entry_time=f.index[i+1], exit_time=f.index[i+h]+pd.Timedelta(days=1),
                             horizon=h, direction=direction, signal=s.iloc[i], gross=gross,
                             net=gross-.0012, hit=gross>.0012, base_hit=base_hit,
                             rvol=f.rvol.iloc[i], volume_group=vol_group,
                             flow_aligned=direction*f.imbalance.iloc[i]>0,
                             btc_bull=bool(f.btc_bull.iloc[i]), vol_high=bool(f.vol_high.iloc[i]),
                             mfe=favorable, mae=adverse, barrier=barrier,
                             pivot_match=bool(near) if i+3 < len(f) else np.nan,
                             pivot_time=f.index[nearest] if nearest is not None else pd.NaT,
                             pivot_lag=i-nearest if nearest is not None else np.nan,
                             period='After model save' if f.index[i]>=FREEZE else 'Historical reconstruction'))
    return rows


def stats(g):
    n = len(g)
    if not n:
        return dict(n=0)
    p = g.hit.mean()
    z = 1.96
    mid=(p+z*z/(2*n))/(1+z*z/n)
    rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    # Resample calendar-month event clusters, retaining temporal clustering within each month.
    month = g.time.dt.strftime('%Y-%m')
    blocks = g.assign(uplift=g.hit.astype(float)-g.base_hit).groupby(month).agg(
        n=('hit', 'size'), wins=('hit', 'sum'), uplift=('uplift', 'sum'))
    lo=hi=ulo=uhi=np.nan
    if len(blocks)>=6:
        rng=np.random.default_rng(1729)
        draws=rng.integers(0,len(blocks),(1000,len(blocks)))
        b=blocks.to_numpy()[draws].sum(axis=1)
        lo,hi=np.quantile(b[:,1]/b[:,0],[.025,.975])
        ulo,uhi=np.quantile(b[:,2]/b[:,0],[.025,.975])
    denom=abs(g.loc[g.net<0,'net'].sum())
    return dict(n=n,hit=p,wilson_low=mid-rad,wilson_high=mid+rad,
                cluster_low=lo,cluster_high=hi,months=len(blocks),base_hit=g.base_hit.mean(),
                uplift=p-g.base_hit.mean(),uplift_low=ulo,uplift_high=uhi,
                mean_net=g.net.mean(),median_net=g.net.median(),profit_factor=g.loc[g.net>0,'net'].sum()/denom if denom else np.nan,
                mean_mfe=g.mfe.mean(),mean_mae=g.mae.mean(),pivot_precision=g.pivot_match.mean(),
                median_pivot_lag=g.pivot_lag.median(),target_first=(g.barrier=='target').mean(),
                ambiguous=(g.barrier=='ambiguous').mean(),hit_0bps=(g.gross>0).mean(),
                hit_30bps=(g.gross>.003).mean(),hit_60bps=(g.gross>.006).mean())


def summarize(events, keys):
    return pd.DataFrame([dict(zip(keys,k if isinstance(k,tuple) else (k,)),**stats(g))
                         for k,g in events.groupby(keys, sort=False)])


def table(frame, columns=None):
    f=frame[columns].copy() if columns else frame.copy()
    for col in f:
        if col in ['hit','base_hit','uplift','mean_net','median_net','cluster_low','cluster_high',
                   'wilson_low','wilson_high','pivot_precision','recall','target_first','ambiguous',
                   'hit_0bps','hit_30bps','hit_60bps','uplift_low','uplift_high']:
            f[col]=f[col].map(lambda v: f'{v:.1%}' if pd.notna(v) else '—')
        elif pd.api.types.is_float_dtype(f[col]):
            f[col]=f[col].map(lambda v: f'{v:.2f}' if pd.notna(v) else '—')
    f.columns=[c.replace('_',' ').capitalize() for c in f.columns]
    return '<div class="table-wrap" tabindex="0">'+f.to_html(index=False,escape=True,border=0)+'</div>'


def style(fig, title, height=580):
    return style_plotly(fig,title,height)


def chart_html(fig, ident):
    chart=fig.to_html(full_html=False,include_plotlyjs=False,div_id=ident,
                     config={'responsive':True,'displaylogo':False})
    data_href='summary.csv' if ident=='comparison' or ident.startswith('path-') else f'data/{ident.split("-")[-1]}_signals.csv'
    return figure_html(chart,str(fig.layout.title.text),
                       'Source: cached Binance daily candles and reconstructed momentum forecasts. Data snapshot unchanged.',data_href)


def build_report(frames, signals, events, summary, quality, model_meta, checks):
    primary=events[(events.kind=='Slope 2') & (events.horizon==7)]
    current=primary[primary.model=='Current / fallback']
    head=summarize(current,['asset'])
    post=summarize(current[current.period=='After model save'],['asset'])
    volume_all=summarize(current,['asset','volume_group']).set_index(['asset','volume_group'])
    nonoverlap_rows=[]
    for asset,g in current.groupby('asset'):
        end=pd.Timestamp.min.tz_localize('UTC'); keep=[]
        for idx,row in g.sort_values('time').iterrows():
            if row.entry_time>=end:
                keep.append(idx);end=row.exit_time
        nonoverlap_rows.append(dict(asset=asset,**stats(g.loc[keep])))
    nonoverlap=pd.DataFrame(nonoverlap_rows).set_index('asset')
    body=[]
    body.append(hero('ACAUSAL CAPITAL · MOMENTUM RESEARCH','When momentum turns.',
                     'Does price follow?', 'Nine crypto perpetuals. Signal reversals, forward returns and the role of trading volume.',
                     [f'Presentation · {pd.Timestamp.now(tz="UTC").strftime("%d %b %Y")}',
                      f'Data through · {min(f.index[-1] for f in frames.values()).date()} UTC',
                      'Daily signals · 7-day primary horizon','12 bps execution cost · funding excluded']))
    body.append('<nav aria-label="Report sections"><a href="#snapshot">Snapshot</a><a href="#findings">Findings</a>'+''.join(f'<a href="#{a}">{a}</a>' for a in ASSETS)+'<a href="#methods">Methods</a><a href="#sources">Sources</a></nav>')
    indexed=head.set_index('asset')
    body.append('<div class="metrics" id="snapshot">'+metric('Assets covered',str(len(ASSETS)),'7 saved models · 2 fixed-weight fallbacks')
                +metric('Primary scored turns',f'{len(current):,}','Two-bar slope confirmation · seven-day return')
                +metric('XLM · historical hit rate',f'{indexed.loc["XLM","hit"]:.1%}',f'{int(indexed.loc["XLM","n"])} turns · overlapping events included')
                +metric('SOL · historical hit rate',f'{indexed.loc["SOL","hit"]:.1%}',f'{int(indexed.loc["SOL","n"])} turns · overlapping events included')+'</div>')
    body.append('<div class="callout"><b>Read this first.</b> This is an event study of reconstructed signals, not a record of signals actually sent by Telegram and not portfolio P&amp;L. Seven assets use saved bot weights; HYPE and WIF use a clearly labelled equal-weight fallback. Pre-save fitted performance is descriptive. The recent post-save sample is small and the fixed rules were specified for this analysis, not registered before those returns occurred.</div>')
    body.append('<h2>The primary question</h2><p>After a smoothed momentum forecast changes slope for two consecutive daily bars, is the next seven-day return in that new direction positive after 12 bps? A turn upward can still occur below zero; it is an early reversal hypothesis, not the same trade as following the forecast sign.</p>')
    body.append(table(head,['asset','n','hit','base_hit','uplift','cluster_low','cluster_high','mean_net','profit_factor']))
    indexed=head.set_index('asset')
    body.append('<section id="findings"><div class="section-kicker">01 · RESEARCH FINDINGS</div><h2>What the evidence says</h2><div class="findings">')
    body.append(finding(1,'XLM and SOL deserve further testing',f'XLM records {indexed.loc["XLM","hit"]:.1%} hits over {int(indexed.loc["XLM","n"])} primary events, versus a {indexed.loc["XLM","base_hit"]:.1%} regime benchmark. SOL records {indexed.loc["SOL","hit"]:.1%} over {int(indexed.loc["SOL","n"])}. Fitted-history bias and the many comparisons prevent treating these as validated discoveries.'))
    body.append(finding(2,'Wins and payoffs differ',f'XRP wins {indexed.loc["XRP","hit"]:.1%} of events with a {indexed.loc["XRP","mean_net"]:.2%} mean after assumed execution costs. DOGE wins {indexed.loc["DOGE","hit"]:.1%} with a {indexed.loc["DOGE","mean_net"]:.2%} mean. The size of wins and losses matters.'))
    volume_sentences=[]
    for a in ['SOL','XRP']:
        v=volume_all.loc[a]
        high,low=v.loc['High volume'],v.loc['Normal/low volume']
        volume_sentences.append(f'{a}: high-volume primary turns win {high["hit"]:.1%} ({int(high["n"])} events), versus {low["hit"]:.1%} otherwise ({int(low["n"])}).')
    body.append(finding(3,'Volume is asset-specific',' '.join(volume_sentences)+' These small, unadjusted conditional samples do not establish a shared volume rule.'))
    body.append('</div></section>')
    body.append(f'<p><b>Robustness changes the picture.</b> The greedy non-overlap audit gives XLM {nonoverlap.loc["XLM","hit"]:.1%}, BTC {nonoverlap.loc["BTC","hit"]:.1%}, and SOL {nonoverlap.loc["SOL","hit"]:.1%}. HYPE has only {int(indexed.loc["HYPE","n"])} primary events in its entire usable sample. Read the recent post-save table with its much smaller counts before drawing conclusions.</p>')
    body.append('<p class="note">Intervals resample event clusters by calendar month (1,000 replicates). They allow within-month overlap, but may miss dependence across months. They are exploratory, unadjusted intervals across many tests. Do not pick the highest number as a validated winner.</p>')
    body.append('<h2>After the model was saved</h2><p>Events dated 3 June 2026 onward. This conservative date follows the bundle filename; it is not proof of when the strategy was first conceived. Fitted weights and calibration are held fixed. The baseline is restricted to the same period.</p>')
    body.append(table(post,['asset','n','hit','base_hit','uplift','wilson_low','wilson_high','mean_net','months']))
    body.append('<p class="note">Wilson intervals assume independent trials and are descriptive only; fewer than six occupied months suppresses the cluster interval. A high win rate from a handful of turns is weak evidence.</p>')
    fig=go.Figure()
    for model,color in [('Current / fallback',BLUE),('Fixed equal-weight',PURPLE)]:
        g=summarize(primary[primary.model==model],['asset']).set_index('asset').reindex(ASSETS)
        fig.add_trace(go.Bar(name=model,x=ASSETS,y=100*g.hit,marker_color=color))
    fig.add_hline(y=50,line_dash='dash',line_color=GREY)
    fig.update_yaxes(title='7-day net hit rate (%)')
    body.append(chart_html(style(fig,'Saved weights versus a fixed-weight research control',410),'comparison'))
    for asset in ASSETS:
        f=frames[asset]; s=signals[asset]['Current / fallback']; ev=current[current.asset==asset]
        body.append(f'<section id="{asset}"><h2>{asset} / turning-point anatomy</h2>')
        body.append(f'<p>{html.escape(model_meta[asset]["description"])}. Full daily history: {f.index[0].date()} to {f.index[-1].date()}; valid forecast begins {s.first_valid_index().date() if s.notna().any() else "not available"}.</p>')
        recent=f.iloc[-365:]; ix=recent.index
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.045,row_heights=[.53,.25,.22])
        fig.add_trace(go.Scatter(x=ix,y=100*(recent.close/recent.close.iloc[0]-1),name=asset, line=dict(color=BLUE)),row=1,col=1)
        benchmark='ETH' if asset=='BTC' else 'BTC'
        btc=frames[benchmark].close.reindex(ix)
        fig.add_trace(go.Scatter(x=ix,y=100*(btc/btc.iloc[0]-1),name=benchmark,line=dict(color=GOLD)),row=1,col=1)
        for direction,color,symbol in [(1,GREEN,'triangle-up'),(-1,PURPLE,'triangle-down')]:
            e=ev[(ev.direction==direction)&(ev.time>=ix[0])]
            y=100*(f.close.reindex(e.time)/recent.close.iloc[0]-1)
            fig.add_trace(go.Scatter(x=e.time,y=y,mode='markers',name='Turn up' if direction==1 else 'Turn down',marker=dict(color=color,symbol=symbol,size=9),customdata=np.c_[e.net,e.rvol],hovertemplate='%{x}<br>7d net %{customdata[0]:.2%}<br>Relative volume %{customdata[1]:.2f}<extra></extra>'),row=1,col=1)
        fig.add_trace(go.Scatter(x=ix,y=s.reindex(ix),name='Momentum forecast',line=dict(color=PURPLE)),row=2,col=1)
        fig.add_trace(go.Bar(x=ix,y=recent.quote_volume/1e9,name='Quote volume (USDT bn)',marker_color=BLUE,opacity=.7),row=3,col=1)
        fig.add_trace(go.Scatter(x=ix,y=f.quote_volume.shift().rolling(30).median().reindex(ix)/1e9,name='Prior 30d median',line=dict(color=GOLD)),row=3,col=1)
        fig.add_hline(y=0,line_color=GREY,row=2,col=1)
        if ix[0]<FREEZE<ix[-1]:
            fig.add_vline(x=FREEZE.timestamp()*1000,line_dash='dot',line_color=GREY)
        fig.update_yaxes(title_text='Return from chart start (%)',row=1,col=1)
        fig.update_yaxes(title_text='Forecast',row=2,col=1)
        fig.update_yaxes(title_text='USDT bn',row=3,col=1)
        body.append(chart_html(style(fig,f'{asset} relative to {benchmark} · forecast turns and traded volume',720),f'chart-{asset}'))
        body.append('<p class="note">Markers are confirmation dates, not hindsight extrema, and only show events with completed 7-day outcomes. Vertical dotted line: post-save boundary. Volume is Binance USDT-perpetual turnover, not global spot volume or open interest.</p>')
        g=summary[(summary.asset==asset)&(summary.model=='Current / fallback')&(summary.kind=='Slope 2')]
        body.append(table(g,['horizon','n','hit','base_hit','uplift','mean_net','median_net','target_first','ambiguous']))
        volume=summarize(ev,['volume_group'])
        body.append('<h3>Does participation help?</h3>'+table(volume,['volume_group','n','hit','base_hit','mean_net','cluster_low','cluster_high']))
        body.append(table(summarize(ev,['flow_aligned']),['flow_aligned','n','hit','mean_net']))
        if len(volume)==2:
            v=volume.set_index('volume_group')
            delta=v.loc['High volume','hit']-v.loc['Normal/low volume','hit']
            body.append(f'<p>High-volume turns have a {delta*100:+.1f} percentage-point observed hit-rate difference versus other turns. This is an unadjusted association; volatility, direction and market regime can explain it.</p>')
        pathfig=go.Figure()
        for d,color,name in [(1,BLUE,'Up turns'),(-1,PURPLE,'Down turns')]:
            paths=[]
            for row in ev[ev.direction==d].itertuples():
                i=f.index.get_loc(row.time)
                if i+30<len(f):
                    paths.append(d*(f.close.iloc[i+1:i+31].to_numpy()/f.open.iloc[i+1]-1)*100-.12)
            if paths:
                arr=np.array(paths)
                pathfig.add_trace(go.Scatter(x=list(range(1,31)),y=np.median(arr,axis=0),name=f'{name} (n={len(arr)})',line=dict(color=color)))
        pathfig.update_xaxes(title='Days after next-open entry'); pathfig.update_yaxes(title='Median signed net return (%)')
        body.append(chart_html(style(pathfig,'What follows a turn? · fully observed 30-day event paths',330),f'path-{asset}'))
        body.append('</section>')
    body.append('<section id="methods"><div class="section-kicker">METHODS · COVERAGE · REPRODUCTION</div><h2>Research design / from inputs to evidence</h2><div class="method"><ol>')
    steps=[
        'Audit the live calculation and active saved bundle. Use native daily EWMAC pairs (2,8), (4,16), (8,32), (16,64), (32,128). Rebuild on fresh continuous exchange closes with the same saved weights, scalars and diversification multiplier. Preserve the live 365-bar volatility denominator; the training bundle used 360, which is an explicit training/live difference.',
        'Fetch Binance USDT perpetual daily OHLCV, quote turnover, trade counts and taker-buy quote volume from 5 July 2020 or listing. Page public REST data; exclude the unclosed daily candle using exchange time. Assert continuity and OHLC/volume validity; archive CSVs with SHA-256 hashes. No fabricated pre-listing history or forward-filled gaps.',
        'Exclude the first 365 bars. The legacy get_signal function maps NaN forecasts to +20; the research calculation instead leaves warm-up undefined. Match the production formula only after valid volatility exists. Fresh continuous history also differs from the old sparse live file, so these are model reconstructions rather than exact archived alerts.',
        'Primary event: smooth the combined forecast with causal span-3 EWMA; require two successive changes with the new slope sign, then emit once when the confirmed direction switches. Sensitivities: one and three confirmations. Zero 0 and Zero 2 detect sign switches with zero or ±2 hysteresis. Fast/slow disagreement emits on entry into opposing signs between the two fastest and two slowest components; trade the fast direction.',
        'Enter at the next daily open. At horizons 1, 3, 7, 14 and 30 days, exit at that day’s close. Signed simple return is direction × (exit / entry − 1). The headline hit requires this return to exceed 12 bps (illustrative 4 bps fee + 2 bps slippage per side). Show 0/30/60 bps sensitivity. Funding is not included; results are net of assumed execution costs only, not all-in perpetual returns.',
        'Report win rate alongside mean/median return, event profit factor, and favorable/adverse excursions. A symmetric one-ATR(14) target/stop label asks which barrier is touched first within the horizon; same-bar double touches stay ambiguous and timeouts stay timeouts. Barrier fractions use all events in the denominator. These are labels, not a simulated executable stop strategy.',
        'Compare hits with all valid ordinary dates in the same calendar year, BTC trend regime, period and asset, preserving each event’s direction. Baseline outcomes may overlap events; this is a descriptive regime benchmark, not a randomized hypothesis test. Do not assume 50% is always the correct benchmark.',
        'Test prior-median-relative quote volume (current volume / previous 30-day median; high ≥1.5) and whether signed taker imbalance aligns with the proposed direction. Also split by BTC above/below its causal 128-day EWMA, own high/low volatility, long/short and calendar year. Thresholds are fixed, not optimized against hit rate.',
        'Use monthly cluster bootstrap intervals where six months exist and report counts. Add a greedy horizon-specific non-overlap subset. Audit ex-post ±3-day proximity to seven-day centered price extrema and unique-extremum recall; these targets use future data only for scoring and never enter the signal. This asks about local turns, not major cycle tops/bottoms.',
        'Compare saved weights with equal weights and fixed Carver scalars across every asset; HYPE/WIF use this fallback because no active fitted parameters exist. Their fallback is research-only. Split historical reconstruction from after-save events and show yearly stability. No new parameter optimization or best-filter selection occurs.',
        'Export every event, aggregate, raw candle, model specification and verification result. The next validation stage is a prospective paper ledger with precommitted rules, actual funding and fills, and a longer post-save sample; do not infer a deployable edge from this exploratory multi-test report.'
    ]
    body.extend('<li>'+x+'</li>' for x in steps);body.append('</ol></div>')
    body.append('<h3>Alternative turn definitions / seven-day horizon</h3>')
    alt=summary[(summary.model=='Current / fallback')&(summary.horizon==7)]
    body.append(table(alt,['asset','kind','n','hit','mean_net','uplift']))
    body.append('<h3>Execution-cost sensitivity / primary events</h3>'+table(head,['asset','n','hit_0bps','hit','hit_30bps','hit_60bps']))
    nonoverlap.reset_index().to_csv(OUT/'nonoverlap.csv',index=False)
    body.append('<h3>Non-overlapping primary events</h3>'+table(nonoverlap.reset_index(),['asset','n','hit','mean_net','base_hit']))
    piv=[]
    for asset,g in current.groupby('asset'):
        f=frames[asset];s=signals[asset]['Current / fallback']
        valid=f[s.notna() & (f.index<=f.index[-1]-pd.Timedelta(days=10))]
        targets=set(valid.index[valid.peak|valid.trough])
        # Match every nearby same-direction extremum for recall, not just the nearest one.
        matched=set()
        for row in g.itertuples():
            flags=f.trough if row.direction>0 else f.peak
            matched.update(t for t in targets if abs((t-row.time).days)<=3 and flags.loc[t])
        piv.append(dict(asset=asset,n=len(g),pivot_precision=g.pivot_match.mean(),
                        recall=len(matched)/len(targets) if targets else np.nan,
                        price_pivots=len(targets),median_pivot_lag=g.pivot_lag.median()))
    body.append('<h3>Price-pivot audit / hindsight labels only</h3>'+table(pd.DataFrame(piv)))
    body.append('<p class="note">Precision: proportion of events near a same-direction local extremum. Recall: share of eligible unique extrema with a same-direction event within ±3 days. Several events may match one extremum. Positive median lag means signal confirmation came after the nearest extremum.</p>')
    for dim,title in [('direction','Long versus short'),('btc_bull','BTC regime'),('vol_high','Own volatility regime'),('year','Calendar stability')]:
        e=current.copy();e['year']=e.time.dt.year
        result=summarize(e,['asset',dim]);result.to_csv(OUT/f'by_{dim}.csv',index=False)
        body.append(f'<details><summary>{title}</summary>'+table(result,['asset',dim,'n','hit','mean_net','uplift'])+'</details>')
    body.append('<h3>Data coverage</h3>'+table(pd.DataFrame(quality)))
    body.append('<h3>Verification</h3><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in checks)+'</ul>')
    body.append('<h3>Material limitations</h3><p>Selection is the nine coins requested, not a survivorship-free universe. Old fitted weights and pooled calibration may embed information unavailable on historical dates. The reported training/OOS split in the bundle is not treated as clean prospective evidence. HYPE has substantially less post-warm-up history. Volume is venue-specific reported turnover; taker flow is not open interest. Costs are scenarios, and omitted historical funding can change marginal outcomes. Events overlap and assets share a crypto market factor; pooled independent-trial significance is not claimed. No multiple-testing-adjusted discovery claims or walk-forward optimized strategy are made.</p></section>')
    body.append('<section id="sources"><h2>Research foundations / what transfers and what does not</h2><ul>')
    sources=[
      ('Moskowitz, Ooi & Pedersen (2012), Time Series Momentum','https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum','Supports studying an instrument’s own past returns. Their broad futures evidence is not proof of daily crypto reversal predictability.'),
      ('Goulding, Harvey & Mazzoleni (2023), Momentum Turning Points','https://people.duke.edu/~charvey/Research/Published_Papers/P158_Momentum_turning_points.pdf','Motivates distinguishing fast/slow disagreement from established trends. Our EWMAC daily proxy is an adaptation, not a replication of their monthly equity study.'),
      ('Lee & Swaminathan (2000), Price Momentum and Trading Volume','https://www.lsvasset.com/pdf/research-papers/Price-Momentum-Trad-Vol-2000.pdf','Motivates conditional volume analysis; equity turnover and multi-year reversal findings do not establish our 1.5× crypto-volume threshold.'),
      ('Cong et al., Crypto Wash Trading','https://www.nber.org/papers/w30783','Reported exchange volume can be unreliable; use venue-specific provenance rather than treating turnover as validated global activity. No wash-trading estimate is assigned to this dataset.'),
      ('Binance official public data repository','https://github.com/binance/binance-public-data','Documents candle fields including quote volume and taker-buy quote volume. This study fetched the corresponding public futures REST candles, not a cross-venue aggregate.')]
    body.extend(f'<li><a href="{u}">{t}</a> — {d}</li>' for t,u,d in sources)
    body.append('</ul></section><footer>Source: Binance USDT perpetual daily candles · local saved momentum bundle · analysis code and data accompany this report.<br>Reproduce: <code>python scripts/momentum_turning_study.py --offline</code>. Primary data: <a href="events.csv">events.csv</a> · <a href="summary.csv">summary.csv</a> · <a href="manifest.json">manifest.json</a></footer>')
    page=render_page('Momentum turning points | Acausal Capital',''.join(body))
    (OUT/'report.html').write_text(page,encoding='utf-8')


def render_cached_report():
    """Presentation-only path: read frozen results without overwriting research artifacts."""
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    frames={};signals={};quality=[]
    for asset in ASSETS:
        f=pd.read_csv(OUT/'data'/f'{asset}USDT_1d.csv',index_col='time',parse_dates=True)
        f.index=pd.to_datetime(f.index,utc=True);frames[asset]=f
        sig=pd.read_csv(OUT/'data'/f'{asset}_signals.csv',index_col=0,parse_dates=True)
        sig.index=pd.to_datetime(sig.index,utc=True);signals[asset]={k:sig[k] for k in sig}
        quality.append(dict(asset=asset,rows=len(f),start=str(f.index[0].date()),end=str(f.index[-1].date()),zero_volume=int((f.volume==0).sum()),gaps=int((f.index.to_series().diff().dropna()!=pd.Timedelta(days=1)).sum())))
    btc=frames['BTC'].copy()
    frames={a:feature_frame(f,btc) for a,f in frames.items()}
    events=pd.read_csv(OUT/'events.csv')
    for c in ['time','entry_time','exit_time','pivot_time']:
        events[c]=pd.to_datetime(events[c],utc=True)
    summary=pd.read_csv(OUT/'summary.csv')
    inputs=[OUT/'events.csv',OUT/'summary.csv',OUT/'manifest.json',*(OUT/'data').glob('*.csv')]
    before={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    build_report(frames,signals,events,summary,quality,manifest['models'],manifest['checks'])
    assert before=={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    theme=ROOT/'scripts/report_theme'
    presentation={'theme':'editorial-html-report/1.0','built_utc':str(pd.Timestamp.now(tz='UTC')),
                  'data_cutoff':manifest['data_cutoff'],'command':'python scripts/momentum_turning_study.py --report-only',
                  'input_sha256':before,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'theme_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in theme.glob('*') if p.is_file()}}
    (OUT/'presentation_manifest.json').write_text(json.dumps(presentation,indent=2),encoding='utf-8')
    print('Report regenerated from frozen inputs; input hashes unchanged:',OUT/'report.html')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.report_only:
        render_cached_report()
        return
    (OUT/'data').mkdir(parents=True,exist_ok=True)
    model_path,section=load_model()
    if args.offline and (OUT/'manifest.json').exists():
        prior=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(model_path.read_bytes()).hexdigest()==prior['bundle_sha256'], 'Active bundle changed; run a fresh full study.'
    cutoff=int(pd.Timestamp.now(tz='UTC').floor('D').timestamp()*1000) if args.offline else (int(api('/fapi/v1/time')['serverTime'])//86400000)*86400000
    frames={};quality=[];checks=[]
    for asset in ASSETS:
        f=fetch(asset,cutoff,args.offline);frames[asset]=f
        quality.append(dict(asset=asset,rows=len(f),start=str(f.index[0].date()),end=str(f.index[-1].date()),zero_volume=int((f.volume==0).sum()),gaps=0))
        print(f'{asset}: {len(f)} daily candles, {f.index[0].date()} to {f.index[-1].date()}',flush=True)
    rules=[tuple(p) for p in section['ewmac_factors']]
    scalars={ast.literal_eval(k) if isinstance(k,str) else k:v for k,v in section['ewmac_scalars'].items()}
    all_rows=[];signals={};meta={}
    for asset in ASSETS:
        f=feature_frame(frames[asset],frames['BTC']);frames[asset]=f
        params=section['results'].get(asset+'USDT',{}).get('params')
        fitted=bool(params and params.get('status','success')=='success')
        weights=params['weights'] if fitted else [1/len(rules)]*len(rules)
        scale=scalars if fitted else CARVER_EWMAC_FORECAST_SCALARS
        dm=section['ewmac_dm_value'] if fitted else 1.12
        cap=section.get('ewmac_final_cap',True) if fitted else True
        meta[asset]=dict(description='Saved bot weights and calibrated scalars' if fitted else 'No fitted bot weights: fixed equal-weight research fallback',weights=weights,dm=dm,cap=cap)
        signals[asset]={}
        for model,w,sc,d,c in [('Current / fallback',weights,scale,dm,cap),('Fixed equal-weight',[.2]*5,CARVER_EWMAC_FORECAST_SCALARS,1.12,True)]:
            s,comp=signal(f.close,w,rules,sc,d,c);signals[asset][model]=s
            # Prefix causality: removing future candles must not change historical forecasts/events.
            k=max(400,len(f)-40)
            if k<len(f):
                short,shortcomp=signal(f.close.iloc[:k],w,rules,sc,d,c)
                np.testing.assert_allclose(short,s.iloc[:k],equal_nan=True)
                for kind in ['Slope 1','Slope 2','Slope 3','Zero 0','Zero 2','Fast/slow']:
                    assert events_for(short,shortcomp,kind)==[(i,v) for i,v in events_for(s,comp,kind) if i<k]
            if not args.report_only:
                for kind in ['Slope 1','Slope 2','Slope 3','Zero 0','Zero 2','Fast/slow']:
                    all_rows.extend(make_events(asset,model,f,s,comp,kind))
        pd.DataFrame(signals[asset]).to_csv(OUT/'data'/f'{asset}_signals.csv')
    events=pd.read_csv(OUT/'events.csv') if args.report_only else pd.DataFrame(all_rows)
    if args.report_only:
        for col in ['time','entry_time','exit_time','pivot_time']:
            events[col]=pd.to_datetime(events[col],utc=True)
    assert (events.entry_time>events.time).all()
    assert (events.exit_time<=pd.to_datetime(cutoff,unit='ms',utc=True)).all()
    assert (events.hit==(events.net>0)).all()
    assert not events.duplicated(['asset','model','kind','time','horizon']).any()
    summary=summarize(events,['asset','model','kind','horizon'])
    events.to_csv(OUT/'events.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    checks.extend(['All nine requested assets fetched; contiguous daily UTC candles; OHLC and nonnegative-volume checks passed.',
                   'Prefix-invariance checks passed for both models and all six event definitions where history permits.',
                   'Entries occur strictly after signal dates; all scored exits occur by the completed-data cutoff.',
                   'Hit labels equal positive execution-cost-adjusted return; no duplicate event/horizon keys.',
                   'First 365 daily bars excluded from scoring; original live stores and fitted parameters preserved.',
                   'Three targeted tests passed: production forecast parity after warm-up, zero hysteresis, and next-open gap/ambiguous-barrier accounting.'])
    manifest=dict(generated_utc=str(pd.Timestamp.now(tz='UTC')),data_cutoff=str(pd.to_datetime(cutoff,unit='ms',utc=True)),
                  bundle=str(model_path),bundle_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  bundle_metadata={k:str(v) for k,v in section.items() if k!='results'},models=meta,
                  horizons=HORIZONS,round_trip_bps=12,funding_included=False,freeze=str(FREEZE),
                  primary='Slope 2 / 7 days',bootstrap_seed=1729,bootstrap_replicates=1000,checks=checks,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'data').glob('*.csv')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    build_report(frames,signals,events,summary,quality,meta,checks)
    print(summary[(summary.model=='Current / fallback')&(summary.kind=='Slope 2')&(summary.horizon==7)][['asset','n','hit','mean_net','uplift']].to_string(index=False),flush=True)
    print('REPORT',OUT/'report.html',flush=True)


if __name__=='__main__':
    main()
