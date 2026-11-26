"""
Matka Predict Pro v7 — Client Dashboard  (v3, Nov 2026)
Walk-forward prediction engine · 35 strategies · 13 markets
Tabs: Predictions · Backtest Explorer · Performance · Jodi Analysis · Update Data
Strategy names and methodology are NOT exposed in the UI.
"""

import streamlit as st
import json, os, io
from datetime import date, datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
ALL_MKT = [
    'Sridevi','TimeBazar','MadhurDay','MilanDay','RajdhaniDay','Kalyan',
    'SrideviNight','MadhurNight','SupremeNight','MilanNight',
    'RajdhaniNight','KalyanNight','MainBazar'
]
SAT_CLOSED = {'RajdhaniNight','KalyanNight','MainBazar'}
SUN_OPEN   = {'Sridevi','MadhurDay','SrideviNight','SupremeNight'}
DOW_NAMES  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DAY_MKT    = {'Sridevi','TimeBazar','MadhurDay','MilanDay','RajdhaniDay','Kalyan'}

# ─────────────────────────────────────────────────────────────
# STRATEGY ENGINE  — 35 algorithms (names never shown in UI)
# ─────────────────────────────────────────────────────────────
def _norm(x):
    s = x.sum()
    return x / s if s > 0 else np.ones(10) / 10


def _run_strat(name, ops, cps, dows, tdow, n):
    """Return (op_probs[10], cp_probs[10]) for a named strategy."""
    def ew(dc=15, W=90):
        Ww = min(n, W)
        e  = np.exp(-np.arange(Ww, 0, -1) / dc); e /= e.sum()
        return (np.array([(e * (ops[n-Ww:n] == d)).sum() for d in range(10)]),
                np.array([(e * (cps[n-Ww:n] == d)).sum() for d in range(10)]))

    def df(last=20):
        s = np.where(dows[:n] == tdow)[0][-last:]
        if len(s) >= 5:
            return (np.array([(ops[s] == d).mean() for d in range(10)]),
                    np.array([(cps[s] == d).mean() for d in range(10)]))
        return None, None

    W3  = min(n,  3);  W7  = min(n,  7); W14 = min(n, 14); W30 = min(n,  30)
    W60 = min(n, 60); W90 = min(n, 90); W120= min(n,120); W180= min(n, 180)
    ol, cl = ew(15)

    if name == 'Hot-Cold':
        return (_norm((np.array([(ops[n-W7:n]==d).mean() for d in range(10)])+.01)/(ol+.01)),
                _norm((np.array([(cps[n-W7:n]==d).mean() for d in range(10)])+.01)/(cl+.01)))
    elif name in ('Mean-Revert-30','Mean-Revert-60','Mean-Revert-120'):
        W = {'Mean-Revert-30':W30,'Mean-Revert-60':W60,'Mean-Revert-120':W120}[name]
        return (_norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Recent-30':
        return (_norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])),
                _norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])))
    elif name == 'GeoAntiF-DOW':
        og = np.array([(ops[n-W90:n]==d).mean() for d in range(10)])
        cg = np.array([(cps[n-W90:n]==d).mean() for d in range(10)])
        od, cd = df()
        return (_norm(1/((og+.01)*(od+.01))) if od is not None else _norm(1/(og+.01)),
                _norm(1/((cg+.01)*(cd+.01))) if cd is not None else _norm(1/(cg+.01)))
    elif name == 'Anti-Freq-DOW':
        od, cd = df()
        return (_norm(1/(od+.01)) if od is not None else _norm(1/(np.array([(ops[n-W90:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(cd+.01)) if cd is not None else _norm(1/(np.array([(cps[n-W90:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Anti-Freq-DOW-50':
        od50, cd50 = df(50)
        return (_norm(1/(od50+.01)) if od50 is not None else _norm(1/(np.array([(ops[n-W90:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(cd50+.01)) if cd50 is not None else _norm(1/(np.array([(cps[n-W90:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Decay-1':  o,c = ew(1,30);   return _norm(o), _norm(c)
    elif name == 'Decay-3':  o,c = ew(3,60);   return _norm(o), _norm(c)
    elif name == 'Decay-5':  o,c = ew(5,90);   return _norm(o), _norm(c)
    elif name == 'Decay5+AntiF':
        od, cd = ew(5, W90)
        return _norm(.5*_norm(od)+.5*_norm(1/(od+.01))), _norm(.5*_norm(cd)+.5*_norm(1/(cd+.01)))
    elif name == 'Balanced':
        oaf = _norm(1/(ol+.01)); caf = _norm(1/(cl+.01))
        o14 = np.array([(ops[n-W14:n]==d).mean() for d in range(10)])
        c14 = np.array([(cps[n-W14:n]==d).mean() for d in range(10)])
        od, cd = df(); od_b = od if od is not None else ol; cd_b = cd if cd is not None else cl
        return _norm(.4*oaf+.35*o14+.25*od_b), _norm(.4*caf+.35*c14+.25*cd_b)
    elif name == 'DOW-Only':
        sdow = np.where(dows[:n] == tdow)[0]
        if len(sdow) >= 8:
            idx = sdow[-min(len(sdow),30):]
            e   = np.exp(-np.arange(len(idx),0,-1)/10); e /= e.sum()
            return (_norm(1/(np.array([(e*(ops[idx]==d)).sum() for d in range(10)])+.01)),
                    _norm(1/(np.array([(e*(cps[idx]==d)).sum() for d in range(10)])+.01)))
        return (_norm(1/(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Markov-1':
        if n > 5:
            M  = np.ones((10,10))*.1; M2 = np.ones((10,10))*.1
            for k in range(1,n): M[ops[k-1],ops[k]] += 1; M2[cps[k-1],cps[k]] += 1
            M /= M.sum(axis=1,keepdims=True); M2 /= M2.sum(axis=1,keepdims=True)
            return _norm(M[ops[n-1]]), _norm(M2[cps[n-1]])
        return _norm(np.ones(10)), _norm(np.ones(10))
    elif name == 'Markov-2':
        if n > 6:
            M3 = np.ones((10,10,10))*.05; M4 = np.ones((10,10,10))*.05
            for k in range(2,n): M3[ops[k-2],ops[k-1],ops[k]] += 1; M4[cps[k-2],cps[k-1],cps[k]] += 1
            return _norm(M3[ops[n-2],ops[n-1]]), _norm(M4[cps[n-2],cps[n-1]])
        return _norm(np.ones(10)), _norm(np.ones(10))
    elif name == 'Extreme-Anti':
        o = np.array([(ops[n-W60:n]==d).mean()+.01 for d in range(10)])
        c = np.array([(cps[n-W60:n]==d).mean()+.01 for d in range(10)])
        return _norm(1/o**2), _norm(1/c**2)
    elif name == 'Recency-Burst':
        od10, cd10 = ew(10)
        op_r = np.array([(ops[n-W3:n]==d).mean() for d in range(10)])
        cp_r = np.array([(cps[n-W3:n]==d).mean() for d in range(10)])
        return (_norm(.3*_norm(1/(op_r+.01))+.7*_norm(od10)),
                _norm(.3*_norm(1/(cp_r+.01))+.7*_norm(cd10)))
    elif name == 'Decay-2':  o,c = ew(2,45);   return _norm(o), _norm(c)
    elif name == 'Decay-7':  o,c = ew(7,90);   return _norm(o), _norm(c)
    elif name == 'Decay-10': o,c = ew(10,120); return _norm(o), _norm(c)
    elif name == 'Decay-20': o,c = ew(20,150); return _norm(o), _norm(c)
    elif name == 'Recent-14':
        return (_norm(np.array([(ops[n-W14:n]==d).mean() for d in range(10)])),
                _norm(np.array([(cps[n-W14:n]==d).mean() for d in range(10)])))
    elif name == 'Recent-60':
        return (_norm(np.array([(ops[n-W60:n]==d).mean() for d in range(10)])),
                _norm(np.array([(cps[n-W60:n]==d).mean() for d in range(10)])))
    elif name == 'Anti-Last3':
        op_r3 = np.array([(ops[n-W3:n]==d).mean() for d in range(10)])
        cp_r3 = np.array([(cps[n-W3:n]==d).mean() for d in range(10)])
        return _norm(1/(op_r3+.01)), _norm(1/(cp_r3+.01))
    elif name == 'DOW-Anti':
        s = np.where(dows[:n] == tdow)[0][-30:]
        if len(s) >= 5:
            od = np.array([(ops[s]==d).mean() for d in range(10)])
            cd = np.array([(cps[s]==d).mean() for d in range(10)])
            return _norm(1/(od+.01)), _norm(1/(cd+.01))
        return (_norm(1/(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Skip-Recent-5':
        W5 = min(n,5)
        op = np.array([(ops[n-W5:n]==d).mean() for d in range(10)])
        cp = np.array([(cps[n-W5:n]==d).mean() for d in range(10)])
        return _norm(1/(op+.01)), _norm(1/(cp+.01))
    elif name == 'Combo-Top3':
        d3o, d3c = ew(3, 60)
        hc_o = _norm((np.array([(ops[n-W7:n]==d).mean() for d in range(10)])+.01)/(ol+.01))
        hc_c = _norm((np.array([(cps[n-W7:n]==d).mean() for d in range(10)])+.01)/(cl+.01))
        r30o = _norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)]))
        r30c = _norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)]))
        return _norm(_norm(d3o)/3+hc_o/3+r30o/3), _norm(_norm(d3c)/3+hc_c/3+r30c/3)
    elif name == 'Hybrid-60':
        af60o = _norm(1/(np.array([(ops[n-W60:n]==d).mean() for d in range(10)])+.01))
        af60c = _norm(1/(np.array([(cps[n-W60:n]==d).mean() for d in range(10)])+.01))
        r60o  = _norm(np.array([(ops[n-W60:n]==d).mean() for d in range(10)]))
        r60c  = _norm(np.array([(cps[n-W60:n]==d).mean() for d in range(10)]))
        s = np.where(dows[:n] == tdow)[0][-20:]; dow_w = 0.25 if len(s) >= 5 else 0.0
        od = np.array([(ops[s]==d).mean() for d in range(10)]) if len(s)>=5 else np.ones(10)/10
        cd = np.array([(cps[s]==d).mean() for d in range(10)]) if len(s)>=5 else np.ones(10)/10
        return (_norm((0.5-dow_w/2)*af60o+(0.25-dow_w/4)*r60o+(0.25-dow_w/4)*ol+dow_w*_norm(1/(od+.01))),
                _norm((0.5-dow_w/2)*af60c+(0.25-dow_w/4)*r60c+(0.25-dow_w/4)*cl+dow_w*_norm(1/(cd+.01))))
    elif name == 'Markov-3':
        if n > 7:
            M5 = np.ones((10,10,10,10))*.02; M6 = np.ones((10,10,10,10))*.02
            for k in range(3,n):
                M5[ops[k-3],ops[k-2],ops[k-1],ops[k]] += 1
                M6[cps[k-3],cps[k-2],cps[k-1],cps[k]] += 1
            s5 = M5[ops[n-3],ops[n-2],ops[n-1]]; s5 /= s5.sum()
            s6 = M6[cps[n-3],cps[n-2],cps[n-1]]; s6 /= s6.sum()
            return _norm(s5), _norm(s6)
        return _norm(np.ones(10)), _norm(np.ones(10))
    elif name == 'Mean-Revert-180':
        W = min(n, 180)
        return (_norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Volatility-Anti':
        W = W90; gap_o = np.zeros(10); gap_c = np.zeros(10)
        for d in range(10):
            idx_o = np.where(ops[n-W:n] == d)[0]
            idx_c = np.where(cps[n-W:n] == d)[0]
            gap_o[d] = np.diff(idx_o).std()+1 if len(idx_o)>2 else 15
            gap_c[d] = np.diff(idx_c).std()+1 if len(idx_c)>2 else 15
        return _norm(gap_o), _norm(gap_c)
    elif name == 'Long-Anti-180':
        W = min(n, 180)
        return (_norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)**1.5),
                _norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)**1.5))
    elif name == 'DOW-Freq':
        s = np.where(dows[:n] == tdow)[0][-40:]
        if len(s) >= 5:
            od = np.array([(ops[s]==d).mean() for d in range(10)])
            cd = np.array([(cps[s]==d).mean() for d in range(10)])
            return _norm(od+.01), _norm(cd+.01)
        return (_norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01),
                _norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01))
    elif name == 'Decay3+DOW':
        d3o, d3c = ew(3, 60)
        s = np.where(dows[:n] == tdow)[0][-20:]
        if len(s) >= 5:
            od = np.array([(ops[s]==d).mean() for d in range(10)])
            cd = np.array([(cps[s]==d).mean() for d in range(10)])
            dow_a_o = _norm(1/(od+.01)); dow_a_c = _norm(1/(cd+.01))
        else:
            dow_a_o = _norm(1/(ol+.01)); dow_a_c = _norm(1/(cl+.01))
        return _norm(.6*_norm(d3o)+.4*dow_a_o), _norm(.6*_norm(d3c)+.4*dow_a_c)
    else:
        o, c = ew(15); return _norm(o), _norm(c)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def _load_all():
    bt  = json.load(open(os.path.join(DATA_DIR, 'bt_data.json')))
    ma  = json.load(open(os.path.join(DATA_DIR, 'market_assign.json')))
    nsr = json.load(open(os.path.join(DATA_DIR, 'newx_strategy_results.json')))
    return bt, ma, nsr


def _get_series(bt, market, cutoff_str):
    """Extract ops/cps/dows arrays for dates strictly before cutoff_str."""
    rows = [r for r in bt.get(market, []) if r['date'] < cutoff_str]
    if len(rows) < 10:
        return None, None, None, 0
    ops  = np.array([r['actual_op'] for r in rows], dtype=int)
    cps  = np.array([r['actual_cp'] for r in rows], dtype=int)
    dows = np.array([datetime.strptime(r['date'], '%Y-%m-%d').weekday() for r in rows], dtype=int)
    return ops, cps, dows, len(rows)


def _markets_open_on(tdow):
    if tdow == 5: return [m for m in ALL_MKT if m not in SAT_CLOSED]
    if tdow == 6: return [m for m in ALL_MKT if m in SUN_OPEN]
    return ALL_MKT


# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
def gen_predictions(bt, ma, nsr, target_str, tdow):
    """Generate Model A + Model B predictions for all active markets on target_str."""
    results = {}
    for mkt in _markets_open_on(tdow):
        ops, cps, dows, n = _get_series(bt, mkt, target_str)
        if ops is None:
            continue
        # Model A — single optimised strategy per market
        cur_s = ma.get(mkt, {}).get('strategy', 'Hot-Cold')
        # Hardcoded override matching main system (daily_runner.py line 1129)
        if mkt == 'SupremeNight':
            cur_s = 'Mean-Revert-30'
        op_p, cp_p = _run_strat(cur_s, ops, cps, dows, tdow, n)
        # Model B — blends best + 2nd-best strategy (60/40), matches daily_runner.py
        cfg   = nsr.get(mkt, {})
        bos   = cfg.get('best_open',  cur_s)
        bcs   = cfg.get('best_close', cur_s)
        bos2_l = cfg.get('top2_open',  [bos, bos])
        bcs2_l = cfg.get('top2_close', [bcs, bcs])
        bos2  = bos2_l[1] if len(bos2_l) > 1 else bos
        bcs2  = bcs2_l[1] if len(bcs2_l) > 1 else bcs
        try:    nx_op_p,  _ = _run_strat(bos,  ops, cps, dows, tdow, n)
        except: nx_op_p     = np.ones(10) / 10
        try:    _,  nx_cp_p = _run_strat(bcs,  ops, cps, dows, tdow, n)
        except: nx_cp_p     = np.ones(10) / 10
        try:    nx_op_p2, _ = _run_strat(bos2, ops, cps, dows, tdow, n)
        except: nx_op_p2    = nx_op_p
        try:    _, nx_cp_p2 = _run_strat(bcs2, ops, cps, dows, tdow, n)
        except: nx_cp_p2    = nx_cp_p
        nx_op_p = _norm(np.array(nx_op_p) * 0.6 + np.array(nx_op_p2) * 0.4)
        nx_cp_p = _norm(np.array(nx_cp_p) * 0.6 + np.array(nx_cp_p2) * 0.4)
        # Recent hit rate from stored bt (last-30 sessions)
        last30 = [r for r in bt.get(mkt, []) if r['date'] < target_str][-30:]
        h4     = sum(r.get('hit4', 0) for r in last30) / max(len(last30), 1)
        results[mkt] = dict(
            top4_op    = list(map(int, np.argsort(op_p)[::-1][:4])),
            top4_cp    = list(map(int, np.argsort(cp_p)[::-1][:4])),
            op_probs   = [round(float(x), 4) for x in op_p],
            cp_probs   = [round(float(x), 4) for x in cp_p],
            nx_top4_op = list(map(int, np.argsort(nx_op_p)[::-1][:4])),
            nx_top4_cp = list(map(int, np.argsort(nx_cp_p)[::-1][:4])),
            nx_op_probs= [round(float(x), 4) for x in nx_op_p],
            nx_cp_probs= [round(float(x), 4) for x in nx_cp_p],
            h4_rate    = round(h4, 3),
        )
    return results


def run_backtest_date(bt, ma, nsr, target_str):
    """Run predictions for a past date and compare with actual results."""
    tdow = datetime.strptime(target_str, '%Y-%m-%d').weekday()
    preds = gen_predictions(bt, ma, nsr, target_str, tdow)
    rows  = []
    for mkt, p in preds.items():
        # Fetch actual result for that date
        actual = next((r for r in bt.get(mkt, []) if r['date'] == target_str), None)
        if actual is None:
            aop = acp = None
        else:
            aop = actual.get('actual_op')
            acp = actual.get('actual_cp')
        hit_a_op = aop in p['top4_op'] if aop is not None else None
        hit_a_cp = acp in p['top4_cp'] if acp is not None else None
        hit_b_op = aop in p['nx_top4_op'] if aop is not None else None
        hit_b_cp = acp in p['nx_top4_cp'] if acp is not None else None
        rows.append(dict(
            market   = mkt,
            actual_op= aop,
            actual_cp= acp,
            a_top4_op= p['top4_op'],
            a_top4_cp= p['top4_cp'],
            b_top4_op= p['nx_top4_op'],
            b_top4_cp= p['nx_top4_cp'],
            hit_a_op = hit_a_op,
            hit_a_cp = hit_a_cp,
            hit_b_op = hit_b_op,
            hit_b_cp = hit_b_cp,
        ))
    return rows


# ─────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────
def compute_all_stats(bt):
    """Comprehensive stats from bt_data.  Runs after cached load — fast pure-Python."""
    today  = date.today()

    # Window hit rates per market
    windows = {'30d': 30, '60d': 60, '90d': 90, '180d': 180}
    win = {k: {} for k in windows}
    for mkt in ALL_MKT:
        for key, days in windows.items():
            cutoff = (today - timedelta(days)).strftime('%Y-%m-%d')
            sub    = [r for r in bt.get(mkt, []) if r['date'] >= cutoff]
            if sub:
                win[key][mkt] = round(sum(r.get('hit4', 0) for r in sub) / len(sub) * 100, 1)

    # Monthly hit rates
    monthly = defaultdict(lambda: defaultdict(lambda: {'h': 0, 't': 0}))
    for mkt in ALL_MKT:
        for r in bt.get(mkt, []):
            ym = r['date'][:7]
            monthly[ym][mkt]['h'] += r.get('hit4', 0)
            monthly[ym][mkt]['t'] += 1
    months_sorted = sorted(monthly.keys())

    # DOW hit rates (combined all markets)
    dow_hits = defaultdict(lambda: {'h': 0, 't': 0})
    for mkt in ALL_MKT:
        for r in bt.get(mkt, []):
            dw = datetime.strptime(r['date'], '%Y-%m-%d').weekday()
            dow_hits[dw]['h'] += r.get('hit4', 0)
            dow_hits[dw]['t'] += 1

    # Open vs Close accuracy
    oc = {}
    for mkt in ALL_MKT:
        rows = bt.get(mkt, [])
        if not rows: continue
        op_hits = sum(1 for r in rows if r.get('actual_op') in (r.get('top4_op') or []))
        cp_hits = sum(1 for r in rows if r.get('actual_cp') in (r.get('top4_cp') or []))
        oc[mkt] = {'op': round(op_hits/len(rows)*100, 1),
                   'cp': round(cp_hits/len(rows)*100, 1),
                   'n' : len(rows)}

    # Zero-hit sessions per market
    zero = {}
    for mkt in ALL_MKT:
        rows = bt.get(mkt, [])
        if not rows: continue
        z = sum(1 for r in rows if r.get('hit4', 0) == 0)
        zero[mkt] = {'zero': z, 'total': len(rows),
                     'pct': round(z / len(rows) * 100, 1)}

    # Overall totals
    total_days  = sum(len(bt.get(m, [])) for m in ALL_MKT)
    total_hits  = sum(r.get('hit4', 0) for m in ALL_MKT for r in bt.get(m, []))
    overall_pct = round(total_hits / total_days * 100, 1) if total_days else 0

    return dict(win=win, monthly=monthly, months_sorted=months_sorted,
                dow_hits=dow_hits, oc=oc, zero=zero,
                total_days=total_days, overall_pct=overall_pct)


def compute_jodi_stats(bt, days=90):
    cutoff = (date.today() - timedelta(days)).strftime('%Y-%m-%d')
    stats  = {}
    for mkt in ALL_MKT:
        rows = [r for r in bt.get(mkt, [])
                if r['date'] >= cutoff and len(str(r.get('actual_jodi', ''))) == 2]
        if not rows: continue
        freq = defaultdict(int)
        for r in rows:
            freq[str(r['actual_jodi'])] += 1
        total  = len(rows)
        sorted_jd = sorted(freq.items(), key=lambda x: -x[1])
        stats[mkt] = {'hot': sorted_jd[:5], 'cold': sorted_jd[-5:][::-1], 'total': total}
    return stats


def compute_digit_freq(bt, days=30):
    cutoff = (date.today() - timedelta(days)).strftime('%Y-%m-%d')
    freq   = {}
    for mkt in ALL_MKT:
        rows = [r for r in bt.get(mkt, []) if r['date'] >= cutoff]
        if not rows: continue
        freq[mkt] = {
            'op': [sum(r['actual_op'] == d for r in rows) / len(rows) for d in range(10)],
            'cp': [sum(r['actual_cp'] == d for r in rows) / len(rows) for d in range(10)],
            'n' : len(rows),
        }
    return freq


# ─────────────────────────────────────────────────────────────
# EXCEL PARSER
# ─────────────────────────────────────────────────────────────
def _panna_to_digit(v):
    try:
        s = str(int(v)).zfill(3)
        return (int(s[0]) + int(s[1]) + int(s[2])) % 10
    except:
        return None


def parse_excel_to_bt(uploaded_bytes, existing_bt):
    try:
        xl  = pd.ExcelFile(io.BytesIO(uploaded_bytes))
        bt  = {k: list(v) for k, v in existing_bt.items()}
        new_count = 0
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df.columns = [str(c).strip() for c in df.columns]
            col_map = {}
            for col in df.columns:
                cl = col.lower()
                if 'date' in cl:                              col_map['date']   = col
                elif any(x in cl for x in ['market','mkt']): col_map['market'] = col
                elif cl in ('open','op'):                     col_map['open']   = col
                elif cl in ('close','cp'):                    col_map['close']  = col
            sheet_mkt = next((m for m in ALL_MKT if m.lower() in sheet.lower()), None)
            if 'date' not in col_map or 'open' not in col_map:
                continue
            existing_dates = {mkt: {r['date'] for r in bt.get(mkt, [])} for mkt in ALL_MKT}
            for _, row in df.iterrows():
                try:
                    dt_raw = row[col_map['date']]
                    if pd.isna(dt_raw): continue
                    if isinstance(dt_raw, (datetime, pd.Timestamp)):
                        dt_str = pd.Timestamp(dt_raw).strftime('%Y-%m-%d')
                    else:
                        dt_str = str(dt_raw).strip()[:10]
                    datetime.strptime(dt_str, '%Y-%m-%d')
                    mkt = sheet_mkt
                    if 'market' in col_map:
                        mr  = str(row[col_map['market']]).strip()
                        mkt = next((m for m in ALL_MKT if m.lower() == mr.lower()), mkt)
                    if not mkt: continue
                    ov = row[col_map['open']]
                    if pd.isna(ov): continue
                    iv    = int(ov)
                    op_d  = iv % 10 if iv < 10 else _panna_to_digit(iv)
                    cp_d  = None
                    if 'close' in col_map:
                        cv = row[col_map['close']]
                        if cv is not None and not (isinstance(cv, float) and np.isnan(cv)):
                            icv  = int(cv)
                            cp_d = icv % 10 if icv < 10 else _panna_to_digit(icv)
                    if op_d is None: continue
                    if dt_str in existing_dates.get(mkt, set()): continue
                    if mkt not in bt: bt[mkt] = []
                    jodi = f"{op_d}{cp_d if cp_d is not None else op_d}"
                    bt[mkt].append({'date': dt_str, 'actual_op': op_d,
                                    'actual_cp': cp_d if cp_d is not None else op_d,
                                    'actual_jodi': jodi, 'hit4': 0, 'strategy': 'uploaded'})
                    existing_dates.setdefault(mkt, set()).add(dt_str)
                    new_count += 1
                except:
                    continue
        for mkt in bt:
            bt[mkt] = sorted(bt[mkt], key=lambda r: r['date'])
        return bt, new_count, None
    except Exception as e:
        return existing_bt, 0, str(e)


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def _stag(mkt):   return "☀️" if mkt in DAY_MKT else "🌙"
def _d4(digits):  return " · ".join(str(d) for d in digits)

def _pills(digits, cls="blue"):
    colors = {"blue": "#1a73e8", "green": "#34a853"}
    c = colors.get(cls, "#1a73e8")
    return "".join(
        f'<span style="display:inline-block;width:30px;height:30px;border-radius:50%;'
        f'background:{c};color:white;text-align:center;line-height:30px;'
        f'font-weight:800;font-size:14px;margin:2px">{d}</span>'
        for d in digits)

def _hr_badge(hr):
    if hr >= 45: return f'<span style="color:#34a853;font-weight:700">{hr:.1f}%</span>'
    if hr >= 38: return f'<span style="color:#fbbc04;font-weight:700">{hr:.1f}%</span>'
    return f'<span style="color:#ea4335;font-weight:700">{hr:.1f}%</span>'

def _trend_arrow(monthly, mkt, months_sorted):
    if len(months_sorted) < 2: return "→"
    last2 = [monthly[m].get(mkt, {}) for m in months_sorted[-2:]]
    r0 = last2[0].get('h', 0) / max(last2[0].get('t', 1), 1)
    r1 = last2[1].get('h', 0) / max(last2[1].get('t', 1), 1)
    if r1 > r0 + 0.03: return "↑"
    if r1 < r0 - 0.03: return "↓"
    return "→"

def _hit_icon(hit):
    if hit is None:  return "—"
    if hit:          return "✅"
    return "❌"


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Matka Predict Pro v7",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""<style>
    .block-container{padding-top:1rem}
    .pred-card{background:#1e2130;border-radius:10px;padding:12px 18px;margin:5px 0;border-left:4px solid #1a73e8}
    .top-card{background:linear-gradient(135deg,#0d2a0d,#1e2130);border:1px solid #34a853;border-radius:10px;padding:12px 18px;margin:4px 0}
    .bot-card{background:linear-gradient(135deg,#2a0d0d,#1e2130);border:1px solid #ea4335;border-radius:10px;padding:12px 18px;margin:4px 0}
    .kpi{background:#1a1f35;border-radius:8px;padding:14px;text-align:center}
    .kpi-v{font-size:2em;font-weight:800;color:#1a73e8}
    .kpi-l{font-size:0.75em;color:#8891b0;margin-top:4px}
    </style>""", unsafe_allow_html=True)

    # ── SIDEBAR ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 Matka Predict Pro")
        st.caption("v7 · Confidential")
        st.markdown("---")
        today  = date.today()
        target = st.date_input(
            "📅 Prediction Date",
            value=today,
            min_value=date(2025, 4, 1),
            max_value=today + timedelta(7),
        )
        tdow = target.weekday()
        tstr = target.strftime('%Y-%m-%d')
        st.markdown(f"**{DOW_NAMES[tdow]}** · `{tstr}`")
        if tdow == 5: st.warning("Saturday: 3 markets closed")
        if tdow == 6: st.info("Sunday: 4 markets open only")

        st.markdown("---")
        st.markdown("**📂 Add New Results**")
        uploaded = st.file_uploader(
            "Upload Excel (.xlsx)",
            type=["xlsx"],
            help="Columns: Date | Market | Open | Close",
        )
        st.markdown("---")
        model_filter = st.radio("Show Model", ["Both", "Model A", "Model B"])

    # ── LOAD DATA ───────────────────────────────────────────────
    bt, ma, nsr = _load_all()

    upload_msg = None
    if uploaded:
        new_bt, cnt, err = parse_excel_to_bt(uploaded.read(), bt)
        if err:
            st.error(f"Upload error: {err}")
        else:
            bt = new_bt
            upload_msg = (f"✅ Added {cnt} new rows." if cnt
                          else "ℹ️ No new rows (all dates already exist).")
            st.cache_data.clear()
    if upload_msg:
        st.success(upload_msg)

    # ── GENERATE PREDICTIONS FOR SELECTED DATE ──────────────────
    with st.spinner(f"Running models for {tstr}…"):
        preds = gen_predictions(bt, ma, nsr, tstr, tdow)

    if not preds:
        st.error("No active markets for this date."); return

    mkt_list = list(preds.keys())
    ranked   = sorted(mkt_list, key=lambda m: preds[m]['h4_rate'], reverse=True)

    # ── KPI BAR ─────────────────────────────────────────────────
    st.markdown(f"## 🎯 Matka Predict Pro  —  {tstr} ({DOW_NAMES[tdow]})")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(mkt_list), "Active Markets"),
        (c2, len([m for m in mkt_list if m in DAY_MKT]), "☀️ Day Markets"),
        (c3, len([m for m in mkt_list if m not in DAY_MKT]), "🌙 Night Markets"),
        (c4, round(sum(preds[m]['h4_rate'] for m in mkt_list)/len(mkt_list)*len(mkt_list)),
              "Expected Hits"),
    ]:
        col.markdown(
            f'<div class="kpi"><div class="kpi-v">{val}</div>'
            f'<div class="kpi-l">{lbl}</div></div>',
            unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── TOP 5 / BOTTOM 3 ────────────────────────────────────────
    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 🏆 Top 5 — Most Reliable Today")
        for i, mkt in enumerate(ranked[:5], 1):
            p  = preds[mkt]
            hr = p['h4_rate'] * 100
            st.markdown(
                f'<div class="top-card">'
                f'<b style="color:#34a853">#{i}</b> {_stag(mkt)} <b>{mkt}</b>'
                f'<span style="float:right">{_hr_badge(hr)}</span><br>'
                f'<small>Open: <b>{_d4(p["top4_op"])}</b> &nbsp;|&nbsp; '
                f'Close: <b>{_d4(p["top4_cp"])}</b></small></div>',
                unsafe_allow_html=True)
    with colB:
        st.markdown("### ⚠️ Bottom 3 — Lowest Reliability")
        for i, mkt in enumerate(ranked[-3:], 1):
            p  = preds[mkt]
            hr = p['h4_rate'] * 100
            st.markdown(
                f'<div class="bot-card">'
                f'⚠️ {_stag(mkt)} <b>{mkt}</b>'
                f'<span style="float:right">{_hr_badge(hr)}</span><br>'
                f'<small>Open: <b>{_d4(p["top4_op"])}</b> &nbsp;|&nbsp; '
                f'Close: <b>{_d4(p["top4_cp"])}</b></small></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── MAIN TABS ───────────────────────────────────────────────
    tab_pred, tab_bt, tab_perf, tab_jodi, tab_upd = st.tabs([
        "📋 All Predictions",
        "📅 Backtest Explorer",
        "📊 Performance",
        "🎲 Jodi Analysis",
        "📥 Data Update",
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — PREDICTIONS
    # ════════════════════════════════════════════════════════════
    with tab_pred:
        show_a = model_filter in ("Both", "Model A")
        show_b = model_filter in ("Both", "Model B")

        # Summary table
        tbl_rows = []
        for mkt in mkt_list:
            p  = preds[mkt]
            hr = p['h4_rate'] * 100
            row = {'Market': f"{_stag(mkt)} {mkt}", 'Hit Rate': f"{hr:.1f}%"}
            if show_a:
                row['A — Open']  = _d4(p['top4_op'])
                row['A — Close'] = _d4(p['top4_cp'])
            if show_b:
                row['B — Open']  = _d4(p['nx_top4_op'])
                row['B — Close'] = _d4(p['nx_top4_cp'])
            tbl_rows.append(row)
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

        # Detailed cards
        st.markdown("#### Detailed Probability View")
        cols = st.columns(2)
        for i, mkt in enumerate(mkt_list):
            p  = preds[mkt]
            hr = p['h4_rate'] * 100
            with cols[i % 2]:
                with st.expander(f"{_stag(mkt)} {mkt}  —  {hr:.1f}% 30d hit rate"):
                    if show_a:
                        st.markdown("**Model A**")
                        st.markdown(_pills(p['top4_op'], "blue"), unsafe_allow_html=True)
                        st.caption("Open: " + " → ".join(f"{d}({p['op_probs'][d]*100:.1f}%)" for d in p['top4_op']))
                        st.markdown(_pills(p['top4_cp'], "blue"), unsafe_allow_html=True)
                        st.caption("Close: " + " → ".join(f"{d}({p['cp_probs'][d]*100:.1f}%)" for d in p['top4_cp']))
                        if show_b: st.markdown("---")

                    if show_b:
                        st.markdown("**Model B**")
                        st.markdown(_pills(p['nx_top4_op'], "green"), unsafe_allow_html=True)
                        st.caption("Open: " + " → ".join(f"{d}({p['nx_op_probs'][d]*100:.1f}%)" for d in p['nx_top4_op']))
                        st.markdown(_pills(p['nx_top4_cp'], "green"), unsafe_allow_html=True)
                        st.caption("Close: " + " → ".join(f"{d}({p['nx_cp_probs'][d]*100:.1f}%)" for d in p['nx_top4_cp']))

                    # Probability chart
                    df_prob = pd.DataFrame({
                        'Digit': list(range(10)),
                        **({"A Open %": [round(x*100,1) for x in p['op_probs']]} if show_a else {}),
                        **({"B Open %": [round(x*100,1) for x in p['nx_op_probs']]} if show_b else {}),
                    }).set_index('Digit')
                    st.bar_chart(df_prob, height=150)

                    # Both-model agreement
                    if show_a and show_b:
                        agree_o = set(p['top4_op']) & set(p['nx_top4_op'])
                        agree_c = set(p['top4_cp']) & set(p['nx_top4_cp'])
                        if agree_o or agree_c:
                            msg = []
                            if agree_o: msg.append(f"Open: **{', '.join(str(d) for d in sorted(agree_o))}**")
                            if agree_c: msg.append(f"Close: **{', '.join(str(d) for d in sorted(agree_c))}**")
                            st.success("Both models agree — " + "  |  ".join(msg))

    # ════════════════════════════════════════════════════════════
    # TAB 2 — BACKTEST EXPLORER
    # ════════════════════════════════════════════════════════════
    with tab_bt:
        st.markdown("### 📅 Backtest Explorer")
        st.caption(
            "Pick any past date to see what the current model **would have predicted** (using only "
            "data before that date), and compare with the actual result. "
            "This is the same walk-forward logic used for today's predictions."
        )

        # Find date range from bt_data
        all_dates = sorted({r['date'] for mkt in ALL_MKT for r in bt.get(mkt, [])})
        if len(all_dates) < 2:
            st.warning("Not enough historical data yet."); st.stop()

        min_date = datetime.strptime(all_dates[16], '%Y-%m-%d').date()   # need 15+ rows
        max_date = datetime.strptime(all_dates[-1], '%Y-%m-%d').date()

        col1, col2 = st.columns([2, 1])
        with col1:
            bt_date = st.date_input(
                "Select date to explore",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key='bt_date_pick',
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_bt = st.button("🔍 Run Comparison", use_container_width=True)

        if run_bt or 'bt_rows' not in st.session_state:
            with st.spinner("Running walk-forward backtest…"):
                st.session_state['bt_rows']     = run_backtest_date(bt, ma, nsr, bt_date.strftime('%Y-%m-%d'))
                st.session_state['bt_date_used'] = bt_date

        rows_bt   = st.session_state.get('bt_rows', [])
        date_used = st.session_state.get('bt_date_used', bt_date)
        dow_lbl   = DOW_NAMES[date_used.weekday()]

        if rows_bt:
            # Summary stats
            has_actual = [r for r in rows_bt if r['actual_op'] is not None]
            a_hits = sum(1 for r in has_actual if r['hit_a_op'] or r['hit_a_cp'])
            b_hits = sum(1 for r in has_actual if r['hit_b_op'] or r['hit_b_cp'])
            n_act  = len(has_actual)

            st.markdown(f"#### Results for {date_used}  ({dow_lbl})  — {len(rows_bt)} markets")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Markets run",    len(rows_bt))
            m2.metric("With actual data", n_act)
            m3.metric("Model A hits",   f"{a_hits}/{n_act}" if n_act else "—")
            m4.metric("Model B hits",   f"{b_hits}/{n_act}" if n_act else "—")

            # Table
            tbl = []
            for r in rows_bt:
                aop = r['actual_op'] if r['actual_op'] is not None else "—"
                acp = r['actual_cp'] if r['actual_cp'] is not None else "—"
                tbl.append({
                    'Market'  : f"{_stag(r['market'])} {r['market']}",
                    'Actual O': aop,
                    'Actual C': acp,
                    'A — Top4 Open' : _d4(r['a_top4_op']),
                    'A — Top4 Close': _d4(r['a_top4_cp']),
                    'A Open ✓' : _hit_icon(r['hit_a_op']),
                    'A Close ✓': _hit_icon(r['hit_a_cp']),
                    'B — Top4 Open' : _d4(r['b_top4_op']),
                    'B — Top4 Close': _d4(r['b_top4_cp']),
                    'B Open ✓' : _hit_icon(r['hit_b_op']),
                    'B Close ✓': _hit_icon(r['hit_b_cp']),
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

            # Per-market detail
            st.markdown("#### Market Detail")
            det_cols = st.columns(2)
            for i, r in enumerate(rows_bt):
                with det_cols[i % 2]:
                    aop = r['actual_op'] if r['actual_op'] is not None else '?'
                    acp = r['actual_cp'] if r['actual_cp'] is not None else '?'
                    hit_a = (r['hit_a_op'] or r['hit_a_cp']) if r['actual_op'] is not None else None
                    hit_b = (r['hit_b_op'] or r['hit_b_cp']) if r['actual_op'] is not None else None
                    border = ("#34a853" if hit_a else "#ea4335") if hit_a is not None else "#555"
                    st.markdown(
                        f'<div style="background:#1e2130;border-left:4px solid {border};'
                        f'border-radius:8px;padding:10px 14px;margin:4px 0">'
                        f'{_stag(r["market"])} <b>{r["market"]}</b>'
                        f'<span style="float:right;color:#aaa">Actual: <b>{aop}</b> / <b>{acp}</b></span><br>'
                        f'<small>A: {_d4(r["a_top4_op"])} | {_d4(r["a_top4_cp"])}'
                        f'  {_hit_icon(r["hit_a_op"])}/{_hit_icon(r["hit_a_cp"])}</small><br>'
                        f'<small>B: {_d4(r["b_top4_op"])} | {_d4(r["b_top4_cp"])}'
                        f'  {_hit_icon(r["hit_b_op"])}/{_hit_icon(r["hit_b_cp"])}</small></div>',
                        unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # TAB 3 — PERFORMANCE
    # ════════════════════════════════════════════════════════════
    with tab_perf:
        stats  = compute_all_stats(bt)

        win           = stats['win']
        monthly       = stats['monthly']
        months_sorted = stats['months_sorted']
        dow_hits      = stats['dow_hits']
        oc            = stats['oc']
        zero          = stats['zero']

        sub_overview, sub_traj, sub_monthly, sub_dow, sub_oc, sub_zero = st.tabs([
            "📊 Overview",
            "📈 Trajectory",
            "📅 Monthly",
            "🔥 DOW Heatmap",
            "🔵🟠 Open / Close",
            "❌ Zero Sessions",
        ])

        # ── OVERVIEW ──────────────────────────────────────────
        with sub_overview:
            st.markdown("### Market Performance Overview")
            st.caption("Hit rate = actual result digit appeared in Top-4 predicted digits (baseline: ~40%)")

            w_rows = []
            for mkt in ALL_MKT:
                if not any(mkt in win[k] for k in win): continue
                trend = _trend_arrow(monthly, mkt, months_sorted)
                w_rows.append({
                    'Market'   : f"{_stag(mkt)} {mkt}",
                    'Trend'    : trend,
                    '30 days'  : f"{win['30d'].get(mkt,0):.1f}%",
                    '60 days'  : f"{win['60d'].get(mkt,0):.1f}%",
                    '90 days'  : f"{win['90d'].get(mkt,0):.1f}%",
                    '180 days' : f"{win['180d'].get(mkt,0):.1f}%",
                })
            w_rows.sort(key=lambda r: -float(r['30 days'].rstrip('%')))
            st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

            # Bar chart — 30d
            bar_data = pd.DataFrame(
                {'30d %': [win['30d'].get(m, 0) for m in ALL_MKT if m in win['30d']]},
                index=[m for m in ALL_MKT if m in win['30d']]
            )
            st.bar_chart(bar_data, height=280)

            # Day vs Night
            day_hr  = [win['30d'].get(m,0) for m in ALL_MKT if m in DAY_MKT   and m in win['30d']]
            ngt_hr  = [win['30d'].get(m,0) for m in ALL_MKT if m not in DAY_MKT and m in win['30d']]
            c1, c2  = st.columns(2)
            c1.metric("☀️ Day avg (30d)",   f"{sum(day_hr)/len(day_hr):.1f}%" if day_hr else "—")
            c2.metric("🌙 Night avg (30d)", f"{sum(ngt_hr)/len(ngt_hr):.1f}%" if ngt_hr else "—")

        # ── TRAJECTORY ────────────────────────────────────────
        with sub_traj:
            st.markdown("### Monthly Hit Rate Trajectory")
            if len(months_sorted) >= 2:
                chart_data = {}
                for mkt in ALL_MKT:
                    series = []
                    for ym in months_sorted:
                        d = monthly[ym].get(mkt, {})
                        t = d.get('t', 0)
                        series.append(d.get('h', 0) / t * 100 if t else None)
                    if any(v is not None for v in series):
                        chart_data[mkt] = series
                if chart_data:
                    df_chart = (
                        pd.DataFrame(chart_data, index=months_sorted)
                        .ffill()       # ← fixed (was fillna(method='ffill'))
                        .bfill()
                    )
                    st.line_chart(df_chart, height=320)

            # Improving / declining
            impr = []; decl = []
            for mkt in ALL_MKT:
                if mkt not in win['90d']: continue
                a = _trend_arrow(monthly, mkt, months_sorted)
                if a == '↑': impr.append(mkt)
                elif a == '↓': decl.append(mkt)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**↑ Improving**")
                for m in impr:
                    r30 = win['30d'].get(m, 0); r90 = win['90d'].get(m, 0)
                    st.markdown(f"✅ {_stag(m)} **{m}** — 30d: {r30:.1f}% / 90d: {r90:.1f}%")
                if not impr: st.caption("None")
            with c2:
                st.markdown("**↓ Declining**")
                for m in decl:
                    r30 = win['30d'].get(m, 0); r90 = win['90d'].get(m, 0)
                    st.markdown(f"⚠️ {_stag(m)} **{m}** — 30d: {r30:.1f}% / 90d: {r90:.1f}%")
                if not decl: st.caption("None")

        # ── MONTHLY BREAKDOWN ─────────────────────────────────
        with sub_monthly:
            st.markdown("### Monthly Hit Rates per Market")
            if months_sorted:
                tbl_m = []
                for ym in months_sorted[-12:]:
                    row = {'Month': ym}
                    for mkt in ALL_MKT:
                        d   = monthly[ym].get(mkt, {})
                        tot = d.get('t', 0)
                        row[mkt] = f"{d.get('h',0)/tot*100:.0f}%" if tot else "—"
                    tbl_m.append(row)
                st.dataframe(pd.DataFrame(tbl_m), use_container_width=True, hide_index=True)

        # ── DOW HEATMAP ───────────────────────────────────────
        with sub_dow:
            st.markdown("### Hit Rate by Day of Week")
            st.caption("All markets combined — last 90 days")

            dow_rows = []
            for dw in range(7):
                h = dow_hits.get(dw, {}).get('h', 0) if isinstance(dow_hits, dict) else dow_hits[dw]['h']
                t = dow_hits.get(dw, {}).get('t', 1) if isinstance(dow_hits, dict) else dow_hits[dw]['t']
                if t > 0:
                    dow_rows.append({'Day': DOW_NAMES[dw],
                                     'Hit Rate %': round(h/t*100, 1),
                                     'Sessions':   t})
            if dow_rows:
                df_dow = pd.DataFrame(dow_rows)
                st.bar_chart(df_dow.set_index('Day')[['Hit Rate %']], height=260)
                st.dataframe(df_dow, use_container_width=True, hide_index=True)

            # Per-market DOW breakdown
            st.markdown("#### Per-Market by Day of Week")
            dow_mkt = defaultdict(lambda: defaultdict(lambda: {'h': 0, 't': 0}))
            for mkt in ALL_MKT:
                for r in bt.get(mkt, []):
                    dw = datetime.strptime(r['date'], '%Y-%m-%d').weekday()
                    dow_mkt[mkt][dw]['h'] += r.get('hit4', 0)
                    dow_mkt[mkt][dw]['t'] += 1
            tbl_dw = []
            for mkt in ALL_MKT:
                if mkt not in dow_mkt: continue
                row = {'Market': f"{_stag(mkt)} {mkt}"}
                for dw in range(7):
                    dd = dow_mkt[mkt][dw]
                    t  = dd['t']
                    row[DOW_NAMES[dw]] = f"{dd['h']/t*100:.0f}%" if t >= 3 else "—"
                tbl_dw.append(row)
            st.dataframe(pd.DataFrame(tbl_dw), use_container_width=True, hide_index=True)

        # ── OPEN / CLOSE ──────────────────────────────────────
        with sub_oc:
            st.markdown("### Open vs Close Accuracy (All-Time)")
            st.caption("How often the actual Open digit and actual Close digit each appeared in the Top-4 prediction")

            if oc:
                tbl_oc = []
                for mkt in ALL_MKT:
                    if mkt not in oc: continue
                    tbl_oc.append({
                        'Market'      : f"{_stag(mkt)} {mkt}",
                        'Open Hit %'  : f"{oc[mkt]['op']:.1f}%",
                        'Close Hit %' : f"{oc[mkt]['cp']:.1f}%",
                        'Δ (O − C)'   : f"{oc[mkt]['op'] - oc[mkt]['cp']:+.1f}pp",
                        'Sessions'    : oc[mkt]['n'],
                    })
                st.dataframe(pd.DataFrame(tbl_oc), use_container_width=True, hide_index=True)

                # Side-by-side bar chart
                mkts_oc = [m for m in ALL_MKT if m in oc]
                df_oc   = pd.DataFrame({
                    'Open %' : [oc[m]['op'] for m in mkts_oc],
                    'Close %': [oc[m]['cp'] for m in mkts_oc],
                }, index=mkts_oc)
                st.bar_chart(df_oc, height=280)

        # ── ZERO SESSIONS ─────────────────────────────────────
        with sub_zero:
            st.markdown("### Zero-Hit Sessions")
            st.caption("Sessions where the actual digit did NOT appear in the Top-4 prediction")
            if zero:
                tbl_z = []
                for mkt in ALL_MKT:
                    if mkt not in zero: continue
                    tbl_z.append({
                        'Market'       : f"{_stag(mkt)} {mkt}",
                        'Zero sessions': zero[mkt]['zero'],
                        'Total'        : zero[mkt]['total'],
                        'Miss rate'    : f"{zero[mkt]['pct']:.1f}%",
                        'Hit rate'     : f"{100-zero[mkt]['pct']:.1f}%",
                    })
                tbl_z.sort(key=lambda r: -float(r['Miss rate'].rstrip('%')))
                st.dataframe(pd.DataFrame(tbl_z), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════
    # TAB 4 — JODI ANALYSIS
    # ════════════════════════════════════════════════════════════
    with tab_jodi:
        st.markdown("### 🎲 Jodi Analysis — Last 90 Days")
        jodi_stats = compute_jodi_stats(bt, days=90)
        dig_freq   = compute_digit_freq(bt, days=30)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔥 Hot Jodis** (most frequent)")
            hot_rows = []
            for mkt in ALL_MKT:
                if mkt not in jodi_stats: continue
                for jd, cnt in jodi_stats[mkt]['hot'][:3]:
                    hot_rows.append({'Market': f"{_stag(mkt)} {mkt}",
                                     'Jodi': jd, 'Count': cnt,
                                     'Rate': f"{cnt/jodi_stats[mkt]['total']*100:.1f}%"})
            hot_rows.sort(key=lambda r: -r['Count'])
            if hot_rows:
                st.dataframe(pd.DataFrame(hot_rows[:20]),
                             use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**❄️ Cold Jodis** (least frequent / overdue)")
            cold_rows = []
            for mkt in ALL_MKT:
                if mkt not in jodi_stats: continue
                for jd, cnt in jodi_stats[mkt]['cold'][:3]:
                    cold_rows.append({'Market': f"{_stag(mkt)} {mkt}",
                                      'Jodi': jd, 'Count': cnt,
                                      'Rate': f"{cnt/jodi_stats[mkt]['total']*100:.1f}%"})
            cold_rows.sort(key=lambda r: r['Count'])
            if cold_rows:
                st.dataframe(pd.DataFrame(cold_rows[:20]),
                             use_container_width=True, hide_index=True)

        # Digit frequency heatmap
        st.markdown("#### Open Digit Frequency — Last 30 Days")
        st.caption("% of sessions each digit 0–9 appeared as Open result")
        if dig_freq:
            heat = pd.DataFrame(
                {mkt: [round(dig_freq[mkt]['op'][d]*100, 1) for d in range(10)]
                 for mkt in ALL_MKT if mkt in dig_freq},
                index=list(range(10))
            )
            heat.index.name = 'Digit'
            st.dataframe(
                heat.style.background_gradient(cmap='YlOrRd', axis=None),
                use_container_width=True,
            )

        # Per-market expanders
        st.markdown("#### Per-Market Jodi Detail")
        col2 = st.columns(2)
        for i, mkt in enumerate(ALL_MKT):
            if mkt not in jodi_stats: continue
            with col2[i % 2]:
                with st.expander(f"{_stag(mkt)} {mkt}  ({jodi_stats[mkt]['total']} sessions)"):
                    hot_j  = "  ".join(f"**{j[0]}**({j[1]})" for j in jodi_stats[mkt]['hot'][:5])
                    cold_j = "  ".join(f"**{j[0]}**({j[1]})" for j in jodi_stats[mkt]['cold'][:5])
                    st.markdown(f"🔥 Hot: {hot_j}")
                    st.markdown(f"❄️ Cold: {cold_j}")
                    if mkt in dig_freq:
                        df_d = pd.DataFrame({
                            'Open %' : [round(x*100,1) for x in dig_freq[mkt]['op']],
                            'Close %': [round(x*100,1) for x in dig_freq[mkt]['cp']],
                        }, index=range(10))
                        df_d.index.name = 'Digit'
                        st.bar_chart(df_d, height=160)

    # ════════════════════════════════════════════════════════════
    # TAB 5 — DATA UPDATE
    # ════════════════════════════════════════════════════════════
    with tab_upd:
        st.markdown("### 📥 Keeping Data Up to Date")
        st.markdown("""
The prediction model uses historical results stored in `data/bt_data.json`.
This file is updated periodically by the data administrator.

**How the update works:**

1. The data admin exports the latest results to an Excel file
2. The `.json` file in the GitHub repository is replaced with the new version
3. Streamlit Cloud auto-deploys within 1–2 minutes

**Using the sidebar upload (immediate, session-only):**
- Use the Excel uploader in the sidebar to add new results to the current session
- Columns needed: `Date` | `Market` | `Open` | `Close`
- Open/Close can be a 3-digit panna (e.g. `168`) or a single digit (0–9)
- Changes last only for your current browser session — they are NOT permanently saved
- For permanent updates, the GitHub file must be replaced

**Current data coverage:**
""")
        # Show data coverage
        cov_rows = []
        for mkt in ALL_MKT:
            rows = bt.get(mkt, [])
            if not rows: continue
            cov_rows.append({
                'Market'   : f"{_stag(mkt)} {mkt}",
                'First date': rows[0]['date'],
                'Last date' : rows[-1]['date'],
                'Records'  : len(rows),
            })
        if cov_rows:
            st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)

        st.info(
            "📌 **For the administrator:** Replace `data/bt_data.json` in the GitHub repo weekly "
            "with the latest export. The app will pick it up automatically within 2 minutes of the push."
        )

    # ── FOOTER ──────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Matka Predict Pro v7 · {tstr} ({DOW_NAMES[tdow]}) · "
        f"{len(mkt_list)} markets active · 35-algorithm engine · Confidential"
    )


if __name__ == '__main__':
    main()
