"""
Matka Predict Pro v7 — Client Dashboard
Full prediction dashboard with date selection, Excel data upload,
Current model + NX model predictions, Top-5 / Bottom-3 markets.
"""

import streamlit as st
import json, os, io
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ALL_MKT = [
    'Sridevi','TimeBazar','MadhurDay','MilanDay','RajdhaniDay','Kalyan',
    'SrideviNight','MadhurNight','SupremeNight','MilanNight',
    'RajdhaniNight','KalyanNight','MainBazar'
]
SAT_CLOSED = {'RajdhaniNight','KalyanNight','MainBazar'}
SUN_OPEN   = {'Sridevi','MadhurDay','SrideviNight','SupremeNight'}
DOW_NAMES  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')

# ─────────────────────────────────────────────
# STRATEGY ENGINE
# ─────────────────────────────────────────────
def norm(x):
    s = x.sum()
    return x / s if s > 0 else np.ones(10) / 10

def run_strat(name, ops, cps, dows, tdow, n):
    """Full 35-strategy library. Returns (op_probs[10], cp_probs[10])."""
    def ew(dc=15, W=90):
        Ww = min(n, W)
        e = np.exp(-np.arange(Ww, 0, -1) / dc); e /= e.sum()
        return (
            np.array([(e * (ops[n-Ww:n] == d)).sum() for d in range(10)]),
            np.array([(e * (cps[n-Ww:n] == d)).sum() for d in range(10)])
        )
    def df(last=20):
        s = np.where(dows[:n] == tdow)[0][-last:]
        if len(s) >= 5:
            return (
                np.array([(ops[s] == d).mean() for d in range(10)]),
                np.array([(cps[s] == d).mean() for d in range(10)])
            )
        return None, None

    W3=min(n,3); W7=min(n,7); W14=min(n,14); W30=min(n,30)
    W60=min(n,60); W90=min(n,90); W120=min(n,120); W180=min(n,180)
    ol, cl = ew(15)

    if name == 'Hot-Cold':
        return (norm((np.array([(ops[n-W7:n]==d).mean() for d in range(10)])+.01)/(ol+.01)),
                norm((np.array([(cps[n-W7:n]==d).mean() for d in range(10)])+.01)/(cl+.01)))
    elif name in ('Mean-Revert-30','Mean-Revert-60','Mean-Revert-120'):
        W={'Mean-Revert-30':W30,'Mean-Revert-60':W60,'Mean-Revert-120':W120}[name]
        return (norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)),
                norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Recent-30':
        return (norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])),
                norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])))
    elif name == 'GeoAntiF-DOW':
        og=np.array([(ops[n-W90:n]==d).mean() for d in range(10)])
        cg=np.array([(cps[n-W90:n]==d).mean() for d in range(10)])
        od,cd=df()
        return (norm(1/((og+.01)*(od+.01))) if od is not None else norm(1/(og+.01)),
                norm(1/((cg+.01)*(cd+.01))) if cd is not None else norm(1/(cg+.01)))
    elif name == 'Anti-Freq-DOW':
        od,cd=df()
        return (norm(1/(od+.01)) if od is not None else norm(1/(np.array([(ops[n-W90:n]==d).mean() for d in range(10)])+.01)),
                norm(1/(cd+.01)) if cd is not None else norm(1/(np.array([(cps[n-W90:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Anti-Freq-DOW-50':
        od50,cd50=df(50)
        return (norm(1/(od50+.01)) if od50 is not None else norm(1/(np.array([(ops[n-W90:n]==d).mean() for d in range(10)])+.01)),
                norm(1/(cd50+.01)) if cd50 is not None else norm(1/(np.array([(cps[n-W90:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Decay-1':
        o,c=ew(1,30); return norm(o),norm(c)
    elif name == 'Decay-3':
        o,c=ew(3,60); return norm(o),norm(c)
    elif name == 'Decay-5':
        o,c=ew(5,90); return norm(o),norm(c)
    elif name == 'Decay5+AntiF':
        od,cd=ew(5,W90)
        return norm(.5*norm(od)+.5*norm(1/(od+.01))),norm(.5*norm(cd)+.5*norm(1/(cd+.01)))
    elif name == 'Balanced':
        oaf=norm(1/(ol+.01)); caf=norm(1/(cl+.01))
        o14=np.array([(ops[n-W14:n]==d).mean() for d in range(10)])
        c14=np.array([(cps[n-W14:n]==d).mean() for d in range(10)])
        od,cd=df(); od_b=od if od is not None else ol; cd_b=cd if cd is not None else cl
        return norm(.4*oaf+.35*o14+.25*od_b),norm(.4*caf+.35*c14+.25*cd_b)
    elif name == 'DOW-Only':
        sdow=np.where(dows[:n]==tdow)[0]
        if len(sdow)>=8:
            idx=sdow[-min(len(sdow),30):]
            e=np.exp(-np.arange(len(idx),0,-1)/10); e/=e.sum()
            return (norm(1/(np.array([(e*(ops[idx]==d)).sum() for d in range(10)])+.01)),
                    norm(1/(np.array([(e*(cps[idx]==d)).sum() for d in range(10)])+.01)))
        return (norm(1/(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01)),
                norm(1/(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Markov-1':
        if n>5:
            M=np.ones((10,10))*.1; M2=np.ones((10,10))*.1
            for k in range(1,n): M[ops[k-1],ops[k]]+=1; M2[cps[k-1],cps[k]]+=1
            M/=M.sum(axis=1,keepdims=True); M2/=M2.sum(axis=1,keepdims=True)
            return norm(M[ops[n-1]]),norm(M2[cps[n-1]])
        return norm(np.ones(10)),norm(np.ones(10))
    elif name == 'Markov-2':
        if n>6:
            M3=np.ones((10,10,10))*.05; M4=np.ones((10,10,10))*.05
            for k in range(2,n): M3[ops[k-2],ops[k-1],ops[k]]+=1; M4[cps[k-2],cps[k-1],cps[k]]+=1
            return norm(M3[ops[n-2],ops[n-1]]),norm(M4[cps[n-2],cps[n-1]])
        return norm(np.ones(10)),norm(np.ones(10))
    elif name == 'Extreme-Anti':
        o=np.array([(ops[n-W60:n]==d).mean()+.01 for d in range(10)])
        c=np.array([(cps[n-W60:n]==d).mean()+.01 for d in range(10)])
        return norm(1/o**2),norm(1/c**2)
    elif name == 'Recency-Burst':
        od10,cd10=ew(10)
        op_r=np.array([(ops[n-W3:n]==d).mean() for d in range(10)])
        cp_r=np.array([(cps[n-W3:n]==d).mean() for d in range(10)])
        return norm(.3*norm(1/(op_r+.01))+.7*norm(od10)),norm(.3*norm(1/(cp_r+.01))+.7*norm(cd10))
    elif name == 'Decay-2':
        o,c=ew(2,45); return norm(o),norm(c)
    elif name == 'Decay-7':
        o,c=ew(7,90); return norm(o),norm(c)
    elif name == 'Decay-10':
        o,c=ew(10,120); return norm(o),norm(c)
    elif name == 'Decay-20':
        o,c=ew(20,150); return norm(o),norm(c)
    elif name == 'Recent-14':
        return (norm(np.array([(ops[n-W14:n]==d).mean() for d in range(10)])),
                norm(np.array([(cps[n-W14:n]==d).mean() for d in range(10)])))
    elif name == 'Recent-60':
        return (norm(np.array([(ops[n-W60:n]==d).mean() for d in range(10)])),
                norm(np.array([(cps[n-W60:n]==d).mean() for d in range(10)])))
    elif name == 'Anti-Last3':
        op_r3=np.array([(ops[n-W3:n]==d).mean() for d in range(10)])
        cp_r3=np.array([(cps[n-W3:n]==d).mean() for d in range(10)])
        return norm(1/(op_r3+.01)),norm(1/(cp_r3+.01))
    elif name == 'DOW-Anti':
        s=np.where(dows[:n]==tdow)[0][-30:]
        if len(s)>=5:
            od=np.array([(ops[s]==d).mean() for d in range(10)]); cd=np.array([(cps[s]==d).mean() for d in range(10)])
            return norm(1/(od+.01)),norm(1/(cd+.01))
        return (norm(1/(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01)),
                norm(1/(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Skip-Recent-5':
        W5=min(n,5)
        op=np.array([(ops[n-W5:n]==d).mean() for d in range(10)]); cp=np.array([(cps[n-W5:n]==d).mean() for d in range(10)])
        return norm(1/(op+.01)),norm(1/(cp+.01))
    elif name == 'Combo-Top3':
        d3o,d3c=ew(3,60)
        hc_o=norm((np.array([(ops[n-W7:n]==d).mean() for d in range(10)])+.01)/(ol+.01))
        hc_c=norm((np.array([(cps[n-W7:n]==d).mean() for d in range(10)])+.01)/(cl+.01))
        r30o=norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])); r30c=norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)]))
        return norm(norm(d3o)/3+hc_o/3+r30o/3),norm(norm(d3c)/3+hc_c/3+r30c/3)
    elif name == 'Hybrid-60':
        af60o=norm(1/(np.array([(ops[n-W60:n]==d).mean() for d in range(10)])+.01))
        af60c=norm(1/(np.array([(cps[n-W60:n]==d).mean() for d in range(10)])+.01))
        r60o=norm(np.array([(ops[n-W60:n]==d).mean() for d in range(10)])); r60c=norm(np.array([(cps[n-W60:n]==d).mean() for d in range(10)]))
        s=np.where(dows[:n]==tdow)[0][-20:]
        dow_w=0.25 if len(s)>=5 else 0.0
        od=np.array([(ops[s]==d).mean() for d in range(10)]) if len(s)>=5 else np.ones(10)/10
        cd=np.array([(cps[s]==d).mean() for d in range(10)]) if len(s)>=5 else np.ones(10)/10
        return (norm((0.5-dow_w/2)*af60o+(0.25-dow_w/4)*r60o+(0.25-dow_w/4)*ol+dow_w*norm(1/(od+.01))),
                norm((0.5-dow_w/2)*af60c+(0.25-dow_w/4)*r60c+(0.25-dow_w/4)*cl+dow_w*norm(1/(cd+.01))))
    elif name == 'Markov-3':
        if n>7:
            M5=np.ones((10,10,10,10))*.02; M6=np.ones((10,10,10,10))*.02
            for k in range(3,n): M5[ops[k-3],ops[k-2],ops[k-1],ops[k]]+=1; M6[cps[k-3],cps[k-2],cps[k-1],cps[k]]+=1
            s5=M5[ops[n-3],ops[n-2],ops[n-1]]; s5/=s5.sum(); s6=M6[cps[n-3],cps[n-2],cps[n-1]]; s6/=s6.sum()
            return norm(s5),norm(s6)
        return norm(np.ones(10)),norm(np.ones(10))
    elif name == 'Mean-Revert-180':
        W=min(n,180)
        return (norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)),
                norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)))
    elif name == 'Volatility-Anti':
        W=W90; gap_o=np.zeros(10); gap_c=np.zeros(10)
        for d in range(10):
            idx_o=np.where(ops[n-W:n]==d)[0]; idx_c=np.where(cps[n-W:n]==d)[0]
            gap_o[d]=np.diff(idx_o).std()+1 if len(idx_o)>2 else 15
            gap_c[d]=np.diff(idx_c).std()+1 if len(idx_c)>2 else 15
        return norm(gap_o),norm(gap_c)
    elif name == 'Long-Anti-180':
        W=min(n,180)
        return (norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)**1.5),
                norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)**1.5))
    elif name == 'DOW-Freq':
        s=np.where(dows[:n]==tdow)[0][-40:]
        if len(s)>=5:
            od=np.array([(ops[s]==d).mean() for d in range(10)]); cd=np.array([(cps[s]==d).mean() for d in range(10)])
            return norm(od+.01),norm(cd+.01)
        return (norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01),
                norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01))
    elif name == 'Decay3+DOW':
        d3o,d3c=ew(3,60); s=np.where(dows[:n]==tdow)[0][-20:]
        if len(s)>=5:
            od=np.array([(ops[s]==d).mean() for d in range(10)]); cd=np.array([(cps[s]==d).mean() for d in range(10)])
            dow_a_o=norm(1/(od+.01)); dow_a_c=norm(1/(cd+.01))
        else:
            dow_a_o=norm(1/(ol+.01)); dow_a_c=norm(1/(cl+.01))
        return norm(.6*norm(d3o)+.4*dow_a_o),norm(.6*norm(d3c)+.4*dow_a_c)
    else:
        o,c=ew(15); return norm(o),norm(c)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading BT data…")
def load_bt_data():
    """Load bt_data.json from data/ folder."""
    path = os.path.join(DATA_DIR, 'bt_data.json')
    with open(path) as f:
        return json.load(f)

@st.cache_data(show_spinner="Loading strategy config…")
def load_configs():
    ma_path  = os.path.join(DATA_DIR, 'market_assign.json')
    nsr_path = os.path.join(DATA_DIR, 'newx_strategy_results.json')
    with open(ma_path) as f:  ma  = json.load(f)
    with open(nsr_path) as f: nsr = json.load(f)
    return ma, nsr


def get_series(bt, market, cutoff_str):
    """Extract ops/cps/dows arrays for a market, up to cutoff_str (exclusive)."""
    rows = [r for r in bt.get(market, []) if r['date'] < cutoff_str]
    if len(rows) < 15:
        return None, None, None, 0
    ops  = np.array([r['actual_op'] for r in rows], dtype=int)
    cps  = np.array([r['actual_cp'] for r in rows], dtype=int)
    dows = np.array([datetime.strptime(r['date'],'%Y-%m-%d').weekday() for r in rows], dtype=int)
    return ops, cps, dows, len(rows)


def gen_predictions(bt, ma, nsr, target_date_str, tdow):
    """Generate both model predictions for all active markets on target_date."""
    td = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    cutoff = target_date_str

    results = {}
    for mkt in ALL_MKT:
        # Skip closed markets
        if tdow == 5 and mkt in SAT_CLOSED:
            continue
        if tdow == 6 and mkt not in SUN_OPEN:
            continue

        ops, cps, dows, n = get_series(bt, mkt, cutoff)
        if ops is None:
            continue

        # ── Current model ──
        cur_strat = ma.get(mkt, {}).get('strategy', 'Hot-Cold')
        op_p, cp_p = run_strat(cur_strat, ops, cps, dows, tdow, n)
        top4_op = list(np.argsort(op_p)[::-1][:4])
        top4_cp = list(np.argsort(cp_p)[::-1][:4])
        cur_conf = float(sum(sorted(op_p)[-4:]) + sum(sorted(cp_p)[-4:])) / 2

        # ── NX model ──
        cfg = nsr.get(mkt, {})
        nx_op_strat = cfg.get('best_open', cur_strat)
        nx_cp_strat = cfg.get('best_close', cur_strat)
        nx_op_p, _  = run_strat(nx_op_strat, ops, cps, dows, tdow, n)
        _, nx_cp_p  = run_strat(nx_cp_strat, ops, cps, dows, tdow, n)
        nx_top4_op  = list(np.argsort(nx_op_p)[::-1][:4])
        nx_top4_cp  = list(np.argsort(nx_cp_p)[::-1][:4])
        nx_conf     = float(sum(sorted(nx_op_p)[-4:]) + sum(sorted(nx_cp_p)[-4:])) / 2

        # Historical hit rate (last 30 BT rows)
        last30 = bt.get(mkt, [])[-30:]
        h4_rate = sum(r.get('hit4',0) for r in last30) / max(len(last30),1)

        results[mkt] = {
            'cur_strat':  cur_strat,
            'top4_op':    top4_op,
            'top4_cp':    top4_cp,
            'op_probs':   [round(float(x),4) for x in op_p],
            'cp_probs':   [round(float(x),4) for x in cp_p],
            'cur_conf':   round(cur_conf,4),
            'nx_op_strat':nx_op_strat,
            'nx_cp_strat':nx_cp_strat,
            'nx_top4_op': nx_top4_op,
            'nx_top4_cp': nx_top4_cp,
            'nx_op_probs':[round(float(x),4) for x in nx_op_p],
            'nx_cp_probs':[round(float(x),4) for x in nx_cp_p],
            'nx_conf':    round(nx_conf,4),
            'h4_rate':    round(h4_rate,3),
        }

    return results


# ─────────────────────────────────────────────
# EXCEL PARSER  (update BT with new Excel data)
# ─────────────────────────────────────────────
def panna_to_digit(panna_str):
    """Convert panna (3-digit sum) to result digit."""
    try:
        s = str(int(str(panna_str).strip())).zfill(3)
        return (int(s[0]) + int(s[1]) + int(s[2])) % 10
    except:
        return None

def parse_excel_to_bt(uploaded_bytes, existing_bt):
    """Parse uploaded Excel → update bt dict with new rows.
    Returns (updated_bt, new_rows_count, error_msg).
    Excel should have columns: Date, Market, Open (panna/digit), Close (panna/digit).
    """
    try:
        xl = pd.ExcelFile(io.BytesIO(uploaded_bytes))
        new_count = 0
        bt = {k: list(v) for k, v in existing_bt.items()}  # copy

        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df.columns = [str(c).strip() for c in df.columns]

            # Try to find date/market/open/close columns (flexible matching)
            col_map = {}
            for col in df.columns:
                cl = col.lower()
                if 'date' in cl:          col_map['date'] = col
                elif 'market' in cl or 'mkt' in cl: col_map['market'] = col
                elif 'open' in cl or 'op' == cl:    col_map['open'] = col
                elif 'close' in cl or 'cp' == cl:   col_map['close'] = col

            # If single-market sheet (sheet name IS the market)
            sheet_market = None
            for m in ALL_MKT:
                if m.lower() in sheet.lower() or sheet.lower() in m.lower():
                    sheet_market = m
                    break

            if 'date' not in col_map:
                continue

            for _, row in df.iterrows():
                try:
                    dt_raw = row[col_map['date']]
                    if pd.isna(dt_raw): continue
                    if isinstance(dt_raw, (datetime, pd.Timestamp)):
                        dt_str = dt_raw.strftime('%Y-%m-%d')
                    else:
                        dt_str = str(dt_raw).strip()[:10]
                    # basic format check
                    datetime.strptime(dt_str, '%Y-%m-%d')

                    mkt = sheet_market
                    if 'market' in col_map:
                        mkt_raw = str(row[col_map['market']]).strip()
                        for m in ALL_MKT:
                            if m.lower() == mkt_raw.lower() or mkt_raw.lower() in m.lower():
                                mkt = m; break

                    if mkt is None or mkt not in ALL_MKT:
                        continue

                    if 'open' in col_map:
                        op_val = row[col_map['open']]
                        op_digit = None
                        if not pd.isna(op_val):
                            v = int(op_val)
                            op_digit = v % 10 if v < 10 else panna_to_digit(v)
                    else:
                        continue

                    if 'close' in col_map:
                        cp_val = row[col_map['close']]
                        cp_digit = None
                        if not pd.isna(cp_val):
                            v = int(cp_val)
                            cp_digit = v % 10 if v < 10 else panna_to_digit(v)
                    else:
                        cp_digit = None

                    if op_digit is None: continue

                    # Check if date already exists for this market
                    existing_dates = {r['date'] for r in bt.get(mkt, [])}
                    if dt_str in existing_dates:
                        continue  # skip duplicates

                    new_row = {
                        'date': dt_str,
                        'actual_op': op_digit,
                        'actual_cp': cp_digit if cp_digit is not None else op_digit,
                        'actual_jodi': f"{op_digit}{cp_digit if cp_digit is not None else op_digit}",
                        'hit4': 0, 'strategy': 'uploaded'
                    }
                    if mkt not in bt:
                        bt[mkt] = []
                    bt[mkt].append(new_row)
                    new_count += 1
                except:
                    continue

        # Sort each market's rows by date
        for mkt in bt:
            bt[mkt] = sorted(bt[mkt], key=lambda r: r['date'])

        return bt, new_count, None
    except Exception as e:
        return existing_bt, 0, str(e)


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def digit_badge(d, highlighted=False):
    color = "#1a73e8" if highlighted else "#e8eaf6"
    text_color = "white" if highlighted else "#333"
    return f'<span style="display:inline-block;width:28px;height:28px;border-radius:50%;background:{color};color:{text_color};text-align:center;line-height:28px;font-weight:bold;font-size:14px;margin:2px">{d}</span>'

def top4_html(digits, probs):
    badges = "".join(digit_badge(d, True) for d in digits[:4])
    prob_str = " | ".join(f"{d}:{probs[d]*100:.1f}%" for d in digits[:4])
    return f"{badges}<br><small style='color:#666'>{prob_str}</small>"


def market_session_tag(mkt):
    if mkt in {'Sridevi','TimeBazar','MadhurDay','MilanDay','RajdhaniDay','Kalyan'}:
        return "☀️ Day"
    return "🌙 Night"


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Matka Predict Pro v7",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
    .stApp { background: #0f1117; color: #e0e0e0; }
    .pred-card { background: #1e2130; border-radius: 12px; padding: 16px 20px; margin: 8px 0;
                 border-left: 4px solid #1a73e8; }
    .pred-card-nx { border-left-color: #34a853; }
    .mkt-name { font-size: 1.1em; font-weight: 700; color: #fff; }
    .strat-tag { font-size: 0.75em; background: #2d3250; color: #9aa0c0; padding: 2px 8px;
                 border-radius: 4px; margin-left: 8px; }
    .top5-card { background: linear-gradient(135deg, #1a3a1a, #1e2130); border: 1px solid #34a853;
                 border-radius: 10px; padding: 12px 16px; margin: 6px 0; }
    .bot3-card { background: linear-gradient(135deg, #3a1a1a, #1e2130); border: 1px solid #ea4335;
                 border-radius: 10px; padding: 12px 16px; margin: 6px 0; }
    .metric-box { background: #1e2130; border-radius: 8px; padding: 12px; text-align: center; }
    .metric-val { font-size: 2em; font-weight: 800; color: #1a73e8; }
    .metric-lbl { font-size: 0.8em; color: #9aa0c0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("# 🎯 Matka Predict Pro v7")
    st.markdown("---")

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        today = date.today()
        target = st.date_input("📅 Target Date", value=today,
                                min_value=date(2025,1,1), max_value=today + timedelta(days=7))
        tdow = target.weekday()
        target_str = target.strftime('%Y-%m-%d')
        dow_name = DOW_NAMES[tdow]

        st.markdown(f"**Day:** {dow_name}")
        if tdow == 5:
            st.warning("Saturday — 3 markets closed (RajdhaniNight, KalyanNight, MainBazar)")
        elif tdow == 6:
            st.info("Sunday — Only 4 markets open")

        st.markdown("---")
        st.markdown("### 📊 Update Data")
        st.caption("Upload your Excel file to add new actuals and refresh predictions")
        uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"],
                                     help="File with columns: Date, Market, Open, Close")

        st.markdown("---")
        model_view = st.radio("🔭 Model View", ["Both", "Current Only", "NX Only"], index=0)

        st.markdown("---")
        st.caption("© Matka Predict Pro v7 · Confidential")

    # ── Load data ──
    bt    = load_bt_data()
    ma, nsr = load_configs()

    # Handle uploaded Excel
    upload_msg = None
    if uploaded is not None:
        new_bt, cnt, err = parse_excel_to_bt(uploaded.read(), bt)
        if err:
            st.error(f"⚠️ Upload error: {err}")
        else:
            bt = new_bt
            if cnt > 0:
                upload_msg = f"✅ Added {cnt} new data rows from uploaded file."
            else:
                upload_msg = "ℹ️ No new rows found (all dates already exist)."
            # Clear prediction cache since data changed
            st.cache_data.clear()

    if upload_msg:
        st.success(upload_msg)

    # ── Generate predictions ──
    with st.spinner(f"Generating predictions for {target_str} ({dow_name})…"):
        preds = gen_predictions(bt, ma, nsr, target_str, tdow)

    if not preds:
        st.error("No markets active for this date/day combination.")
        return

    n_markets = len(preds)
    mkt_list = list(preds.keys())

    # ── Summary row ──
    day_mkts   = [m for m in mkt_list if m in {'Sridevi','TimeBazar','MadhurDay','MilanDay','RajdhaniDay','Kalyan'}]
    night_mkts = [m for m in mkt_list if m not in day_mkts]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{n_markets}</div>'
                    f'<div class="metric-lbl">Active Markets</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(day_mkts)}</div>'
                    f'<div class="metric-lbl">☀️ Day Sessions</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(night_mkts)}</div>'
                    f'<div class="metric-lbl">🌙 Night Sessions</div></div>', unsafe_allow_html=True)
    with col4:
        avg_h4 = sum(preds[m]['h4_rate'] for m in mkt_list) / n_markets
        expected_hits = round(avg_h4 * n_markets)
        st.markdown(f'<div class="metric-box"><div class="metric-val">{expected_hits}</div>'
                    f'<div class="metric-lbl">Expected Hits (~{avg_h4*100:.1f}%)</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TOP 5 / BOTTOM 3 ──
    ranked = sorted(mkt_list, key=lambda m: preds[m]['h4_rate'], reverse=True)
    top5   = ranked[:5]
    bot3   = ranked[-3:]

    col_t, col_b = st.columns(2)

    with col_t:
        st.markdown("### 🏆 Top 5 — Most Reliable Markets")
        for i, mkt in enumerate(top5, 1):
            p = preds[mkt]
            hr = p['h4_rate']
            o4 = "  ".join(str(d) for d in p['top4_op'])
            c4 = "  ".join(str(d) for d in p['top4_cp'])
            st.markdown(
                f'<div class="top5-card">'
                f'<b style="color:#34a853">#{i}</b> '
                f'<span class="mkt-name">{mkt}</span>'
                f'<span class="strat-tag">{market_session_tag(mkt)}</span>'
                f'<span class="strat-tag">{hr*100:.1f}% hit rate</span><br>'
                f'<small>Open: <b>{o4}</b> &nbsp;&nbsp; Close: <b>{c4}</b></small>'
                f'</div>',
                unsafe_allow_html=True
            )

    with col_b:
        st.markdown("### ⚠️ Bottom 3 — Least Reliable Markets")
        for i, mkt in enumerate(bot3, 1):
            p = preds[mkt]
            hr = p['h4_rate']
            o4 = "  ".join(str(d) for d in p['top4_op'])
            c4 = "  ".join(str(d) for d in p['top4_cp'])
            st.markdown(
                f'<div class="bot3-card">'
                f'<b style="color:#ea4335">⚠{i}</b> '
                f'<span class="mkt-name">{mkt}</span>'
                f'<span class="strat-tag">{market_session_tag(mkt)}</span>'
                f'<span class="strat-tag">{hr*100:.1f}% hit rate</span><br>'
                f'<small>Open: <b>{o4}</b> &nbsp;&nbsp; Close: <b>{c4}</b></small>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Full Predictions Table ──
    st.markdown(f"### 📋 All Markets — {target_str} ({dow_name})")

    tab_cur, tab_nx, tab_cmp = st.tabs(["Current Model", "NX Model", "Side-by-Side"])

    # ── Tab 1: Current Model ──
    with tab_cur:
        st.caption("Current model: single strategy per market, jodi-optimised")
        rows_cur = []
        for mkt in mkt_list:
            p = preds[mkt]
            rows_cur.append({
                'Market':     mkt,
                'Session':    market_session_tag(mkt),
                'Strategy':   p['cur_strat'],
                'Open Top-4': "  ".join(str(d) for d in p['top4_op']),
                'Close Top-4':"  ".join(str(d) for d in p['top4_cp']),
                'Open Probs': "  ".join(f"{d}:{p['op_probs'][d]*100:.1f}%" for d in p['top4_op']),
                'Close Probs':"  ".join(f"{d}:{p['cp_probs'][d]*100:.1f}%" for d in p['top4_cp']),
                'Hit Rate %': f"{p['h4_rate']*100:.1f}%",
            })
        df_cur = pd.DataFrame(rows_cur)
        st.dataframe(df_cur, use_container_width=True, hide_index=True)

        st.markdown("#### 🃏 Detailed Cards")
        cols = st.columns(2)
        for i, mkt in enumerate(mkt_list):
            p = preds[mkt]
            with cols[i % 2]:
                o4 = p['top4_op']; c4 = p['top4_cp']
                op_bars = "  ".join(f"**{d}** ({p['op_probs'][d]*100:.1f}%)" for d in o4)
                cp_bars = "  ".join(f"**{d}** ({p['cp_probs'][d]*100:.1f}%)" for d in c4)
                with st.expander(f"{market_session_tag(mkt)} {mkt}  |  {p['h4_rate']*100:.1f}% hit"):
                    st.markdown(f"**Strategy:** `{p['cur_strat']}`")
                    st.markdown(f"**Open Top-4:** {op_bars}")
                    st.markdown(f"**Close Top-4:** {cp_bars}")
                    # Mini probability chart
                    chart_data = pd.DataFrame({
                        'Open':  p['op_probs'],
                        'Close': p['cp_probs'],
                    }, index=list(range(10)))
                    st.bar_chart(chart_data, height=180)

    # ── Tab 2: NX Model ──
    with tab_nx:
        st.caption("NX model: separate best strategies for OPEN and CLOSE — higher accuracy")
        rows_nx = []
        for mkt in mkt_list:
            p = preds[mkt]
            rows_nx.append({
                'Market':       mkt,
                'Session':      market_session_tag(mkt),
                'Open Strategy':p['nx_op_strat'],
                'Close Strategy':p['nx_cp_strat'],
                'Open Top-4':   "  ".join(str(d) for d in p['nx_top4_op']),
                'Close Top-4':  "  ".join(str(d) for d in p['nx_top4_cp']),
                'Open Probs':   "  ".join(f"{d}:{p['nx_op_probs'][d]*100:.1f}%" for d in p['nx_top4_op']),
                'Close Probs':  "  ".join(f"{d}:{p['nx_cp_probs'][d]*100:.1f}%" for d in p['nx_top4_cp']),
                'Hit Rate %':   f"{p['h4_rate']*100:.1f}%",
            })
        df_nx = pd.DataFrame(rows_nx)
        st.dataframe(df_nx, use_container_width=True, hide_index=True)

        st.markdown("#### 🃏 Detailed Cards")
        cols2 = st.columns(2)
        for i, mkt in enumerate(mkt_list):
            p = preds[mkt]
            with cols2[i % 2]:
                o4 = p['nx_top4_op']; c4 = p['nx_top4_cp']
                op_bars = "  ".join(f"**{d}** ({p['nx_op_probs'][d]*100:.1f}%)" for d in o4)
                cp_bars = "  ".join(f"**{d}** ({p['nx_cp_probs'][d]*100:.1f}%)" for d in c4)
                with st.expander(f"{market_session_tag(mkt)} {mkt}  |  NX"):
                    st.markdown(f"**Open Strategy:** `{p['nx_op_strat']}`")
                    st.markdown(f"**Close Strategy:** `{p['nx_cp_strat']}`")
                    st.markdown(f"**Open Top-4:** {op_bars}")
                    st.markdown(f"**Close Top-4:** {cp_bars}")
                    chart_data2 = pd.DataFrame({
                        'Open':  p['nx_op_probs'],
                        'Close': p['nx_cp_probs'],
                    }, index=list(range(10)))
                    st.bar_chart(chart_data2, height=180)

    # ── Tab 3: Side-by-Side compare ──
    with tab_cmp:
        st.caption("Compare Current vs NX predictions per market")
        for mkt in mkt_list:
            p = preds[mkt]
            with st.expander(f"{market_session_tag(mkt)}  {mkt}  —  Hit Rate: {p['h4_rate']*100:.1f}%"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Current Model**")
                    st.markdown(f"Strategy: `{p['cur_strat']}`")
                    st.markdown(f"Open: **{'  '.join(str(d) for d in p['top4_op'])}**")
                    st.markdown(f"Close: **{'  '.join(str(d) for d in p['top4_cp'])}**")
                with c2:
                    st.markdown("**NX Model**")
                    st.markdown(f"Open strat: `{p['nx_op_strat']}`")
                    st.markdown(f"Close strat: `{p['nx_cp_strat']}`")
                    st.markdown(f"Open: **{'  '.join(str(d) for d in p['nx_top4_op'])}**")
                    st.markdown(f"Close: **{'  '.join(str(d) for d in p['nx_top4_cp'])}**")

                # Overlap check
                cur_jodis = {f"{o}{c}" for o in p['top4_op'] for c in p['top4_cp']}
                nx_jodis  = {f"{o}{c}" for o in p['nx_top4_op'] for c in p['nx_top4_cp']}
                overlap   = cur_jodis & nx_jodis
                st.markdown(f"**Jodi overlap (both models agree):** "
                            f"{len(overlap)}/16 jodis → {', '.join(sorted(overlap)[:8])}"
                            f"{'…' if len(overlap) > 8 else ''}")

    # ── Footer ──
    st.markdown("---")
    st.caption(f"Matka Predict Pro v7 · Generated for {target_str} ({dow_name}) · "
               f"{n_markets} active markets · 35-strategy engine")


if __name__ == '__main__':
    main()
