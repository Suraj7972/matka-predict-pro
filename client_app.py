"""
Matka Predict Pro v7 — Client Dashboard
Full prediction dashboard: date selection, Excel upload, both model predictions,
Top-5/Bottom-3, Backtest performance, Monthly trends, Jodi analysis.
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
# STRATEGY ENGINE  (all 35 strategies — names hidden from UI)
# ─────────────────────────────────────────────────────────────
def _norm(x):
    s = x.sum(); return x / s if s > 0 else np.ones(10) / 10

def _run_strat(name, ops, cps, dows, tdow, n):
    def ew(dc=15, W=90):
        Ww=min(n,W); e=np.exp(-np.arange(Ww,0,-1)/dc); e/=e.sum()
        return (np.array([(e*(ops[n-Ww:n]==d)).sum() for d in range(10)]),
                np.array([(e*(cps[n-Ww:n]==d)).sum() for d in range(10)]))
    def df(last=20):
        s=np.where(dows[:n]==tdow)[0][-last:]
        if len(s)>=5: return (np.array([(ops[s]==d).mean() for d in range(10)]),
                              np.array([(cps[s]==d).mean() for d in range(10)]))
        return None,None
    W3=min(n,3);W7=min(n,7);W14=min(n,14);W30=min(n,30)
    W60=min(n,60);W90=min(n,90);W120=min(n,120);W180=min(n,180)
    ol,cl=ew(15)
    if name=='Hot-Cold':
        return (_norm((np.array([(ops[n-W7:n]==d).mean() for d in range(10)])+.01)/(ol+.01)),
                _norm((np.array([(cps[n-W7:n]==d).mean() for d in range(10)])+.01)/(cl+.01)))
    elif name in ('Mean-Revert-30','Mean-Revert-60','Mean-Revert-120'):
        W={'Mean-Revert-30':W30,'Mean-Revert-60':W60,'Mean-Revert-120':W120}[name]
        return (_norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)))
    elif name=='Recent-30':
        return (_norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])),
                _norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])))
    elif name=='GeoAntiF-DOW':
        og=np.array([(ops[n-W90:n]==d).mean() for d in range(10)]); cg=np.array([(cps[n-W90:n]==d).mean() for d in range(10)])
        od,cd=df()
        return (_norm(1/((og+.01)*(od+.01))) if od is not None else _norm(1/(og+.01)),
                _norm(1/((cg+.01)*(cd+.01))) if cd is not None else _norm(1/(cg+.01)))
    elif name=='Anti-Freq-DOW':
        od,cd=df()
        return (_norm(1/(od+.01)) if od is not None else _norm(1/(np.array([(ops[n-W90:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(cd+.01)) if cd is not None else _norm(1/(np.array([(cps[n-W90:n]==d).mean() for d in range(10)])+.01)))
    elif name=='Anti-Freq-DOW-50':
        od50,cd50=df(50)
        return (_norm(1/(od50+.01)) if od50 is not None else _norm(1/(np.array([(ops[n-W90:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(cd50+.01)) if cd50 is not None else _norm(1/(np.array([(cps[n-W90:n]==d).mean() for d in range(10)])+.01)))
    elif name=='Decay-1':  o,c=ew(1,30);  return _norm(o),_norm(c)
    elif name=='Decay-3':  o,c=ew(3,60);  return _norm(o),_norm(c)
    elif name=='Decay-5':  o,c=ew(5,90);  return _norm(o),_norm(c)
    elif name=='Decay5+AntiF':
        od,cd=ew(5,W90)
        return _norm(.5*_norm(od)+.5*_norm(1/(od+.01))),_norm(.5*_norm(cd)+.5*_norm(1/(cd+.01)))
    elif name=='Balanced':
        oaf=_norm(1/(ol+.01)); caf=_norm(1/(cl+.01))
        o14=np.array([(ops[n-W14:n]==d).mean() for d in range(10)]); c14=np.array([(cps[n-W14:n]==d).mean() for d in range(10)])
        od,cd=df(); od_b=od if od is not None else ol; cd_b=cd if cd is not None else cl
        return _norm(.4*oaf+.35*o14+.25*od_b),_norm(.4*caf+.35*c14+.25*cd_b)
    elif name=='DOW-Only':
        sdow=np.where(dows[:n]==tdow)[0]
        if len(sdow)>=8:
            idx=sdow[-min(len(sdow),30):]; e=np.exp(-np.arange(len(idx),0,-1)/10); e/=e.sum()
            return (_norm(1/(np.array([(e*(ops[idx]==d)).sum() for d in range(10)])+.01)),
                    _norm(1/(np.array([(e*(cps[idx]==d)).sum() for d in range(10)])+.01)))
        return (_norm(1/(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01)))
    elif name=='Markov-1':
        if n>5:
            M=np.ones((10,10))*.1; M2=np.ones((10,10))*.1
            for k in range(1,n): M[ops[k-1],ops[k]]+=1; M2[cps[k-1],cps[k]]+=1
            M/=M.sum(axis=1,keepdims=True); M2/=M2.sum(axis=1,keepdims=True)
            return _norm(M[ops[n-1]]),_norm(M2[cps[n-1]])
        return _norm(np.ones(10)),_norm(np.ones(10))
    elif name=='Markov-2':
        if n>6:
            M3=np.ones((10,10,10))*.05; M4=np.ones((10,10,10))*.05
            for k in range(2,n): M3[ops[k-2],ops[k-1],ops[k]]+=1; M4[cps[k-2],cps[k-1],cps[k]]+=1
            return _norm(M3[ops[n-2],ops[n-1]]),_norm(M4[cps[n-2],cps[n-1]])
        return _norm(np.ones(10)),_norm(np.ones(10))
    elif name=='Extreme-Anti':
        o=np.array([(ops[n-W60:n]==d).mean()+.01 for d in range(10)]); c=np.array([(cps[n-W60:n]==d).mean()+.01 for d in range(10)])
        return _norm(1/o**2),_norm(1/c**2)
    elif name=='Recency-Burst':
        od10,cd10=ew(10); op_r=np.array([(ops[n-W3:n]==d).mean() for d in range(10)]); cp_r=np.array([(cps[n-W3:n]==d).mean() for d in range(10)])
        return _norm(.3*_norm(1/(op_r+.01))+.7*_norm(od10)),_norm(.3*_norm(1/(cp_r+.01))+.7*_norm(cd10))
    elif name=='Decay-2':  o,c=ew(2,45);   return _norm(o),_norm(c)
    elif name=='Decay-7':  o,c=ew(7,90);   return _norm(o),_norm(c)
    elif name=='Decay-10': o,c=ew(10,120); return _norm(o),_norm(c)
    elif name=='Decay-20': o,c=ew(20,150); return _norm(o),_norm(c)
    elif name=='Recent-14':
        return (_norm(np.array([(ops[n-W14:n]==d).mean() for d in range(10)])),
                _norm(np.array([(cps[n-W14:n]==d).mean() for d in range(10)])))
    elif name=='Recent-60':
        return (_norm(np.array([(ops[n-W60:n]==d).mean() for d in range(10)])),
                _norm(np.array([(cps[n-W60:n]==d).mean() for d in range(10)])))
    elif name=='Anti-Last3':
        op_r3=np.array([(ops[n-W3:n]==d).mean() for d in range(10)]); cp_r3=np.array([(cps[n-W3:n]==d).mean() for d in range(10)])
        return _norm(1/(op_r3+.01)),_norm(1/(cp_r3+.01))
    elif name=='DOW-Anti':
        s=np.where(dows[:n]==tdow)[0][-30:]
        if len(s)>=5:
            od=np.array([(ops[s]==d).mean() for d in range(10)]); cd=np.array([(cps[s]==d).mean() for d in range(10)])
            return _norm(1/(od+.01)),_norm(1/(cd+.01))
        return (_norm(1/(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01)))
    elif name=='Skip-Recent-5':
        W5=min(n,5); op=np.array([(ops[n-W5:n]==d).mean() for d in range(10)]); cp=np.array([(cps[n-W5:n]==d).mean() for d in range(10)])
        return _norm(1/(op+.01)),_norm(1/(cp+.01))
    elif name=='Combo-Top3':
        d3o,d3c=ew(3,60)
        hc_o=_norm((np.array([(ops[n-W7:n]==d).mean() for d in range(10)])+.01)/(ol+.01))
        hc_c=_norm((np.array([(cps[n-W7:n]==d).mean() for d in range(10)])+.01)/(cl+.01))
        r30o=_norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])); r30c=_norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)]))
        return _norm(_norm(d3o)/3+hc_o/3+r30o/3),_norm(_norm(d3c)/3+hc_c/3+r30c/3)
    elif name=='Hybrid-60':
        af60o=_norm(1/(np.array([(ops[n-W60:n]==d).mean() for d in range(10)])+.01)); af60c=_norm(1/(np.array([(cps[n-W60:n]==d).mean() for d in range(10)])+.01))
        r60o=_norm(np.array([(ops[n-W60:n]==d).mean() for d in range(10)])); r60c=_norm(np.array([(cps[n-W60:n]==d).mean() for d in range(10)]))
        s=np.where(dows[:n]==tdow)[0][-20:]; dow_w=0.25 if len(s)>=5 else 0.0
        od=np.array([(ops[s]==d).mean() for d in range(10)]) if len(s)>=5 else np.ones(10)/10
        cd=np.array([(cps[s]==d).mean() for d in range(10)]) if len(s)>=5 else np.ones(10)/10
        return (_norm((0.5-dow_w/2)*af60o+(0.25-dow_w/4)*r60o+(0.25-dow_w/4)*ol+dow_w*_norm(1/(od+.01))),
                _norm((0.5-dow_w/2)*af60c+(0.25-dow_w/4)*r60c+(0.25-dow_w/4)*cl+dow_w*_norm(1/(cd+.01))))
    elif name=='Markov-3':
        if n>7:
            M5=np.ones((10,10,10,10))*.02; M6=np.ones((10,10,10,10))*.02
            for k in range(3,n): M5[ops[k-3],ops[k-2],ops[k-1],ops[k]]+=1; M6[cps[k-3],cps[k-2],cps[k-1],cps[k]]+=1
            s5=M5[ops[n-3],ops[n-2],ops[n-1]]; s5/=s5.sum(); s6=M6[cps[n-3],cps[n-2],cps[n-1]]; s6/=s6.sum()
            return _norm(s5),_norm(s6)
        return _norm(np.ones(10)),_norm(np.ones(10))
    elif name=='Mean-Revert-180':
        W=min(n,180)
        return (_norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)),
                _norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)))
    elif name=='Volatility-Anti':
        W=W90; gap_o=np.zeros(10); gap_c=np.zeros(10)
        for d in range(10):
            idx_o=np.where(ops[n-W:n]==d)[0]; idx_c=np.where(cps[n-W:n]==d)[0]
            gap_o[d]=np.diff(idx_o).std()+1 if len(idx_o)>2 else 15; gap_c[d]=np.diff(idx_c).std()+1 if len(idx_c)>2 else 15
        return _norm(gap_o),_norm(gap_c)
    elif name=='Long-Anti-180':
        W=min(n,180)
        return (_norm(1/(np.array([(ops[n-W:n]==d).mean() for d in range(10)])+.01)**1.5),
                _norm(1/(np.array([(cps[n-W:n]==d).mean() for d in range(10)])+.01)**1.5))
    elif name=='DOW-Freq':
        s=np.where(dows[:n]==tdow)[0][-40:]
        if len(s)>=5:
            od=np.array([(ops[s]==d).mean() for d in range(10)]); cd=np.array([(cps[s]==d).mean() for d in range(10)])
            return _norm(od+.01),_norm(cd+.01)
        return (_norm(np.array([(ops[n-W30:n]==d).mean() for d in range(10)])+.01),
                _norm(np.array([(cps[n-W30:n]==d).mean() for d in range(10)])+.01))
    elif name=='Decay3+DOW':
        d3o,d3c=ew(3,60); s=np.where(dows[:n]==tdow)[0][-20:]
        if len(s)>=5:
            od=np.array([(ops[s]==d).mean() for d in range(10)]); cd=np.array([(cps[s]==d).mean() for d in range(10)])
            dow_a_o=_norm(1/(od+.01)); dow_a_c=_norm(1/(cd+.01))
        else:
            dow_a_o=_norm(1/(ol+.01)); dow_a_c=_norm(1/(cl+.01))
        return _norm(.6*_norm(d3o)+.4*dow_a_o),_norm(.6*_norm(d3c)+.4*dow_a_c)
    else:
        o,c=ew(15); return _norm(o),_norm(c)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def _load_all():
    bt  = json.load(open(os.path.join(DATA_DIR,'bt_data.json')))
    ma  = json.load(open(os.path.join(DATA_DIR,'market_assign.json')))
    nsr = json.load(open(os.path.join(DATA_DIR,'newx_strategy_results.json')))
    return bt, ma, nsr


def _get_series(bt, market, cutoff_str):
    rows=[r for r in bt.get(market,[]) if r['date'] < cutoff_str]
    if len(rows)<15: return None,None,None,0
    ops =np.array([r['actual_op'] for r in rows],dtype=int)
    cps =np.array([r['actual_cp'] for r in rows],dtype=int)
    dows=np.array([datetime.strptime(r['date'],'%Y-%m-%d').weekday() for r in rows],dtype=int)
    return ops,cps,dows,len(rows)


def gen_predictions(bt, ma, nsr, target_str, tdow):
    results={}
    for mkt in ALL_MKT:
        if tdow==5 and mkt in SAT_CLOSED: continue
        if tdow==6 and mkt not in SUN_OPEN: continue
        ops,cps,dows,n=_get_series(bt,mkt,target_str)
        if ops is None: continue
        cur_s=ma.get(mkt,{}).get('strategy','Hot-Cold')
        op_p,cp_p=_run_strat(cur_s,ops,cps,dows,tdow,n)
        cfg=nsr.get(mkt,{})
        nx_os=cfg.get('best_open',cur_s); nx_cs=cfg.get('best_close',cur_s)
        nx_op_p,_=_run_strat(nx_os,ops,cps,dows,tdow,n)
        _,nx_cp_p=_run_strat(nx_cs,ops,cps,dows,tdow,n)
        last30=bt.get(mkt,[])[-30:]
        h4=sum(r.get('hit4',0) for r in last30)/max(len(last30),1)
        results[mkt]=dict(
            top4_op=list(map(int,np.argsort(op_p)[::-1][:4])),
            top4_cp=list(map(int,np.argsort(cp_p)[::-1][:4])),
            op_probs=[round(float(x),4) for x in op_p],
            cp_probs=[round(float(x),4) for x in cp_p],
            nx_top4_op=list(map(int,np.argsort(nx_op_p)[::-1][:4])),
            nx_top4_cp=list(map(int,np.argsort(nx_cp_p)[::-1][:4])),
            nx_op_probs=[round(float(x),4) for x in nx_op_p],
            nx_cp_probs=[round(float(x),4) for x in nx_cp_p],
            h4_rate=round(h4,3),
        )
    return results


# ─────────────────────────────────────────────────────────────
# BACKTEST ANALYTICS
# ─────────────────────────────────────────────────────────────
def compute_bt_stats(bt, months=6):
    """Compute monthly hit4 rates per market for the last N months."""
    today = date.today()
    cutoff = (today - timedelta(days=months*30)).strftime('%Y-%m-%d')

    # Monthly breakdown
    monthly = defaultdict(lambda: defaultdict(lambda: {'hits':0,'total':0}))
    window  = {'30d':{},'60d':{},'90d':{},'all':{}}

    for mkt in ALL_MKT:
        rows = [r for r in bt.get(mkt,[]) if r['date'] >= cutoff]
        if not rows: continue
        d30 = (today-timedelta(30)).strftime('%Y-%m-%d')
        d60 = (today-timedelta(60)).strftime('%Y-%m-%d')
        d90 = (today-timedelta(90)).strftime('%Y-%m-%d')

        for key, since in [('30d',d30),('60d',d60),('90d',d90),('all',cutoff)]:
            sub = [r for r in bt.get(mkt,[]) if r['date'] >= since]
            if sub:
                window[key][mkt] = round(sum(r.get('hit4',0) for r in sub)/len(sub)*100,1)

        for r in rows:
            ym = r['date'][:7]  # YYYY-MM
            monthly[ym][mkt]['hits']  += r.get('hit4',0)
            monthly[ym][mkt]['total'] += 1

    months_sorted = sorted(monthly.keys())
    return monthly, months_sorted, window


def compute_jodi_stats(bt, days=90):
    """Most and least common jodis per market over last N days."""
    cutoff = (date.today()-timedelta(days)).strftime('%Y-%m-%d')
    stats = {}
    for mkt in ALL_MKT:
        rows = [r for r in bt.get(mkt,[]) if r['date'] >= cutoff and 'actual_jodi' in r]
        if not rows: continue
        freq = defaultdict(int)
        for r in rows:
            jd = r.get('actual_jodi','')
            if len(str(jd))==2: freq[str(jd)] += 1
        total = len(rows)
        sorted_jd = sorted(freq.items(), key=lambda x:-x[1])
        stats[mkt] = {'hot':sorted_jd[:5], 'cold':sorted_jd[-5:][::-1], 'total':total}
    return stats


def compute_digit_freq(bt, days=30):
    """Open/close digit frequency per market, last N days."""
    cutoff = (date.today()-timedelta(days)).strftime('%Y-%m-%d')
    freq = {}
    for mkt in ALL_MKT:
        rows = [r for r in bt.get(mkt,[]) if r['date'] >= cutoff]
        if not rows: continue
        op_f = [sum(r['actual_op']==d for r in rows)/len(rows) for d in range(10)]
        cp_f = [sum(r['actual_cp']==d for r in rows)/len(rows) for d in range(10)]
        freq[mkt] = {'op':op_f,'cp':cp_f,'n':len(rows)}
    return freq


# ─────────────────────────────────────────────────────────────
# EXCEL PARSER
# ─────────────────────────────────────────────────────────────
def _panna_to_digit(v):
    try:
        s=str(int(v)).zfill(3); return (int(s[0])+int(s[1])+int(s[2]))%10
    except: return None

def parse_excel_to_bt(uploaded_bytes, existing_bt):
    try:
        xl = pd.ExcelFile(io.BytesIO(uploaded_bytes))
        bt = {k:list(v) for k,v in existing_bt.items()}
        new_count=0
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df.columns=[str(c).strip() for c in df.columns]
            col_map={}
            for col in df.columns:
                cl=col.lower()
                if 'date' in cl: col_map['date']=col
                elif any(x in cl for x in ['market','mkt']): col_map['market']=col
                elif col.lower() in ('open','op'): col_map['open']=col
                elif col.lower() in ('close','cp','close panna','open panna'): col_map['close']=col
            sheet_mkt=next((m for m in ALL_MKT if m.lower() in sheet.lower()),None)
            if 'date' not in col_map: continue
            existing_dates={mkt:{r['date'] for r in bt.get(mkt,[])} for mkt in ALL_MKT}
            for _,row in df.iterrows():
                try:
                    dt_raw=row[col_map['date']]
                    if pd.isna(dt_raw): continue
                    if isinstance(dt_raw,(datetime,pd.Timestamp)): dt_str=pd.Timestamp(dt_raw).strftime('%Y-%m-%d')
                    else: dt_str=str(dt_raw).strip()[:10]
                    datetime.strptime(dt_str,'%Y-%m-%d')
                    mkt=sheet_mkt
                    if 'market' in col_map:
                        mr=str(row[col_map['market']]).strip()
                        mkt=next((m for m in ALL_MKT if m.lower()==mr.lower()),mkt)
                    if not mkt: continue
                    if 'open' not in col_map: continue
                    ov=row[col_map['open']]
                    if pd.isna(ov): continue
                    iv=int(ov); op_d=iv%10 if iv<10 else _panna_to_digit(iv)
                    cv=row.get(col_map.get('close',''),None) if 'close' in col_map else None
                    cp_d=None
                    if cv is not None and not (isinstance(cv,float) and np.isnan(cv)):
                        icv=int(cv); cp_d=icv%10 if icv<10 else _panna_to_digit(icv)
                    if op_d is None: continue
                    if dt_str in existing_dates.get(mkt,set()): continue
                    if mkt not in bt: bt[mkt]=[]
                    bt[mkt].append({'date':dt_str,'actual_op':op_d,'actual_cp':cp_d if cp_d is not None else op_d,
                                    'actual_jodi':f"{op_d}{cp_d if cp_d is not None else op_d}",'hit4':0,'strategy':'uploaded'})
                    existing_dates.setdefault(mkt,set()).add(dt_str)
                    new_count+=1
                except: continue
        for mkt in bt: bt[mkt]=sorted(bt[mkt],key=lambda r:r['date'])
        return bt,new_count,None
    except Exception as e:
        return existing_bt,0,str(e)


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def _session_tag(mkt):
    return "☀️" if mkt in DAY_MKT else "🌙"

def _d4(digits):
    return "  ".join(f"**{d}**" for d in digits)

def _prob4(digits, probs):
    return "  |  ".join(f"{d} → {probs[d]*100:.1f}%" for d in digits)

def _hr_color(hr):
    if hr>=0.45: return "#34a853"
    if hr>=0.38: return "#fbbc04"
    return "#ea4335"

def _trend_arrow(monthly, mkt, months_sorted):
    if len(months_sorted)<2: return "→"
    last2=[monthly[m].get(mkt,{}) for m in months_sorted[-2:]]
    r0=last2[0].get('hits',0)/max(last2[0].get('total',1),1)
    r1=last2[1].get('hits',0)/max(last2[1].get('total',1),1)
    if r1>r0+0.03: return "↑"
    if r1<r0-0.03: return "↓"
    return "→"


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Matka Predict Pro v7", page_icon="🎯",
                       layout="wide", initial_sidebar_state="expanded")

    st.markdown("""<style>
    .block-container{padding-top:1rem}
    .pred-row{background:#1e2130;border-radius:10px;padding:12px 18px;margin:6px 0;border-left:4px solid #1a73e8}
    .pred-row-nx{border-left-color:#34a853}
    .top-card{background:linear-gradient(135deg,#0d2a0d,#1e2130);border:1px solid #34a853;border-radius:10px;padding:14px 18px;margin:5px 0}
    .bot-card{background:linear-gradient(135deg,#2a0d0d,#1e2130);border:1px solid #ea4335;border-radius:10px;padding:14px 18px;margin:5px 0}
    .kpi{background:#1a1f35;border-radius:8px;padding:14px;text-align:center}
    .kpi-v{font-size:2.1em;font-weight:800;color:#1a73e8}
    .kpi-l{font-size:0.78em;color:#8891b0;margin-top:4px}
    .digit-pill{display:inline-block;width:32px;height:32px;border-radius:50%;background:#1a73e8;color:white;
                text-align:center;line-height:32px;font-weight:800;font-size:15px;margin:2px}
    .digit-pill-nx{background:#34a853}
    </style>""", unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 Matka Predict Pro")
        st.markdown("---")
        today = date.today()
        target = st.date_input("📅 Prediction Date", value=today,
                                min_value=date(2025,4,1), max_value=today+timedelta(7))
        tdow = target.weekday()
        tstr = target.strftime('%Y-%m-%d')
        dow  = DOW_NAMES[tdow]

        st.markdown(f"**{dow}** · {tstr}")
        if tdow==5: st.warning("Saturday: 3 markets closed")
        elif tdow==6: st.info("Sunday: 4 markets open only")

        st.markdown("---")
        st.markdown("**📂 Add New Data**")
        uploaded = st.file_uploader("Upload Excel (.xlsx)",type=["xlsx"],
                                     help="Columns: Date | Market | Open | Close")
        st.markdown("---")
        model_filter = st.radio("Show Model", ["Both","Model A (Current)","Model B (NX)"])
        st.markdown("---")
        st.caption("Matka Predict Pro v7 · Confidential")

    # ── Load ─────────────────────────────────────────────────
    bt, ma, nsr = _load_all()

    upload_msg=None
    if uploaded:
        new_bt,cnt,err=parse_excel_to_bt(uploaded.read(),bt)
        if err: st.error(f"Upload error: {err}")
        else:
            bt=new_bt
            upload_msg=f"✅ Added {cnt} new rows." if cnt else "ℹ️ No new rows (already exist)."
            st.cache_data.clear()
    if upload_msg: st.success(upload_msg)

    # ── Generate predictions ──────────────────────────────────
    with st.spinner(f"Running models for {tstr}…"):
        preds=gen_predictions(bt,ma,nsr,tstr,tdow)

    if not preds:
        st.error("No active markets for this date."); return

    mkt_list=list(preds.keys())
    day_mkts =[m for m in mkt_list if m in DAY_MKT]
    ngt_mkts =[m for m in mkt_list if m not in DAY_MKT]
    ranked=sorted(mkt_list,key=lambda m:preds[m]['h4_rate'],reverse=True)

    # ── KPI bar ───────────────────────────────────────────────
    st.markdown(f"## 🎯 Predictions — {tstr} ({dow})")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(f'<div class="kpi"><div class="kpi-v">{len(mkt_list)}</div><div class="kpi-l">Active Markets</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi"><div class="kpi-v">{len(day_mkts)}</div><div class="kpi-l">☀️ Day Sessions</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi"><div class="kpi-v">{len(ngt_mkts)}</div><div class="kpi-l">🌙 Night Sessions</div></div>',unsafe_allow_html=True)
    with c4:
        avg_hr=sum(preds[m]['h4_rate'] for m in mkt_list)/len(mkt_list)
        exp=round(avg_hr*len(mkt_list))
        st.markdown(f'<div class="kpi"><div class="kpi-v">{exp}</div><div class="kpi-l">Expected Hits (~{avg_hr*100:.0f}%)</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    # ── TOP 5 / BOTTOM 3 ─────────────────────────────────────
    colA,colB=st.columns(2)
    with colA:
        st.markdown("### 🏆 Top 5 — Most Reliable")
        for i,mkt in enumerate(ranked[:5],1):
            p=preds[mkt]; hr=p['h4_rate']
            o4=" · ".join(str(d) for d in p['top4_op']); c4=" · ".join(str(d) for d in p['top4_cp'])
            st.markdown(
                f'<div class="top-card">'
                f'<b style="color:#34a853">#{i}</b> {_session_tag(mkt)} <b style="font-size:1.05em">{mkt}</b>'
                f'<span style="float:right;color:#34a853;font-weight:700">{hr*100:.1f}%</span><br>'
                f'<small>Open: <b>{o4}</b> &nbsp;|&nbsp; Close: <b>{c4}</b></small></div>',
                unsafe_allow_html=True)

    with colB:
        st.markdown("### ⚠️ Bottom 3 — Lowest Reliability")
        for i,mkt in enumerate(ranked[-3:],1):
            p=preds[mkt]; hr=p['h4_rate']
            o4=" · ".join(str(d) for d in p['top4_op']); c4=" · ".join(str(d) for d in p['top4_cp'])
            st.markdown(
                f'<div class="bot-card">'
                f'<b style="color:#ea4335">⚠{i}</b> {_session_tag(mkt)} <b style="font-size:1.05em">{mkt}</b>'
                f'<span style="float:right;color:#ea4335;font-weight:700">{hr*100:.1f}%</span><br>'
                f'<small>Open: <b>{o4}</b> &nbsp;|&nbsp; Close: <b>{c4}</b></small></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── MAIN TABS ─────────────────────────────────────────────
    tab_pred, tab_bt, tab_trend, tab_jodi = st.tabs([
        "📋 All Predictions", "📊 Backtest", "📈 Trends", "🎲 Jodi Analysis"
    ])

    # ════════════════════════════════════════
    # TAB 1 — PREDICTIONS
    # ════════════════════════════════════════
    with tab_pred:
        show_nx  = model_filter in ("Both","Model B (NX)")
        show_cur = model_filter in ("Both","Model A (Current)")

        # ── Summary table ──
        rows=[]
        for mkt in mkt_list:
            p=preds[mkt]
            row={'Market':f"{_session_tag(mkt)} {mkt}",
                 'Hit Rate':f"{p['h4_rate']*100:.1f}%"}
            if show_cur:
                row['A — Open Top 4']=" ".join(str(d) for d in p['top4_op'])
                row['A — Close Top 4']=" ".join(str(d) for d in p['top4_cp'])
            if show_nx:
                row['B — Open Top 4']=" ".join(str(d) for d in p['nx_top4_op'])
                row['B — Close Top 4']=" ".join(str(d) for d in p['nx_top4_cp'])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        # ── Detailed cards ──
        st.markdown("#### Detailed Predictions")
        cols=st.columns(2)
        for i,mkt in enumerate(mkt_list):
            p=preds[mkt]; hr=p['h4_rate']
            with cols[i%2]:
                with st.expander(f"{_session_tag(mkt)} {mkt}  —  {hr*100:.1f}% hit rate"):
                    if show_cur:
                        st.markdown("**Model A**")
                        pills_op="".join(f'<span class="digit-pill">{d}</span>' for d in p['top4_op'])
                        pills_cp="".join(f'<span class="digit-pill">{d}</span>' for d in p['top4_cp'])
                        st.markdown(f"Open: {pills_op}",unsafe_allow_html=True)
                        st.markdown(f"Close: {pills_cp}",unsafe_allow_html=True)
                        st.markdown(f"<small>{_prob4(p['top4_op'],p['op_probs'])}</small>",unsafe_allow_html=True)
                        if show_nx: st.markdown("---")
                    if show_nx:
                        st.markdown("**Model B**")
                        pills_nop="".join(f'<span class="digit-pill digit-pill-nx">{d}</span>' for d in p['nx_top4_op'])
                        pills_ncp="".join(f'<span class="digit-pill digit-pill-nx">{d}</span>' for d in p['nx_top4_cp'])
                        st.markdown(f"Open: {pills_nop}",unsafe_allow_html=True)
                        st.markdown(f"Close: {pills_ncp}",unsafe_allow_html=True)
                        st.markdown(f"<small>{_prob4(p['nx_top4_op'],p['nx_op_probs'])}</small>",unsafe_allow_html=True)
                    # Probability bars
                    if show_cur and show_nx:
                        chart=pd.DataFrame({'Model A':p['op_probs'],'Model B (open)':p['nx_op_probs']},index=range(10))
                        st.bar_chart(chart,height=160)
                    elif show_cur:
                        chart=pd.DataFrame({'Open':p['op_probs'],'Close':p['cp_probs']},index=range(10))
                        st.bar_chart(chart,height=160)

                    # Jodi overlap
                    if show_cur and show_nx:
                        cur_j={f"{o}{c}" for o in p['top4_op'] for c in p['top4_cp']}
                        nx_j ={f"{o}{c}" for o in p['nx_top4_op'] for c in p['nx_top4_cp']}
                        overlap=cur_j&nx_j
                        if overlap:
                            st.success(f"Both models agree on **{len(overlap)}** jodis: "
                                       f"{', '.join(sorted(overlap)[:8])}")

    # ════════════════════════════════════════
    # TAB 2 — BACKTEST
    # ════════════════════════════════════════
    with tab_bt:
        monthly, months_sorted, window = compute_bt_stats(bt, months=6)

        st.markdown("### 📊 Hit Rate Performance — Last 90 Days")
        st.caption("Hit rate = % of sessions where actual result digit appeared in Top-4 predictions. Baseline (random) = 40%.")

        # Window summary table
        wrows=[]
        for mkt in ALL_MKT:
            if mkt not in window['all']: continue
            trend=_trend_arrow(monthly,mkt,months_sorted)
            wrows.append({
                'Market':f"{_session_tag(mkt)} {mkt}",
                'Trend':trend,
                'Last 30d':f"{window['30d'].get(mkt,0):.1f}%",
                'Last 60d':f"{window['60d'].get(mkt,0):.1f}%",
                'Last 90d':f"{window['90d'].get(mkt,0):.1f}%",
            })
        wrows.sort(key=lambda r:-float(r['Last 30d'].rstrip('%')))
        st.dataframe(pd.DataFrame(wrows),use_container_width=True,hide_index=True)

        # Hit rate bar chart (30d)
        st.markdown("#### Hit Rate by Market (Last 30 Days)")
        bar_data=pd.DataFrame({
            'Hit Rate %':[window['30d'].get(m,0) for m in ALL_MKT if m in window['30d']]
        },index=[m for m in ALL_MKT if m in window['30d']])
        st.bar_chart(bar_data,height=280)
        st.markdown('<small style="color:#666">— — 40% baseline (random)</small>',unsafe_allow_html=True)

        # Monthly breakdown
        if months_sorted:
            st.markdown("#### Monthly Hit Rates")
            month_rows=[]
            for ym in months_sorted[-6:]:
                row={'Month':ym}
                for mkt in ALL_MKT:
                    d=monthly[ym].get(mkt,{})
                    tot=d.get('total',0)
                    row[mkt]=f"{d.get('hits',0)/tot*100:.0f}%" if tot else "—"
                month_rows.append(row)
            st.dataframe(pd.DataFrame(month_rows),use_container_width=True,hide_index=True)

        # Day-session vs Night-session
        st.markdown("#### Day vs Night Session Performance (Last 30d)")
        dc=[window['30d'].get(m,0) for m in ALL_MKT if m in DAY_MKT and m in window['30d']]
        nc=[window['30d'].get(m,0) for m in ALL_MKT if m not in DAY_MKT and m in window['30d']]
        avg_d=sum(dc)/len(dc) if dc else 0
        avg_n=sum(nc)/len(nc) if nc else 0
        col1,col2=st.columns(2)
        with col1: st.metric("☀️ Day Avg Hit Rate",f"{avg_d:.1f}%","(last 30d)")
        with col2: st.metric("🌙 Night Avg Hit Rate",f"{avg_n:.1f}%","(last 30d)")

    # ════════════════════════════════════════
    # TAB 3 — TRENDS
    # ════════════════════════════════════════
    with tab_trend:
        monthly, months_sorted, window = compute_bt_stats(bt, months=6)

        st.markdown("### 📈 Market Trends — 6 Month View")

        if len(months_sorted) >= 2:
            # Build a line chart: monthly hit rate per market
            chart_data = {}
            for mkt in ALL_MKT:
                series=[]
                for ym in months_sorted:
                    d=monthly[ym].get(mkt,{})
                    tot=d.get('total',0)
                    series.append(d.get('hits',0)/tot*100 if tot else None)
                if any(v is not None for v in series):
                    chart_data[mkt]=series

            if chart_data:
                df_chart=pd.DataFrame(chart_data,index=months_sorted).fillna(method='ffill')
                st.line_chart(df_chart,height=320)

        # Trending up/down markets
        improving=[]; declining=[]
        for mkt in ALL_MKT:
            if mkt not in window['all']: continue
            arrow=_trend_arrow(monthly,mkt,months_sorted)
            if arrow=='↑': improving.append(mkt)
            elif arrow=='↓': declining.append(mkt)

        col1,col2=st.columns(2)
        with col1:
            st.markdown("**↑ Improving Markets**")
            for m in improving:
                r30=window['30d'].get(m,0); r60=window['60d'].get(m,0)
                st.markdown(f"✅ {_session_tag(m)} **{m}** — 30d: {r30:.1f}% &nbsp; 60d: {r60:.1f}%",unsafe_allow_html=True)
            if not improving: st.caption("None right now")
        with col2:
            st.markdown("**↓ Declining Markets**")
            for m in declining:
                r30=window['30d'].get(m,0); r60=window['60d'].get(m,0)
                st.markdown(f"⚠️ {_session_tag(m)} **{m}** — 30d: {r30:.1f}% &nbsp; 60d: {r60:.1f}%",unsafe_allow_html=True)
            if not declining: st.caption("None right now")

        # DOW performance
        st.markdown("#### Hit Rate by Day of Week (all markets combined, last 90d)")
        cutoff90=(date.today()-timedelta(90)).strftime('%Y-%m-%d')
        dow_hits=defaultdict(lambda:{'h':0,'t':0})
        for mkt in ALL_MKT:
            for r in bt.get(mkt,[]):
                if r['date']<cutoff90: continue
                dw=datetime.strptime(r['date'],'%Y-%m-%d').weekday()
                dow_hits[dw]['h']+=r.get('hit4',0); dow_hits[dw]['t']+=1
        dow_df=pd.DataFrame({
            'Day':[DOW_NAMES[i] for i in range(7) if dow_hits[i]['t']>0],
            'Hit Rate %':[dow_hits[i]['h']/dow_hits[i]['t']*100 for i in range(7) if dow_hits[i]['t']>0],
            'Sessions':[dow_hits[i]['t'] for i in range(7) if dow_hits[i]['t']>0]
        })
        if not dow_df.empty:
            st.bar_chart(dow_df.set_index('Day')[['Hit Rate %']],height=250)
            st.dataframe(dow_df,use_container_width=True,hide_index=True)

    # ════════════════════════════════════════
    # TAB 4 — JODI ANALYSIS
    # ════════════════════════════════════════
    with tab_jodi:
        st.markdown("### 🎲 Jodi Analysis — Last 90 Days")
        jodi_stats = compute_jodi_stats(bt, days=90)
        dig_freq   = compute_digit_freq(bt, days=30)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔥 Hot Jodis (most frequent)**")
            hot_rows=[]
            for mkt in ALL_MKT:
                if mkt not in jodi_stats: continue
                for jd,cnt in jodi_stats[mkt]['hot'][:3]:
                    hot_rows.append({'Market':f"{_session_tag(mkt)} {mkt}",'Jodi':jd,'Count':cnt,
                                     'Rate %':f"{cnt/jodi_stats[mkt]['total']*100:.1f}%"})
            hot_rows.sort(key=lambda r:-r['Count'])
            if hot_rows: st.dataframe(pd.DataFrame(hot_rows[:20]),use_container_width=True,hide_index=True)

        with col2:
            st.markdown("**❄️ Cold Jodis (least frequent / due)**")
            cold_rows=[]
            for mkt in ALL_MKT:
                if mkt not in jodi_stats: continue
                for jd,cnt in jodi_stats[mkt]['cold'][:3]:
                    cold_rows.append({'Market':f"{_session_tag(mkt)} {mkt}",'Jodi':jd,'Count':cnt,
                                      'Rate %':f"{cnt/jodi_stats[mkt]['total']*100:.1f}%"})
            cold_rows.sort(key=lambda r:r['Count'])
            if cold_rows: st.dataframe(pd.DataFrame(cold_rows[:20]),use_container_width=True,hide_index=True)

        # Digit frequency heatmap
        st.markdown("#### Digit Frequency — Last 30 Days (Open)")
        st.caption("How often each digit 0–9 appeared in Open results per market")
        if dig_freq:
            heat_data=pd.DataFrame(
                {mkt:[round(dig_freq[mkt]['op'][d]*100,1) for d in range(10)]
                 for mkt in ALL_MKT if mkt in dig_freq},
                index=list(range(10))
            )
            st.dataframe(heat_data.style.background_gradient(cmap='YlOrRd',axis=None),
                         use_container_width=True)

        # Per-market jodi expanders
        st.markdown("#### Per-Market Jodi Breakdown")
        cols=st.columns(2)
        for i,mkt in enumerate(ALL_MKT):
            if mkt not in jodi_stats: continue
            with cols[i%2]:
                with st.expander(f"{_session_tag(mkt)} {mkt}  ({jodi_stats[mkt]['total']} sessions)"):
                    hot_j="  ".join(f"**{j[0]}**({j[1]})" for j in jodi_stats[mkt]['hot'][:5])
                    cold_j="  ".join(f"**{j[0]}**({j[1]})" for j in jodi_stats[mkt]['cold'][:5])
                    st.markdown(f"🔥 Hot: {hot_j}")
                    st.markdown(f"❄️ Cold: {cold_j}")
                    if mkt in dig_freq:
                        df_d=pd.DataFrame({'Open %':[round(x*100,1) for x in dig_freq[mkt]['op']],
                                           'Close %':[round(x*100,1) for x in dig_freq[mkt]['cp']]},
                                          index=range(10))
                        st.bar_chart(df_d,height=150)

    # ── Footer ───────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"Matka Predict Pro v7 · {tstr} ({dow}) · {len(mkt_list)} markets active · 35-algorithm engine")


if __name__ == '__main__':
    main()
