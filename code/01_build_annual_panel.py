# -*- coding: utf-8 -*-
"""
扩样本后的完整重跑（v3）。回答 D3：
  · 2016 安慰剂是否仍然显著？
  · 股票特定趋势下的衰减是否仍存？
  · Holm 校正后有哪些期限存活？
  · 长期限的精度是否改善？
输出与 360 只样本的并排对比。
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
PANEL_IN = ARGS.get("panel", None)
OUT = ARGS.get("outdir", os.path.join(BASE, "results_v3"))
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "v3.log")
open(LOG, "w").close()
def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

MS = list(range(2, 41))

# ================= 建面板（限幅恒定窗口） =================
SRC = os.path.join(DATA, "panel_merged.parquet")
if not os.path.exists(SRC):
    SRC = os.path.join(DATA, "panel_raw.parquet")
log(f"数据源: {SRC}")
ct = pd.read_parquet(SRC)
for c in ["open","high","low","close","preclose","volume"]:
    ct[c] = pd.to_numeric(ct[c], errors="coerce")
ct["tradestatus"] = pd.to_numeric(ct["tradestatus"], errors="coerce")
ct["isST"] = pd.to_numeric(ct["isST"], errors="coerce").fillna(0).astype(int)
ct["date"] = pd.to_datetime(ct["date"])
ct = ct[ct.tradestatus == 1].dropna(subset=["preclose","close"])
ct = ct[(ct.preclose > 0) & (ct.close > 0)].copy()
ct["board"] = ct.code.str[:5].map({"sh.60":"main","sz.00":"main","sz.30":"gem","sh.68":"star"})
ct = ct.sort_values(["code","date"]).reset_index(drop=True)
ct["ret"] = np.log(ct.close/ct.preclose)
GEM = pd.Timestamp("2020-08-24")
ct["band"] = np.where((ct.board=="gem") & (ct.date>=GEM), 0.20,
             np.where(ct.isST==1, 0.05, 0.10))
ct["lim"] = np.round(ct.preclose*(1+ct.band), 2)
ct["hit"] = (ct.close >= ct.lim - 1e-6).astype(int)

cal = np.sort(ct["date"].unique()); cim = {d:i for i,d in enumerate(cal)}; ND = len(cal)
ct["ci"] = ct["date"].map(cim)
board_of = ct.groupby("code")["board"].first().to_dict()
tickers = sorted(board_of)
W, STRIDE, MINWIN = 250, 5, 10
MS = list(range(2, 41))
log(f"面板 {ct.shape}, 股票 {len(tickers)} ({pd.Series(board_of).value_counts().to_dict()}), 交易日 {ND}")

rows = []
for t in tickers:
    g = ct[ct.code == t]
    idx = g["ci"].to_numpy()
    x = np.full(ND, np.nan); h = np.zeros(ND); yr = np.full(ND,-1); bd = np.full(ND, np.nan)
    x[idx] = g["ret"].to_numpy(); h[idx] = g["hit"].to_numpy()
    yr[idx] = g["date"].dt.year.to_numpy(); bd[idx] = g["band"].to_numpy()
    cfin = np.concatenate([[0], np.cumsum(np.isfinite(x))])
    nw = ND - W + 1
    if nw <= STRIDE: continue
    st = np.arange(0, nw, STRIDE)
    st = st[(cfin[st+W]-cfin[st]) == W]
    if st.size == 0: continue
    sw = np.lib.stride_tricks.sliding_window_view(x, W)[st]
    v1 = sw.var(axis=1)
    bwin = np.lib.stride_tricks.sliding_window_view(bd, W)[st]
    bok = np.isfinite(bwin).all(axis=1) & (np.nanmax(bwin,axis=1)-np.nanmin(bwin,axis=1) < 1e-12)
    raw = bwin[:,0]
    keep = (v1>0) & bok
    if keep.sum()==0: continue
    ch = np.concatenate([[0], np.cumsum(h)])
    hitr = (ch[st+W]-ch[st])/W
    vol = sw.std(axis=1); yc = yr[st+W//2]
    vr = {}
    for m in MS:
        ys = pd.Series(x).rolling(m, min_periods=m).sum().to_numpy()
        yy = np.lib.stride_tricks.sliding_window_view(ys, W-m+1)[st]
        good = np.isfinite(yy).all(axis=1)
        vr[m] = np.where(good, yy.var(axis=1)/(m*v1), np.nan)
        keep &= good
    for j in np.where(keep & (yc>=0))[0]:
        r = {"ticker":t,"year":int(yc[j]),"board":board_of[t],"band":raw[j],
             "hit_rate":hitr[j],"vol":vol[j]}
        for m in MS: r[f"vr{m}"] = vr[m][j]
        rows.append(r)
R = pd.DataFrame(rows)
agg = {"hit_rate":("hit_rate","mean"),"vol":("vol","mean"),"band":("band","first"),
       "n":("vr2","count")}
for m in MS: agg[f"vr{m}"] = (f"vr{m}","mean")
cell = R.groupby(["ticker","year","board"]).agg(**agg).reset_index()
cell = cell[(cell.n>=MINWIN) & (cell.year!=2020)].copy()
cell["treated"] = (cell.board=="gem").astype(int)
cell["post"] = (cell.year>=2021).astype(int)
cell.to_parquet(os.path.join(OUT,"panel_v3.parquet"), index=False)
log(f"窗口 {len(R)}, 股票-年单元 {len(cell)}, 处理 {cell[cell.treated==1].ticker.nunique()} 只, "
    f"对照 {cell[cell.treated==0].ticker.nunique()} 只")

# ================= 引擎 =================
def build_op(d, trend=False):
    tv = np.sort(d.ticker.unique()); tm = {v:i for i,v in enumerate(tv)}
    yv = np.sort(d.year.unique());   ym = {v:i for i,v in enumerate(yv)}
    gj = d.ticker.map(tm).to_numpy(); gi = d.year.map(ym).to_numpy()
    n, nt, ny = len(d), len(tv), len(yv)
    Dst = np.zeros((n,nt)); Dst[np.arange(n),gj] = 1
    Dyr = np.zeros((n,ny)); Dyr[np.arange(n),gi] = 1
    blk = [np.ones((n,1)), Dst, Dyr]
    if trend:
        tt = (d.year-d.year.min()).to_numpy(float)
        T = np.zeros((n,nt)); T[np.arange(n),gj] = tt
        blk.append(T)
    D = np.hstack(blk)
    Q,Rq = np.linalg.qr(D)
    dg = np.abs(np.diag(Rq)); rk = int((dg > dg.max()*1e-10).sum())
    Q = Q[:,:rk]
    # 预计算：每只股票一个残差化后的 treated×post 方向
    Rbasis = (Q.T @ (d.post.to_numpy(float)[:,None] * Dst))
    base = (d.post.to_numpy(float)[:,None]*Dst) - Q @ Rbasis
    return dict(Q=Q, gj=gj, gi=gi, nt=nt, ny=ny, tv=tv, base=base,
                post=d.post.to_numpy(float), rank=rk)

def did(d, y, op=None, trend=False, wcol=None, B=299, seed=17):
    d2 = d.dropna(subset=[y]).copy()
    op = build_op(d2, trend) if op is None else op
    Q = op["Q"]
    Y = d2[y].to_numpy(float); Y = Y - Q @ (Q.T @ Y)
    G0 = (d2.treated*d2.post).to_numpy(float)
    G = G0 - Q @ (Q.T @ G0)
    ww = np.ones(len(d2)) if wcol is None else d2[wcol].to_numpy(float)
    s = (ww*G**2).sum()
    if not np.isfinite(s) or s <= 1e-14: return None
    b = (ww*G*Y).sum()/s; r = Y - b*G
    XtXi = 1.0/s; gj, gi, nt, ny = op["gj"], op["gi"], op["nt"], op["ny"]
    z1 = np.bincount(gj, weights=ww*G*r, minlength=nt)
    z2 = np.bincount(gi, weights=ww*G*r, minlength=ny)
    se_tk = np.sqrt((z1**2).sum())*XtXi*nt/(nt-1)
    se_yr = np.sqrt((z2**2).sum())*XtXi*ny/(ny-1)
    rng = np.random.default_rng(seed)
    tvv = op["tv"]; ntr = d2[d2.treated==1].ticker.nunique()
    null = []
    for _ in range(B):
        fake = np.isin(tvv, rng.choice(tvv, ntr, replace=False)).astype(float)
        gg = op["base"] @ fake
        ss = (ww*gg**2).sum()
        if not np.isfinite(ss) or ss <= 1e-14: continue
        null.append((ww*gg*Y).sum()/ss)
    null = np.array(null)
    return dict(beta=b, se_tk=se_tk, se_yr=se_yr,
                p=float(np.mean(np.abs(null)>=abs(b))) if len(null) else np.nan, N=len(d2))

d0 = cell[cell.year.isin([2017,2018,2019,2021,2022,2023])].copy()
op0 = build_op(d0.dropna(subset=["vr2"]))

# ---- 1 全期限 + Holm ----
log("\n" + "="*100); log("1 全期限扫描 + Holm"); log("="*100)
scan = []
for m in MS:
    o = did(d0, f"vr{m}", op=op0 if m==2 else None, B=299)
    if o is None: continue
    scan.append({"m":m,"beta":o["beta"],"t_stock":o["beta"]/o["se_tk"],
                 "t_year":o["beta"]/o["se_yr"],"p":o["p"],"N":o["N"]})
S = pd.DataFrame(scan)
p = S.p.to_numpy(); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
for rk, i in enumerate(order):
    run = max(run, min(1.0, (len(p)-rk)*p[i])); adj[i] = run
S["p_holm"] = adj
S.to_csv(os.path.join(OUT,"scan.csv"), index=False, encoding="utf-8-sig")
log(f"{'m':>3} {'beta':>9} {'t(股票)':>9} {'置换p':>7} {'Holm p':>8}")
for _,r in S.iterrows():
    if r.m<=12 or r.m%5==0:
        log(f"{int(r.m):>3} {r.beta:>+9.4f} {r.t_stock:>+9.2f} {r.p:>7.3f} {r.p_holm:>8.3f}")
log(f"  Holm<0.10: {S[S.p_holm<0.10].m.astype(int).tolist()}")
log(f"  Holm<0.05: {S[S.p_holm<0.05].m.astype(int).tolist()}")

# ---- 2 联合检验 ----
log("\n" + "="*100); log("2 短期联合检验 m=2..5"); log("="*100)
short=[2,3,4,5]
ops = {m: build_op(d0.dropna(subset=[f"vr{m}"])) for m in short}
Ys = {m: (lambda d,o: (lambda Y: Y - o["Q"]@(o["Q"].T@Y))(d[f"vr{m}"].to_numpy(float)))(
        d0.dropna(subset=[f"vr{m}"]), ops[m]) for m in short}
def jstat(fake=None):
    tot=[]
    for m in short:
        o = ops[m]; Y = Ys[m]
        if fake is None:
            d2 = d0.dropna(subset=[f"vr{m}"])
            g = (d2.treated*d2.post).to_numpy(float)
            G = g - o["Q"]@(o["Q"].T@g)
        else:
            G = o["base"] @ fake
        s=(G**2).sum()
        if s<=1e-14: return np.nan
        tot.append((G*Y).sum()/s)
    return float(np.mean(tot))
base = jstat()
rng = np.random.default_rng(99); ntr = d0[d0.treated==1].ticker.nunique()
null=[]
for _ in range(499):
    fake = np.isin(ops[2]["tv"], rng.choice(ops[2]["tv"], ntr, replace=False)).astype(float)
    v = jstat(fake)
    if np.isfinite(v): null.append(v)
null=np.array(null)
log(f"  平均短期效应 {base:+.4f}，零 sd {null.std():.4f}，置换 p = {np.mean(np.abs(null)>=abs(base)):.3f}")

# ---- 3 窗口敏感性 ----
log("\n" + "="*100); log("3 窗口敏感性"); log("="*100)
ws=[]
for ps in [2014,2015,2016,2017,2018]:
    d = cell[cell.year.isin(list(range(ps,2020))+[2021,2022,2023])].copy()
    line=f"  {ps}-2019: "
    for m in [2,3,5]:
        o = did(d, f"vr{m}", B=199)
        if o is None: line+=f"m{m} n/a  "; continue
        line += f"m{m} {o['beta']:+.4f}(p{o['p']:.3f})  "
        ws.append({"pre_start":ps,"m":m,"beta":o["beta"],"p":o["p"],"N":o["N"]})
    log(line)
pd.DataFrame(ws).to_csv(os.path.join(OUT,"window.csv"), index=False, encoding="utf-8-sig")

# ---- 4 安慰剂日期（关键） ----
log("\n" + "="*100); log("4 安慰剂改制日（回答 D3）"); log("="*100)
pl=[]
for fy, pre, post in [(2016,[2014,2015],[2017,2018,2019]),
                      (2017,[2014,2015,2016],[2018,2019]),
                      (2018,[2015,2016,2017],[2019])]:
    d = cell[cell.year.isin(pre+post)].copy()
    d["post"] = (d.year>=fy+1).astype(int)
    o2 = did(d,"vr2",B=499); o3 = did(d,"vr3",B=299)
    log(f"  假 {fy}: m=2 β {o2['beta']:+.4f}  t {o2['beta']/o2['se_tk']:+.2f}  "
        f"置换 p {o2['p']:.3f}   | m=3 p {o3['p'] if o3 else np.nan:.3f}")
    pl.append({"fake_year":fy,"b2":o2["beta"],"t2":o2["beta"]/o2["se_tk"],
               "p2":o2["p"],"b3":o3["beta"] if o3 else np.nan,
               "p3":o3["p"] if o3 else np.nan})
pd.DataFrame(pl).to_csv(os.path.join(OUT,"placebo.csv"), index=False, encoding="utf-8-sig")

# ---- 5 股票特定趋势 ----
log("\n" + "="*100); log("5 股票特定线性趋势"); log("="*100)
for m in [2,3,4,5,10,20]:
    a = did(d0, f"vr{m}", B=299); b = did(d0, f"vr{m}", trend=True, B=299)
    log(f"  m={m:>2}: 主 {a['beta']:+.4f}(t{a['beta']/a['se_tk']:+.2f}, p{a['p']:.3f})"
        f"  | 加趋势 {b['beta']:+.4f}(t{b['beta']/b['se_tk']:+.2f}, p{b['p']:.3f})")

# ---- 6 逐年 ----
log("\n" + "="*100); log("6 逐年 DiD"); log("="*100)
by=[]
for py in [2021,2022,2023,2024]:
    d = cell[cell.year.isin([2017,2018,2019,py])].copy()
    d["post"] = (d.year==py).astype(int)
    o = did(d,"vr2",B=299)
    log(f"  {py}: β {o['beta']:+.4f}  t {o['beta']/o['se_tk']:+.2f}  置换 p {o['p']:.3f}")
    by.append({"year":py,"b":o["beta"],"t":o["beta"]/o["se_tk"],"p":o["p"]})
pd.DataFrame(by).to_csv(os.path.join(OUT,"byyear.csv"), index=False, encoding="utf-8-sig")
log("\nDONE")
