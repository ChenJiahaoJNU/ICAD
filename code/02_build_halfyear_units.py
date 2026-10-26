# -*- coding: utf-8 -*-
"""合成控制 v3：单纯形拟合改用 NNLS+罚项（快），空间安慰剂抽样 150 个"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "results_sc")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "sc3.log")
open(LOG, "w").close()
def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

CACHE = os.path.join(OUT, "units.parquet")
if os.path.exists(CACHE):
    R = pd.read_parquet(CACHE)
    log(f"[缓存] 窗口级 {len(R)}")
else:
    ct = pd.read_parquet(os.path.join(DATA, "panel_merged.parquet"))
    for c in ["close","preclose"]:
        ct[c] = pd.to_numeric(ct[c], errors="coerce")
    ct["tradestatus"] = pd.to_numeric(ct["tradestatus"], errors="coerce")
    ct["isST"] = pd.to_numeric(ct["isST"], errors="coerce").fillna(0).astype(int)
    ct["date"] = pd.to_datetime(ct["date"])
    ct = ct[(ct.tradestatus==1)&(ct.preclose>0)&(ct.close>0)].copy()
    ct["board"] = ct.code.str[:5].map({"sh.60":"main","sz.00":"main","sz.30":"gem","sh.68":"star"})
    ct = ct.sort_values(["code","date"]).reset_index(drop=True)
    ct["ret"] = np.log(ct.close/ct.preclose)
    G = pd.Timestamp("2020-08-24")
    ct["band"] = np.where((ct.board=="gem")&(ct.date>=G),0.20,np.where(ct.isST==1,0.05,0.10))
    ct["lim"] = np.round(ct.preclose*(1+ct.band),2)
    ct["hit"] = (ct.close>=ct.lim-1e-6).astype(int)
    cal=np.sort(ct["date"].unique()); cim={d:i for i,d in enumerate(cal)}; ND=len(cal)
    ct["ci"]=ct["date"].map(cim)
    bo=ct.groupby("code")["board"].first().to_dict()
    W,ST,MS=125,4,[2,3,5]; rows=[]
    for t,g in ct.groupby("code",sort=False):
        idx=g["ci"].to_numpy()
        x=np.full(ND,np.nan); bd=np.full(ND,np.nan); hh=np.zeros(ND)
        x[idx]=g["ret"].to_numpy(); bd[idx]=g["band"].to_numpy(); hh[idx]=g["hit"].to_numpy()
        cf=np.concatenate([[0],np.cumsum(np.isfinite(x))]); nw=ND-W+1
        if nw<=ST: continue
        st=np.arange(0,nw,ST); st=st[(cf[st+W]-cf[st])==W]
        if st.size==0: continue
        sw=np.lib.stride_tricks.sliding_window_view(x,W)[st]; v1=sw.var(axis=1)
        bw=np.lib.stride_tricks.sliding_window_view(bd,W)[st]
        bok=np.isfinite(bw).all(axis=1)&(np.nanmax(bw,axis=1)-np.nanmin(bw,axis=1)<1e-12)
        keep=(v1>0)&bok
        if keep.sum()==0: continue
        mid=pd.DatetimeIndex(cal[st+W//2]); tt=mid.year*2+(mid.month>6).astype(int)
        ch=np.concatenate([[0],np.cumsum(hh)]); hitr=(ch[st+W]-ch[st])/W; vol=sw.std(axis=1)
        vr={}
        for m in MS:
            ys=pd.Series(x).rolling(m,min_periods=m).sum().to_numpy()
            yy=np.lib.stride_tricks.sliding_window_view(ys,W-m+1)[st]
            good=np.isfinite(yy).all(axis=1)
            vr[m]=np.where(good,yy.var(axis=1)/(m*v1),np.nan); keep&=good
        for j in np.where(keep)[0]:
            r={"ticker":t,"t":int(tt[j]),"board":bo[t],"hit":hitr[j],"vol":vol[j]}
            for m in MS: r[f"vr{m}"]=vr[m][j]
            rows.append(r)
    R=pd.DataFrame(rows); R.to_parquet(CACHE,index=False); log(f"窗口级 {len(R)}")

PRE_T=list(range(2014*2,2020*2)); POST_T=list(range(2021*2,2025*2))
MS=[2,3,5]
SEG=[(2014*2,2015*2+1),(2016*2,2017*2+1),(2018*2,2019*2+1)]
board_of=R.groupby("ticker")["board"].first().to_dict()

def simplex_nnls(z1, Z0, V, mu=1e3):
    A = np.vstack([np.sqrt(V)@Z0, mu*np.ones((1,Z0.shape[1]))])
    b = np.concatenate([np.sqrt(V)@z1, [mu]])
    w,_ = nnls(A,b)
    s=w.sum()
    return w/s if s>0 else np.ones(Z0.shape[1])/Z0.shape[1]

def build(m):
    col=f"vr{m}"
    cell=R.groupby(["ticker","t"]).agg(v=(col,"mean"),n=(col,"count")).reset_index()
    cell=cell[cell.n>=4]
    wide=cell.pivot(index="t",columns="ticker",values="v").reindex(PRE_T+POST_T)
    wide=wide.loc[:, wide.notna().all(axis=0)]
    pred={}
    for tk in wide.columns:
        s=wide[tk]; vals=[s.loc[[t for t in PRE_T if a<=t<=b]].mean() for a,b in SEG]
        sub=R[(R.ticker==tk)&(R.t.isin(PRE_T))]
        vals += [sub.vol.mean(), sub.hit.mean()]
        pred[tk]=vals
    Z=pd.DataFrame(pred,index=["vr1","vr2","vr3","vol","hit"]).T.dropna()
    return wide, Z

def synth(wide,Z,treated,donors,pre,post):
    donors=[d for d in donors if d in Z.index and d in wide.columns]
    treated=[x for x in treated if x in Z.index and x in wide.columns]
    if len(donors)<8 or len(treated)<1: return None
    z1=Z.loc[treated].mean(axis=0).to_numpy()
    Z0=Z.loc[donors].to_numpy().T
    sd=np.abs(Z.loc[donors+treated].std(axis=0)).replace(0,1).to_numpy()
    w=simplex_nnls(z1,Z0,np.diag(1.0/sd**2))
    y=wide[treated].mean(axis=1); X=wide[donors]
    yp=y.loc[pre].to_numpy(); Xp=X.loc[pre].to_numpy()
    yq=y.loc[post].to_numpy(); Xq=X.loc[post].to_numpy()
    fp=Xp@w; fq=Xq@w
    pre_rm=float(np.sqrt(np.nanmean((yp-fp)**2))); post_rm=float(np.sqrt(np.nanmean((yq-fq)**2)))
    return dict(w=w,donors=donors,pre_rm=pre_rm,post_rm=post_rm,
                ratio=post_rm/max(pre_rm,1e-9),att=float(np.nanmean(yq-fq)),
                nz=int((w>1e-3).sum()))

log("\n"+"="*100); log("合成控制（NNLS 单纯形，预测变量匹配）"); log("="*100)
summary=[]
rng=np.random.default_rng(7)
for m in MS:
    wide,Z=build(m)
    gem=[c for c in wide.columns if board_of.get(c)=="gem"]
    main=[c for c in wide.columns if board_of.get(c)=="main"]
    o=synth(wide,Z,gem,main,PRE_T,POST_T)
    if o is None: log(f"  m={m}: 不可估"); continue
    log(f"\n  m={m}: 处理 {len(gem)} 只, 捐赠 {len(o['donors'])} 只, 有效权重 {o['nz']} 个")
    log(f"       事前 RMSPE {o['pre_rm']:.5f} | 事后 RMSPE {o['post_rm']:.5f} | "
        f"比值 {o['ratio']:.3f} | ATT {o['att']:+.4f}")
    pool=list(o["donors"]); 
    sample = pool if len(pool)<=150 else list(rng.choice(pool,150,replace=False))
    ratios=[]
    for k,d in enumerate(sample):
        dn=[x for x in pool if x!=d]
        oj=synth(wide,Z,[d],dn,PRE_T,POST_T)
        if oj is not None and np.isfinite(oj["ratio"]): ratios.append(oj["ratio"])
        if (k+1)%50==0: log(f"       ...空间安慰剂 {k+1}/{len(sample)}")
    ratios=np.array(ratios)
    if len(ratios):
        p=float(np.mean(ratios>=o["ratio"]))
        log(f"       空间安慰剂 n={len(ratios)}: 中位 {np.quantile(ratios,.5):.3f} "
            f"90% {np.quantile(ratios,.9):.3f} 最大 {ratios.max():.3f}")
        log(f"       => **p = {p:.3f}**  {'✅ 显著' if p<0.10 else '❌ 不显著'}")
        summary.append({"m":m,"pre_rm":o["pre_rm"],"post_rm":o["post_rm"],
                        "ratio":o["ratio"],"att":o["att"],"p":p})
    PRE2=list(range(2014*2,2017*2)); POST2=list(range(2017*2,2020*2))
    o2=synth(wide,Z,gem,main,PRE2,POST2)
    if o2 is not None:
        s2=list(o2["donors"])[:80]
        r2=[]
        for d in s2:
            dn=[x for x in o2["donors"] if x!=d]
            oj=synth(wide,Z,[d],dn,PRE2,POST2)
            if oj is not None and np.isfinite(oj["ratio"]): r2.append(oj["ratio"])
        r2=np.array(r2); p2=float(np.mean(r2>=o2["ratio"])) if len(r2) else np.nan
        log(f"       时间安慰剂(假2016): 比值 {o2['ratio']:.3f}, ATT {o2['att']:+.4f}, "
            f"p {p2:.3f} {'⚠️ 显著 —— pre-trend 仍在' if p2<0.10 else '✓ 不显著'}")
if summary:
    pd.DataFrame(summary).to_csv(os.path.join(OUT,"sc_summary.csv"),index=False,encoding="utf-8-sig")
log("\nDONE")
