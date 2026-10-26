# -*- coding: utf-8 -*-
"""
生成四篇范文都有、而我们缺的三样证据：
  1. 描述性统计表（按板别 × 改革前后，VR(m) 的 N/mean/sd/min/p25/median/p75/max）
  2. 异质性与机制表（改革前涨停强度三分组的合成控制，path 与 segment 两种设定）
  3. 事件时间数据（改革前 12 期 + 改革后 8 期，供动态效应图使用）
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_sc")
LOG = os.path.join(OUT, "ee_tables.log")
open(LOG, "w").close()


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")


R = pd.read_parquet(os.path.join(OUT, "units.parquet"))
board_of = R.groupby("ticker")["board"].first().to_dict()
PRE = list(range(2014 * 2, 2020 * 2))
POST = list(range(2021 * 2, 2025 * 2))
MS = [2, 3, 5]

# ============================================================ 1. 描述性统计
v3 = pd.read_parquet(os.path.join(BASE, "results_v3", "panel_v3.parquet"))
rows = []
for board, blab in [("gem", "ChiNext"), ("main", "Main board")]:
    for yrs, plab in [((2014, 2019), "2014--2019"), ((2021, 2024), "2021--2024")]:
        s = v3[(v3.board == board) & (v3.year.between(*yrs))]
        for m in MS:
            x = s["vr%d" % m].dropna().to_numpy()
            if len(x) == 0:
                continue
            rows.append(dict(board=blab, period=plab, m=m, N=len(x),
                             mean=x.mean(), sd=x.std(ddof=1),
                             p25=np.quantile(x, .25), p50=np.quantile(x, .50),
                             p75=np.quantile(x, .75),
                             min=x.min(), max=x.max()))
# 涨停率与波动率也进描述性表
for board, blab in [("gem", "ChiNext"), ("main", "Main board")]:
    for yrs, plab in [((2014, 2019), "2014--2019"), ((2021, 2024), "2021--2024")]:
        s = v3[(v3.board == board) & (v3.year.between(*yrs))]
        rows.append(dict(board=blab, period=plab, m="hit (%)",
                         N=len(s), mean=s.hit_rate.mean() * 100,
                         sd=s.hit_rate.std(ddof=1) * 100,
                         p25=np.nan, p50=np.nan, p75=np.nan,
                         min=np.nan, max=np.nan))
        rows.append(dict(board=blab, period=plab, m="vol (%)",
                         N=len(s), mean=s.vol.mean() * 100,
                         sd=s.vol.std(ddof=1) * 100,
                         p25=np.nan, p50=np.nan, p75=np.nan,
                         min=np.nan, max=np.nan))

desc = pd.DataFrame(rows)
desc.to_csv(os.path.join(OUT, "tab_descriptive.csv"), index=False, encoding="utf-8-sig")
log("=== 描述性统计 ===")
for _, r in desc.iterrows():
    log("  %-11s %-10s %-9s N=%4d  mean=%.4f  sd=%.4f"
        % (r["board"], r["period"], r["m"], r["N"], r["mean"], r["sd"]))

# ============================================================ 2. 设定与拟合
def _snnls(z1, Z0, V, mu=1e3):
    A = np.vstack([np.sqrt(V) @ Z0, mu * np.ones((1, Z0.shape[1]))])
    b = np.concatenate([np.sqrt(V) @ z1, [mu]])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.ones(Z0.shape[1]) / Z0.shape[1]


def _pred(A, w):
    fin = np.isfinite(A)
    sw = (np.where(fin, w[None, :], 0.0)).sum(axis=1)
    return np.where(sw > 0, (np.where(fin, A, 0.0) @ w) / np.where(sw > 0, sw, 1.0), np.nan)


def design(m, spec="path", restrict=None):
    cell = R.groupby(["ticker", "t"]).agg(v=("vr%d" % m, "mean"),
                                          n=("vr%d" % m, "count")).reset_index()
    cell = cell[cell.n >= 4]
    wide = cell.pivot(index="t", columns="ticker", values="v").reindex(PRE + POST)
    if restrict is not None:
        wide = wide.loc[:, [c for c in wide.columns if c in restrict]]
    pc = wide.loc[PRE].notna().mean(axis=0)
    qc = wide.loc[POST].notna().mean(axis=0)
    wide = wide.loc[:, (pc >= .65) & (qc >= .65)]
    cols = list(wide.columns)
    sub = R[R.t.isin(PRE)]
    vol = sub.groupby("ticker")["vol"].mean()
    hit = sub.groupby("ticker")["hit"].mean()
    if spec == "path":
        X = np.vstack([wide.loc[t].to_numpy() for t in PRE]).T
    else:
        k = len(PRE) // 3
        segs = [PRE[0:k], PRE[k:2 * k], PRE[2 * k:]]
        X = np.vstack([wide.loc[sg].mean(axis=0, skipna=True).to_numpy() for sg in segs]).T
    cm = np.nanmean(X, axis=0)
    bad = np.where(~np.isfinite(X))
    X[bad] = np.take(cm, bad[1])
    X = np.column_stack([X, vol.reindex(cols).to_numpy(), hit.reindex(cols).to_numpy()])
    Z = pd.DataFrame(np.nan_to_num(X, nan=0.0), index=cols)
    return wide, Z


def fit_sc(m, treated, restrict=None, spec="path"):
    wide, Z = design(m, spec, restrict)
    ci = {c: i for i, c in enumerate(wide.columns)}
    tr = [ci[c] for c in treated if c in ci]
    dn = [ci[c] for c in wide.columns if board_of.get(c) == "main" and c in ci]
    if len(tr) < 10 or len(dn) < 20:
        return None
    z1 = Z.iloc[tr].to_numpy().mean(axis=0)
    sd = np.abs(Z.iloc[dn + tr].std(axis=0)).to_numpy()
    sd[sd == 0] = 1.0
    w = _snnls(z1, Z.iloc[dn].to_numpy().T, np.diag(1.0 / sd ** 2))
    Y = wide.to_numpy(float)
    gap = np.nanmean(Y[:, tr], axis=1) - _pred(Y[:, dn], w)
    pi = {t: i for i, t in enumerate(wide.index)}
    return dict(gap=gap, pi=pi, n_t=len(tr), n_d=len(dn), index=wide.index)


def placebo_p(m, nsz, obs, restrict=None, spec="path", B=1000):
    wide, Z = design(m, spec, restrict)
    ci = {c: i for i, c in enumerate(wide.columns)}
    main = [ci[c] for c in wide.columns if board_of.get(c) == "main"]
    pi = {t: i for i, t in enumerate(wide.index)}
    Y = wide.to_numpy(float)
    rng = np.random.default_rng(7000 + nsz)
    ma = np.array(main)
    pl = []
    for _ in range(B):
        pick = rng.choice(len(ma), nsz, replace=False)
        trp = list(ma[pick])
        dnp = [j for j in main if j not in set(trp)]
        z1 = Z.iloc[trp].to_numpy().mean(axis=0)
        sd = np.abs(Z.iloc[dnp + trp].std(axis=0)).to_numpy()
        sd[sd == 0] = 1.0
        w = _snnls(z1, Z.iloc[dnp].to_numpy().T, np.diag(1.0 / sd ** 2))
        g = np.nanmean(Y[:, trp], axis=1) - _pred(Y[:, dnp], w)
        pl.append(float(np.mean(g[[pi[t] for t in POST]])))
    pl = np.asarray(pl)
    return float((1 + np.sum(pl <= obs)) / (1 + len(pl)))

# ============================================================ 3. 异质性（三分组）
log("")
log("=== 异质性：改革前涨停强度三分组（path 与 segment 两种设定）===")
het = []
for spec in ["path", "segment"]:
    for m in MS:
        pre_hit = (R[(R.ticker.isin([c for c in board_of if board_of[c] == "gem"]))
                     & (R.t.isin(PRE))].groupby("ticker")["hit"].mean().dropna())
        q1, q2 = pre_hit.quantile(1 / 3), pre_hit.quantile(2 / 3)
        hi = [t for t in pre_hit.index if pre_hit[t] >= q2]
        lo = [t for t in pre_hit.index if pre_hit[t] <= q1]
        for tag, grp in [("High", hi), ("Low", lo)]:
            o = fit_sc(m, grp, spec=spec)
            if o is None:
                continue
            att = float(np.mean(o["gap"][[o["pi"][t] for t in POST]]))
            p = placebo_p(m, o["n_t"], att, spec=spec)
            het.append(dict(spec=spec, m=m, group=tag, n=o["n_t"], att=att, p=p))
            log("  %-7s m=%d %-4s n=%3d  ATT %+.4f  p=%.3f" % (spec, m, tag,
                                                              o["n_t"], att, p))
pd.DataFrame(het).to_csv(os.path.join(OUT, "tab_heterogeneity.csv"),
                         index=False, encoding="utf-8-sig")

# ============================================================ 4. 事件时间
log("")
log("=== 事件时间（path 设定，全样本）===")
ev = []
for m in MS:
    wide, Z = design(m, "path")
    gem = [c for c in wide.columns if board_of.get(c) == "gem"]
    o = fit_sc(m, gem, spec="path")
    for t in PRE + POST:
        ev.append(dict(m=m, t=t,
                       label="%dH%d" % (t // 2, 1 if t % 2 == 0 else 2),
                       event_time=(t - 2020 * 2) if t >= 2020 * 2 else (t - 2019 * 2 - 1),
                       phase="pre" if t in PRE else "post",
                       gap=float(o["gap"][o["pi"][t]])))
evd = pd.DataFrame(ev)
evd.to_csv(os.path.join(OUT, "tab_event_time.csv"), index=False, encoding="utf-8-sig")
log("  wrote tab_event_time.csv (%d rows)" % len(evd))
log("")
log("wrote tab_descriptive.csv / tab_heterogeneity.csv / tab_event_time.csv")
