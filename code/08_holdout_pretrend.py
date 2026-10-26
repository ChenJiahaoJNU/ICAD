# -*- coding: utf-8 -*-
"""
留出式合成控制的正式推断：
权重只用 2014H1-2017H2 拟合，改革前 2018H1-2019H2 完全未参与，
因此它同时提供 (i) 一个未被预趋势污染的效应估计 与 (ii) 一个可检验的预趋势诊断。

对每个期限报告：
  - 全样本拟合 ATT（常规呈现）
  - 留出式拟合 ATT（诚实估计）
  - 留出期预趋势缺口（诊断）
  - 两者各自的聚合安慰剂 p 值
  - 「差值」检验：post gap − pre-trend gap，并对该差值做安慰剂推断
    这是关键：它把任何残留预趋势差掉，直接检验改革后是否偏离了改革前的偏离。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_sc")
LOG = os.path.join(OUT, "holdout_sc.log")
open(LOG, "w").close()


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")


R = pd.read_parquet(os.path.join(OUT, "units.parquet"))
board_of = R.groupby("ticker")["board"].first().to_dict()
PRE = list(range(2014 * 2, 2020 * 2))
POST = list(range(2021 * 2, 2025 * 2))
FIT = PRE[:8]
HOLD = PRE[8:]
MS = [2, 3, 5]
B = 2000


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


rows = []
for m in MS:
    cell = R.groupby(["ticker", "t"]).agg(v=("vr%d" % m, "mean"),
                                          n=("vr%d" % m, "count")).reset_index()
    cell = cell[cell.n >= 4]
    wide = cell.pivot(index="t", columns="ticker", values="v").reindex(PRE + POST)
    pc = wide.loc[PRE].notna().mean(axis=0)
    qc = wide.loc[POST].notna().mean(axis=0)
    wide = wide.loc[:, (pc >= .65) & (qc >= .65)]
    cols = list(wide.columns)
    ci = {c: i for i, c in enumerate(cols)}
    Y = wide.to_numpy(float)
    pi = {t: i for i, t in enumerate(wide.index)}
    gem = [ci[c] for c in cols if board_of.get(c) == "gem"]
    main = [ci[c] for c in cols if board_of.get(c) == "main"]

    def Zmat(fitp):
        sub = R[R.t.isin(fitp)]
        vol = sub.groupby("ticker")["vol"].mean()
        hit = sub.groupby("ticker")["hit"].mean()
        X = np.vstack([wide.loc[t].to_numpy() for t in fitp]).T
        cm = np.nanmean(X, axis=0)
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(cm, bad[1])
        X = np.column_stack([X, vol.reindex(cols).to_numpy(), hit.reindex(cols).to_numpy()])
        return pd.DataFrame(np.nan_to_num(X, nan=0.0), index=cols)

    def stats(tr, dn, Z):
        z1 = Z.iloc[tr].to_numpy().mean(axis=0)
        sd = np.abs(Z.iloc[dn + tr].std(axis=0)).to_numpy()
        sd[sd == 0] = 1.0
        w = _snnls(z1, Z.iloc[dn].to_numpy().T, np.diag(1.0 / sd ** 2))
        gap = np.nanmean(Y[:, tr], axis=1) - _pred(Y[:, dn], w)
        post = float(np.mean(gap[[pi[t] for t in POST]]))
        hold = float(np.mean(gap[[pi[t] for t in HOLD]]))
        return post, hold

    # ---- 全样本拟合 ----
    Zfull = Zmat(PRE)
    post_f, hold_f = stats(gem, main, Zfull)
    # ---- 留出式拟合 ----
    Zho = Zmat(FIT)
    post_h, hold_h = stats(gem, main, Zho)

    # ---- 安慰剂：同规模聚合组，两种拟合各跑一遍 ----
    rng = np.random.default_rng(7000 + len(gem))
    ma = np.array(main)
    nsz = len(gem)
    pl_pf, pl_ph, pl_d = [], [], []
    for _ in range(B):
        pick = rng.choice(len(ma), nsz, replace=False)
        trp = list(ma[pick])
        dnp = [j for j in main if j not in set(trp)]
        a, b_ = stats(trp, dnp, Zfull)
        c, d = stats(trp, dnp, Zho)
        pl_pf.append(a); pl_ph.append(c); pl_d.append(c - d)

    pl_pf = np.asarray(pl_pf); pl_ph = np.asarray(pl_ph); pl_d = np.asarray(pl_d)
    p_full = float((1 + np.sum(pl_pf <= post_f)) / (1 + len(pl_pf)))
    p_ho = float((1 + np.sum(pl_ph <= post_h)) / (1 + len(pl_ph)))
    diff = post_h - hold_h
    p_diff = float((1 + np.sum(pl_d <= diff)) / (1 + len(pl_d)))

    rows.append(dict(m=m, n_treated=len(gem), n_donors=len(main),
                     post_full=post_f, post_holdout=post_h,
                     pretrend=hold_h, diff=diff,
                     p_full=p_full, p_holdout=p_ho, p_diff=p_diff))
    log("")
    log("m=%d  处理 %d / donor %d" % (m, len(gem), len(main)))
    log("  全样本拟合   post %+.4f   p=%.4f" % (post_f, p_full))
    log("  留出式拟合   post %+.4f   p=%.4f" % (post_h, p_ho))
    log("              预趋势(2018-19) %+.4f" % hold_h)
    log("  差值 post-pre %+.4f        p=%.4f   <<< 差分掉预趋势后的净效应"
        % (diff, p_diff))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "sc_holdout_inference.csv"),
                          index=False, encoding="utf-8-sig")
log("")
log("wrote sc_holdout_inference.csv")
