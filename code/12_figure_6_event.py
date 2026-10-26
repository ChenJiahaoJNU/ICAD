# -*- coding: utf-8 -*-
"""
三面板图（仿四篇范文的动态效应图，但把预趋势问题如实画出）：
 (a) 事件时间缺口，全样本拟合 —— 合成控制的常规呈现
 (b) 留出式拟合（仅用 2014H1-2017H2 建权）下的缺口 —— 这才是预趋势的检验
 (c) 估计值与机制标定区间对照
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_sc")
FIG = os.path.join(BASE, "figures")
LOG = os.path.join(OUT, "event_fig.log")
open(LOG, "w").close()


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "legend.fontsize": 6.2, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False,
    "axes.spines.right": False})

R = pd.read_parquet(os.path.join(OUT, "units.parquet"))
board_of = R.groupby("ticker")["board"].first().to_dict()
PRE = list(range(2014 * 2, 2020 * 2))
POST = list(range(2021 * 2, 2025 * 2))
FIT = PRE[:8]      # 2014H1-2017H2
HOLD = PRE[8:]     # 2018H1-2019H2


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


def event_time(t):
    if t in PRE:
        return -(PRE[-1] - t) - 1
    return t - POST[0] + 1


def gaps(m, fitp, treated=None):
    cell = R.groupby(["ticker", "t"]).agg(v=("vr%d" % m, "mean"),
                                          n=("vr%d" % m, "count")).reset_index()
    cell = cell[cell.n >= 4]
    wide = cell.pivot(index="t", columns="ticker", values="v").reindex(PRE + POST)
    pc = wide.loc[PRE].notna().mean(axis=0)
    qc = wide.loc[POST].notna().mean(axis=0)
    wide = wide.loc[:, (pc >= .65) & (qc >= .65)]
    cols = list(wide.columns)
    sub = R[R.t.isin(fitp)]
    vol = sub.groupby("ticker")["vol"].mean()
    hit = sub.groupby("ticker")["hit"].mean()
    X = np.vstack([wide.loc[t].to_numpy() for t in fitp]).T
    cm = np.nanmean(X, axis=0)
    bad = np.where(~np.isfinite(X))
    X[bad] = np.take(cm, bad[1])
    X = np.column_stack([X, vol.reindex(cols).to_numpy(), hit.reindex(cols).to_numpy()])
    Z = pd.DataFrame(np.nan_to_num(X, nan=0.0), index=cols)
    ci = {c: i for i, c in enumerate(cols)}
    tr = [ci[c] for c in (treated if treated else [x for x in cols if board_of.get(x) == "gem"])
          if c in ci]
    dn = [ci[c] for c in cols if board_of.get(c) == "main"]
    z1 = Z.iloc[tr].to_numpy().mean(axis=0)
    sd = np.abs(Z.iloc[dn + tr].std(axis=0)).to_numpy()
    sd[sd == 0] = 1.0
    w = _snnls(z1, Z.iloc[dn].to_numpy().T, np.diag(1.0 / sd ** 2))
    Y = wide.to_numpy(float)
    gap = np.nanmean(Y[:, tr], axis=1) - _pred(Y[:, dn], w)
    pi = {t: i for i, t in enumerate(wide.index)}
    return gap, pi


STYLE = {2: ("#7f8c8d", "o"), 3: ("#1f4e79", "s"), 5: ("#b03a2e", "^")}
fig, axs = plt.subplots(1, 3)

# ---------- (a) 全样本拟合 ----------
a = axs[0]
for m in [2, 3, 5]:
    gap, pi = gaps(m, PRE)
    ts = PRE + POST
    colr, mk = STYLE[m]
    a.plot([event_time(t) for t in ts], [gap[pi[t]] for t in ts], "-", color=colr, lw=0.8)
    a.plot([event_time(t) for t in ts], [gap[pi[t]] for t in ts], mk, ms=2.4,
           color=colr, mfc="none", label="$m=%d$" % m)
a.axvline(0, color="0.45", lw=0.8, ls=":")
a.axhline(0, color="0.45", lw=0.6)
a.axvspan(-0.5, 0.5, color="0.86", alpha=0.6, lw=0)
a.set_xlabel("Half-years relative to reform")
a.set_ylabel("Gap: treated $-$ synthetic")
a.legend(frameon=False, ncol=3, loc="lower left", handletextpad=0.3)
a.grid(alpha=0.2, lw=0.3)
a.set_title("(a) full-sample fit", fontsize=7)

# ---------- (b) 留出式拟合（预趋势检验）----------
b = axs[1]
for m in [2, 3, 5]:
    gap, pi = gaps(m, FIT)
    ts = FIT + HOLD + POST
    colr, mk = STYLE[m]
    b.plot([event_time(t) for t in ts], [gap[pi[t]] for t in ts], "-", color=colr, lw=0.8)
    b.plot([event_time(t) for t in ts], [gap[pi[t]] for t in ts], mk, ms=2.4,
           color=colr, mfc="none")
    # 标出真正未参与拟合的期（2018H1-2019H2 与改革后）
    b.axvspan(-4.5, -0.5, color="#f4e3a1", alpha=0.35, lw=0)
b.axvline(0, color="0.45", lw=0.8, ls=":")
b.axhline(0, color="0.45", lw=0.6)
b.axvspan(-0.5, 0.5, color="0.86", alpha=0.6, lw=0)
b.set_xlabel("Half-years relative to reform")
b.set_ylabel("Gap: treated $-$ synthetic")
b.text(-2.5, b.get_ylim()[1] * 0.86, "held out\n(not fitted)", fontsize=5.4,
       ha="center", color="#8a6d0b")
b.grid(alpha=0.2, lw=0.3)
b.set_title("(b) weights from 2014H1--2017H2 only", fontsize=7)

# ---------- (c) 估计值 vs 标定区间 ----------
c = axs[2]
lo = {2: -0.030, 3: -0.040, 5: -0.049}
hi = {2: -0.021, 3: -0.038, 5: -0.061}
est = {2: -0.017, 3: -0.014, 5: -0.063}
xs = np.arange(3)
for i, m in enumerate([2, 3, 5]):
    c.vlines(i, min(lo[m], hi[m]), max(lo[m], hi[m]), color="#b03a2e", lw=7, alpha=0.32)
c.plot(xs, [est[m] for m in [2, 3, 5]], "o", ms=4.5, color="#1f4e79")
c.vlines([], [], [], color="#b03a2e", lw=7, alpha=0.32, label="predicted range")
c.plot([], [], "o", ms=4.5, color="#1f4e79", label="estimated ATT")
c.axhline(0, color="0.45", lw=0.6)
c.set_xticks(xs)
c.set_xticklabels(["$m=2$", "$m=3$", "$m=5$"])
c.set_ylabel("Post-reform change in VR")
c.legend(frameon=False, loc="lower left", handletextpad=0.4)
c.grid(alpha=0.2, lw=0.3, axis="y")
c.set_title("(c) estimate vs calibrated range", fontsize=7)

fig.set_size_inches(7.1, 2.45)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(FIG, "Fig6.pdf"))
plt.close(fig)
log("saved Fig6.pdf (3 panels)")

for m in [2, 3, 5]:
    g1, p1 = gaps(m, PRE)
    g2, p2 = gaps(m, FIT)
    log("m=%d  full-fit pre %+.4f post %+.4f | holdout-fit 2018-19 mean %+.4f, post %+.4f"
        % (m, np.mean([g1[p1[t]] for t in PRE]), np.mean([g1[p1[t]] for t in POST]),
           np.mean([g2[p2[t]] for t in HOLD]), np.mean([g2[p2[t]] for t in POST])))
