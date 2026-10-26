# -*- coding: utf-8 -*-
"""用当前数据（444 只、panel_v3、path 设定）重画全部 5 张图。"""
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_sc")
VER = os.path.join(BASE, "results_verify")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = os.path.join(VER, "figures_v3.log")
open(LOG, "w").close()


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False,
    "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    log("saved " + name)


# ======================= fig1: DiD 系数按期限扫描 =======================
v3 = pd.read_parquet(os.path.join(BASE, "results_v3", "panel_v3.parquet"))


def twoway(x, ti, yi, nt, ny, iters=60, tol=1e-13):
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        old = x.copy()
        s = np.bincount(ti, weights=x, minlength=nt); c = np.bincount(ti, minlength=nt)
        x -= (s / np.maximum(c, 1))[ti]
        s = np.bincount(yi, weights=x, minlength=ny); c = np.bincount(yi, minlength=ny)
        x -= (s / np.maximum(c, 1))[yi]
        if np.max(np.abs(x - old)) < tol:
            break
    return x


PRE, POST = [2017, 2018, 2019], [2021, 2022, 2023]
d = v3[v3.year.isin(PRE + POST)].copy()
d["treat"] = (d.board == "gem").astype(int)
ti = pd.Categorical(d.ticker).codes
yi = pd.Categorical(d.year).codes
nt, ny = ti.max() + 1, yi.max() + 1
D = ((d.treat == 1) & (d.year.isin(POST))).astype(float).to_numpy()
Dy = twoway(D, ti, yi, nt, ny)
den = float(Dy @ Dy)

hs, bs, ses = [], [], []
for m in range(2, 31):
    col = "vr%d" % m
    if col not in d.columns:
        continue
    yy = twoway(d[col].to_numpy(), ti, yi, nt, ny)
    b = float(Dy @ yy) / den
    u = yy - b * Dy
    Zu = np.bincount(ti, weights=Dy * u, minlength=nt)
    G = len(Zu)
    se = float(np.sqrt((Zu @ Zu) / den ** 2 * G / (G - 1.0)))
    hs.append(m); bs.append(b); ses.append(se)
hs, bs, ses = np.array(hs), np.array(bs), np.array(ses)
log("fig1: %d horizons, beta range %+.4f..%+.4f" % (len(hs), bs.min(), bs.max()))

fig, ax = plt.subplots()
ax.axhline(0, color="0.4", lw=0.7)
ax.fill_between(hs, bs - 1.96 * ses, bs + 1.96 * ses, color="#1f4e79", alpha=0.15, lw=0)
ax.plot(hs, bs, "-o", ms=2.6, color="#1f4e79")
sig = np.abs(bs / ses) > 1.96
ax.plot(hs[sig], bs[sig], "o", ms=4.2, mfc="none", color="#b03a2e")
ax.set_xlabel("Horizon $m$ (trading days)")
ax.set_ylabel(r"DiD coefficient on $\mathrm{VR}(m)$")
ax.grid(alpha=0.25, lw=0.4)
ax.text(0.97, 0.06, "open circles: 95% CI excludes zero",
        transform=ax.transAxes, ha="right", fontsize=6, color="#b03a2e")
fig.set_size_inches(3.3, 2.4)
save(fig, "fig1_horizon_scan.pdf")

# ======================= fig2: 波动率十分位的限幅频率 =======================
vd = json.load(open(os.path.join(VER, "recount.json"), encoding="utf-8"))
vol = vd["vol_deciles"]
ks = sorted(int(k) for k in vol.keys())
vv = [vol[str(k)]["vol"] for k in ks]
uu = [vol[str(k)]["up"] for k in ks]
dd_ = [vol[str(k)]["dn"] for k in ks]
log("fig2: up %.3f -> %.3f" % (uu[0], uu[-1]))

fig, ax = plt.subplots()
x = np.arange(len(ks))
ax.plot(x, uu, "-o", ms=3, color="#b03a2e", label="Limit-up")
ax.plot(x, dd_, "-s", ms=3, mfc="none", color="#1f4e79", label="Limit-down")
ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("Pre-event volatility decile (1 = calmest, 10 = most volatile)")
ax.set_ylabel("Share of traded days sealed at the limit (\\%)")
ax.legend(frameon=False)
ax.grid(alpha=0.25, lw=0.4)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(x)
ax2.set_xticklabels(["%.1f" % v for v in vv], fontsize=5.5)
ax2.set_xlabel("Mean volatility in decile (\\% per session)", fontsize=6.5)
for sp in ax2.spines.values():
    sp.set_visible(False)
fig.set_size_inches(3.4, 2.3)
save(fig, "fig2_censoring_by_vol.pdf")

# ======================= fig3: VR 期限结构（板别 × 前后）=======================
fig, ax = plt.subplots()
ms = [m for m in range(2, 41) if "vr%d" % m in v3.columns]
for board, colr, lab in [("gem", "#b03a2e", "ChiNext"), ("main", "#1f4e79", "Main board")]:
    for per, ls, tag in [((2014, 2019), "--", "2014--19"), ((2021, 2024), "-", "2021--24")]:
        s = v3[(v3.board == board) & (v3.year.between(*per))]
        means = [s["vr%d" % m].mean() for m in ms]
        ax.plot(ms, means, ls, lw=1.0, color=colr,
                label="%s, %s" % (lab, tag))
ax.axhline(1.0, color="0.4", lw=0.7, ls=":")
ax.set_xlabel("Horizon $m$ (trading days)")
ax.set_ylabel(r"Mean $\mathrm{VR}(m)$")
ax.legend(frameon=False, ncol=1, loc="upper right")
ax.grid(alpha=0.25, lw=0.4)
fig.set_size_inches(3.4, 2.5)
save(fig, "fig3_vr_profile.pdf")

# ======================= SC：path 设定重跑，供 fig4/fig5 =======================
PRE_H = list(range(2014 * 2, 2020 * 2))
POST_H = list(range(2021 * 2, 2025 * 2))
R = pd.read_parquet(os.path.join(OUT, "units.parquet"))
board_of = R.groupby("ticker")["board"].first().to_dict()


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


def build(m):
    cell = R.groupby(["ticker", "t"]).agg(v=(f"vr{m}", "mean"),
                                          n=(f"vr{m}", "count")).reset_index()
    cell = cell[cell.n >= 4]
    wide = cell.pivot(index="t", columns="ticker", values="v").reindex(PRE_H + POST_H)
    pc = wide.loc[PRE_H].notna().mean(axis=0); qc = wide.loc[POST_H].notna().mean(axis=0)
    wide = wide.loc[:, (pc >= .65) & (qc >= .65)]
    gem = [c for c in wide.columns if board_of.get(c) == "gem"]
    main = [c for c in wide.columns if board_of.get(c) == "main"]
    sub = R[R.t.isin(PRE_H)]
    vol = sub.groupby("ticker")["vol"].mean(); hit = sub.groupby("ticker")["hit"].mean()
    cols = list(wide.columns)
    X = np.vstack([wide.loc[t].to_numpy() for t in PRE_H]).T
    cm = np.nanmean(X, axis=0); bad = np.where(~np.isfinite(X)); X[bad] = np.take(cm, bad[1])
    X = np.column_stack([X, vol.reindex(cols).to_numpy(), hit.reindex(cols).to_numpy()])
    Z = pd.DataFrame(np.nan_to_num(X, nan=0.0), index=cols)
    return wide, Z, gem, main


def fit(wide, Z, tr, dn):
    z1 = Z.iloc[tr].to_numpy().mean(axis=0)
    sd = np.abs(Z.iloc[list(dn) + list(tr)].std(axis=0)).to_numpy(); sd[sd == 0] = 1.0
    w = _snnls(z1, Z.iloc[dn].to_numpy().T, np.diag(1.0 / sd ** 2))
    Y = wide.to_numpy(float)
    gap = np.nanmean(Y[:, tr], axis=1) - _pred(Y[:, dn], w)
    pi = {t: i for i, t in enumerate(wide.index)}
    return gap, pi, w


store = {}
for m in [2, 3, 5]:
    wide, Z, gem, main = build(m)
    ci = {c: i for i, c in enumerate(wide.columns)}
    gi = [ci[c] for c in gem]; mi = [ci[c] for c in main]
    gap, pi, w = fit(wide, Z, gi, mi)
    store[m] = dict(gap=gap, pi=pi, wide=wide, att=float(np.mean(gap[[pi[t] for t in POST_H]])))
    log("fig set m=%d: ATT %+.4f, effN %.1f" % (m, store[m]["att"],
        1.0 / float(np.sum(w ** 2))))

# 安慰剂分布（m=5）
m = 5
wide, Z, gem, main = build(m)
ci = {c: i for i, c in enumerate(wide.columns)}
gi = [ci[c] for c in gem]; mi = [ci[c] for c in main]
gap, pi, _ = fit(wide, Z, gi, mi)
rng = np.random.default_rng(7000 + len(gem))
ma = np.array(mi)
pl = []
for _ in range(2000):
    pick = rng.choice(len(ma), len(gi), replace=False)
    dn = [j for j in ma if j not in set(ma[pick])]
    g2, pi2, _ = fit(wide, Z, list(ma[pick]), dn)
    pl.append(float(np.mean(g2[[pi2[t] for t in POST_H]])))
pl = np.asarray(pl)
log("fig4: placebo mean %+.4f sd %.4f" % (pl.mean(), pl.std()))

# ======================= fig4: 推断诊断 =======================
fig, axs = plt.subplots(1, 2)
a = axs[0]
a.hist(pl, bins=45, color="#1f4e79", alpha=0.55, lw=0)
a.axvline(store[5]["att"], color="#b03a2e", lw=1.4)
a.text(store[5]["att"], a.get_ylim()[1] * 0.92, "  treated\n  %.3f" % store[5]["att"],
       color="#b03a2e", fontsize=6, va="top")
a.set_xlabel("Post-reform gap of placebo group")
a.set_ylabel("Count")
a.set_title("(a) 2,000 aggregate placebos, $m=5$", fontsize=7)
a.grid(alpha=0.2, lw=0.3)

b = axs[1]
for mm, colr in [(2, "#7f8c8d"), (3, "#1f4e79"), (5, "#b03a2e")]:
    s = store[mm]
    g = [s["gap"][s["pi"][t]] for t in POST_H]
    b.plot(range(len(POST_H)), g, "-o", ms=2.2, color=colr, label="$m=%d$" % mm)
b.axhline(0, color="0.35", lw=0.7)
b.set_xticks(range(len(POST_H)))
b.set_xticklabels(["%dH%d" % (t // 2, 1 if t % 2 == 0 else 2) for t in POST_H],
                  rotation=45, fontsize=5.5)
b.set_ylabel("Gap (treated $-$ synthetic)")
b.legend(frameon=False, fontsize=6)
b.set_title("(b) post-reform gap by half-year", fontsize=7)
b.grid(alpha=0.2, lw=0.3)
fig.set_size_inches(7.1, 2.5)
fig.tight_layout()
save(fig, "fig4_diagnostics.pdf")

# ======================= fig5: 处理组 vs 合成控制 =======================
fig, axs = plt.subplots(1, 2)
a = axs[0]
s = store[5]
wide = s["wide"]
ci = {c: i for i, c in enumerate(wide.columns)}
gem = [c for c in wide.columns if board_of.get(c) == "gem"]
main = [c for c in wide.columns if board_of.get(c) == "main"]
gi = [ci[c] for c in gem]; mi = [ci[c] for c in main]
gap, pi, w = fit(wide, Z, gi, mi)
Y = wide.to_numpy(float)
treated = np.nanmean(Y[:, gi], axis=1)
synth = np.nanmean(Y[:, mi], axis=1) - gap
tt = PRE_H + POST_H
a.plot(tt, [treated[pi[t]] for t in tt], "-o", ms=2, color="#b03a2e",
       label="ChiNext (treated)")
a.plot(tt, [synth[pi[t]] for t in tt], "--s", ms=2, mfc="none", color="#1f4e79",
       label="Synthetic control")
a.axvline(2019 * 2 + 1.5, color="0.5", lw=0.8, ls=":")
a.set_xlabel("Half-year")
a.set_ylabel(r"VR($m=5$)")
a.set_xticks([2014 * 2, 2016 * 2, 2018 * 2, 2022 * 2, 2024 * 2])
a.set_xticklabels(["2014", "2016", "2018", "2022", "2024"], fontsize=6)
a.legend(frameon=False, loc="lower left")
a.grid(alpha=0.25, lw=0.4)
a.set_title("(a) treated against synthetic, $m=5$", fontsize=7)

b = axs[1]
pred = {2: -0.031, 3: -0.042, 5: -0.052}
xs = np.arange(3)
vals = [store[mm]["att"] for mm in [2, 3, 5]]
b.bar(xs - 0.18, vals, width=0.34, color="#1f4e79", label="estimated ATT")
b.bar(xs + 0.18, [pred[mm] for mm in [2, 3, 5]], width=0.34, color="#b03a2e",
      alpha=0.55, label="calibrated bound")
b.axhline(0, color="0.35", lw=0.7)
b.set_xticks(xs); b.set_xticklabels(["$m=2$", "$m=3$", "$m=5$"])
b.set_ylabel("Post-reform change in VR")
b.legend(frameon=False, fontsize=6)
b.grid(alpha=0.2, lw=0.3, axis="y")
b.set_title("(b) estimate against the mechanism's bound", fontsize=7)
fig.set_size_inches(7.1, 2.6)
fig.tight_layout()
save(fig, "fig5_sc_gap.pdf")

log("DONE")

