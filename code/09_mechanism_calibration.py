# -*- coding: utf-8 -*-
"""
最终版延迟机制模拟：多种子 × 多延迟结构，报告均值与区间。
自由度按实测截断率标定（nu≈4）；nu 的四阶矩不存在，VR 估计的蒙特卡洛噪声不可忽略，
因此必须报告跨种子分布而不是单点。
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats, optimize

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_verify")
LOG = os.path.join(OUT, "deferral_final.log")
open(LOG, "w").close()


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")


# ---------- 1. 实测标定 ----------
raw = pd.read_parquet(os.path.join(BASE, "data", "panel_merged.parquet"))
raw["d"] = pd.to_datetime(raw["date"])
raw["board"] = np.where(raw.code.str.startswith("sz.30"), "gem", "main")
raw = raw[raw.tradestatus.astype(str).isin(["1", "1.0"]) & raw.preclose.gt(0)].copy()
raw["r"] = np.log(raw.close.astype(float) / raw.preclose.astype(float))
raw["year"] = raw.d.dt.year
pre = raw[(raw.board == "gem") & raw.year.between(2017, 2019)]
sig = float(pre.r.std())
target = float(np.mean(np.abs(pre.r.to_numpy()) > 0.10))

f = lambda nu: 2.0 * stats.t.sf((0.10 / sig) * np.sqrt(nu / (nu - 2.0)), nu) - target
nu = float(optimize.brentq(f, 3.05, 60))
log("Calibration: sigma = %.5f, observed P(|r|>10%%) = %.5f, fitted nu = %.3f"
    % (sig, target, nu))
log("           model P(|r|>10%%) = %.5f, P(|r|>20%%) = %.5f"
    % (2 * stats.t.sf((0.10 / sig) * np.sqrt(nu / (nu - 2)), nu),
       2 * stats.t.sf((0.20 / sig) * np.sqrt(nu / (nu - 2)), nu)))


def draw(n, nu, sigma, g):
    return sigma * g.standard_t(nu, n) / np.sqrt(nu / (nu - 2.0))


def vr(r, m):
    v1 = np.var(r[1:], ddof=0)
    rm = np.convolve(r, np.ones(m), "valid")
    return float(np.var(rm, ddof=0) / (m * v1))


# ---------- 2. 模拟 ----------
N = 600000
SEEDS = 20
MS = [2, 3, 5, 10, 20]
SCHEMES = [("truncation only", None),
           ("deferral, 1 session", [1.0]),
           ("deferral, 2 sessions", [0.5, 0.5]),
           ("deferral, 4 sessions", [0.4, 0.3, 0.2, 0.1])]

levels, deltas = {}, {}
for name, weights in SCHEMES:
    L = {0.10: [], 0.20: []}
    for s in range(SEEDS):
        g = np.random.default_rng(7000 + s)
        for a in (0.10, 0.20):
            eps = draw(N, nu, sig, g)
            cl = np.clip(eps, -a, a)
            r = cl.copy() if weights is None else cl.copy()
            if weights is not None:
                spill = eps - cl
                for k, wgt in enumerate(weights, start=1):
                    r[k:] += wgt * spill[:-k]
            L[a].append([vr(r, m) for m in MS])
    L = {a: np.asarray(v) for a, v in L.items()}
    levels[name] = L
    d = L[0.20] - L[0.10]
    deltas[name] = d
    log("")
    log("%-22s  level at 10%%: %s" % (
        name, " ".join("VR(%d)=%.4f" % (m, L[0.10][:, i].mean())
                       for i, m in enumerate(MS))))
    log("%-22s  predicted dVR: %s" % (
        name, " ".join("m=%d %+.4f [%+.4f,%+.4f]" % (
            m, d[:, i].mean(), np.quantile(d[:, i], 0.05), np.quantile(d[:, i], 0.95))
            for i, m in enumerate(MS))))

log("")
log("=" * 92)
log("PANEL B levels (mean over %d seeds, N=%d each)" % (SEEDS, N))
log("%-22s %8s " % ("scheme", "band") + " ".join("%9s" % ("VR(%d)" % m) for m in MS))
rows = []
for name in levels:
    for a in (0.10, 0.20):
        v = levels[name][a].mean(axis=0)
        log("%-22s %8.2f " % (name, a) + " ".join("%9.4f" % x for x in v))
        rows.append(dict(scheme=name, band=a,
                         **{"VR%d" % m: x for m, x in zip(MS, v)}))

log("")
log("=" * 92)
log("PANEL C predicted change from widening 10%% -> 20%% (mean [5%%,95%%] over seeds)")
log("%-22s " % "scheme" + " ".join("%26s" % ("m=%d" % m) for m in MS))
for name in deltas:
    d = deltas[name]
    log("%-22s " % name + " ".join(
        "%26s" % ("%+.4f [%+.4f,%+.4f]" % (d[:, i].mean(),
                                           np.quantile(d[:, i], .05),
                                           np.quantile(d[:, i], .95)))
        for i in range(len(MS))))

# 机制能产生的总区间（跨延迟结构与种子）
allm = {m: np.concatenate([deltas[n][:, i] for n in deltas]) for i, m in enumerate(MS)}
log("")
log("Overall range of the mechanism's prediction (all lag structures x seeds):")
for i, m in enumerate(MS):
    x = allm[m]
    log("  m=%2d : mean %+.4f   5%%-95%% [%+.4f, %+.4f]   min %+.4f  max %+.4f"
        % (m, x.mean(), np.quantile(x, .05), np.quantile(x, .95), x.min(), x.max()))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "deferral_final.csv"),
                          index=False, encoding="utf-8-sig")
json.dump({"nu": nu, "sigma": sig, "censor_obs": target, "seeds": SEEDS, "N": N,
           "mean": {n: deltas[n].mean(axis=0).tolist() for n in deltas},
           "q05": {n: np.quantile(deltas[n], .05, axis=0).tolist() for n in deltas},
           "q95": {n: np.quantile(deltas[n], .95, axis=0).tolist() for n in deltas},
           "overall": {str(m): [float(allm[m].min()), float(allm[m].max())]
                       for i, m in enumerate(MS)}},
          open(os.path.join(OUT, "deferral_final.json"), "w", encoding="utf-8"), indent=2)
log("")
log("wrote deferral_final.csv / deferral_final.json")
