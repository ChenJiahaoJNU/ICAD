# -*- coding: utf-8 -*-
"""
重算全部 DiD 数字，确保 tab:did-fail 与 tab:window 相互一致且与 §5 正文一致。
方法：FWL 双向去均值 + 单参数聚类标准误 + 整体打乱板别的随机化推断。
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_verify")
LOG = os.path.join(OUT, "did_all.log")
open(LOG, "w").close()


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")


v3 = pd.read_parquet(os.path.join(BASE, "results_v3", "panel_v3.parquet"))
v3["y"] = v3.hit_rate * 0.0     # placeholder, replaced per horizon below


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


def did(d, var, post_years, rng=None, reps=0):
    """d 已含 y 与 treat；返回 beta, se, t, p。"""
    ti = pd.Categorical(d.ticker).codes
    yi = pd.Categorical(d.year).codes
    nt, ny = ti.max() + 1, yi.max() + 1
    D = ((d.treat == 1) & (d.year.isin(post_years))).astype(float).to_numpy()
    y = d[var].to_numpy()
    Dy = twoway(D, ti, yi, nt, ny)
    yy = twoway(y, ti, yi, nt, ny)
    den = float(Dy @ Dy)
    b = float(Dy @ yy) / den
    u = yy - b * Dy
    Zu = np.bincount(ti, weights=Dy * u, minlength=nt)
    G = len(Zu)
    se = float(np.sqrt((Zu @ Zu) / den ** 2 * G / (G - 1.0)))
    p = np.nan
    if reps > 0:
        main_ids = d.loc[d.treat == 0, "ticker"].unique()
        ng = int(d.loc[d.treat == 1, "ticker"].nunique())
        ys = d[var].to_numpy()
        yy2 = twoway(ys, ti, yi, nt, ny)
        betas = np.empty(reps)
        for k in range(reps):
            pick = set(rng.choice(main_ids, ng, replace=False))
            D2 = ((d.ticker.isin(pick)) & (d.year.isin(post_years))).astype(float).to_numpy()
            D2y = twoway(D2, ti, yi, nt, ny)
            betas[k] = float(D2y @ yy2) / float(D2y @ D2y)
        p = float((1 + np.sum(betas <= b)) / (1 + reps))
    return b, se, b / se if se > 0 else np.nan, p


def signed(d, var, post_years, rng, reps=499):
    """双侧 p 用 |beta| 的安慰剂分布。"""
    ti = pd.Categorical(d.ticker).codes
    yi = pd.Categorical(d.year).codes
    nt, ny = ti.max() + 1, yi.max() + 1
    yy2 = twoway(d[var].to_numpy(), ti, yi, nt, ny)
    main_ids = d.loc[d.treat == 0, "ticker"].unique()
    ng = int(d.loc[d.treat == 1, "ticker"].nunique())
    betas = np.empty(reps)
    for k in range(reps):
        pick = set(rng.choice(main_ids, ng, replace=False))
        D2 = ((d.ticker.isin(pick)) & (d.year.isin(post_years))).astype(float).to_numpy()
        D2y = twoway(D2, ti, yi, nt, ny)
        betas[k] = float(D2y @ yy2) / float(D2y @ D2y)
    return betas


# ---------------- Panel A：基线 ----------------
PRE, POST = [2017, 2018, 2019], [2021, 2022, 2023]
base = v3[v3.year.isin(PRE + POST)].copy()
base["treat"] = (base.board == "gem").astype(int)
rng = np.random.default_rng(20261026)
resA = []
log("Panel A: baseline, pre 2017-19, post 2021-23, m=2..6")
mcols = [c for c in v3.columns if c.startswith("vr") and c[2:].isdigit()]
mcols = sorted(mcols, key=lambda c: int(c[2:]))
log("%6s %12s %10s %8s %10s" % ("m", "beta", "cluSE", "t", "p_one"))
for m in [2, 3, 4, 5, 6]:
    var = "vr%d" % m
    b, se, t, p = did(base, var, POST, rng, reps=499)
    resA.append(dict(m=m, beta=b, se=se, t=t, p=p))
    log("%6d %+12.4f %10.4f %8.2f %10.3f" % (m, b, se, t, p))

# ---------------- Panel B：安慰剂年份 ----------------
log("")
log("Panel B: placebo reform dates, m=2, using pre-reform data only")
resB = []
for fake in [2016, 2017, 2018]:
    pre_y = list(range(2014, fake))
    post_y = list(range(fake, 2020))
    d = v3[v3.year.isin(pre_y + post_y)].copy()
    d["treat"] = (d.board == "gem").astype(int)
    if d.treat.sum() == 0 or (d.treat == 0).sum() == 0:
        continue
    r2 = np.random.default_rng(1000 + fake)
    b, se, t, p = did(d, "vr2", post_y, r2, reps=499)
    resB.append(dict(fake=fake, beta=b, se=se, t=t, p=p,
                     pre=pre_y, post=post_y, n=len(d)))
    log("  fake %d: beta %+.4f se %.4f t %.2f p %.3f  (pre %d-%d, post %d-%d)"
        % (fake, b, se, t, p, pre_y[0], pre_y[-1], post_y[0], post_y[-1]))

# ---------------- tab:window：预期网格 ----------------
log("")
log("Pre-window grid, post fixed 2021-2023")
resW = []
for start in [2014, 2015, 2016, 2017, 2018]:
    pre_y = list(range(start, 2020))
    d = v3[v3.year.isin(pre_y + POST)].copy()
    d["treat"] = (d.board == "gem").astype(int)
    row = dict(start=start)
    for m in [2, 3, 5]:
        r3 = np.random.default_rng(500 + start * 10 + m)
        b, se, t, p = did(d, "vr%d" % m, POST, r3, reps=499)
        row["b%d" % m] = b
        row["p%d" % m] = p
    resW.append(row)
    log("  %d-%d: m2 %+.4f (%.3f)  m3 %+.4f (%.3f)  m5 %+.4f (%.3f)"
        % (start, 2019, row["b2"], row["p2"], row["b3"], row["p3"], row["b5"], row["p5"]))

json.dump({"panelA": resA, "panelB": resB, "window": resW},
          open(os.path.join(OUT, "did_all.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False, default=str)
log("wrote did_all.json")
