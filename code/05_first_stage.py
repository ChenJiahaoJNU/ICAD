# -*- coding: utf-8 -*-
"""
首阶段 DiD 的正确算法：FWL 双向去均值 + 单参数聚类标准误 + 随机化推断。
上一版把 440 个股票虚拟变量直接塞进 CRVE 的 meat，得到 t=-47.4，不可信；
单参数情形用 FWL 残差化后计算聚类方差是良定义的。
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_verify")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "firststage.log")
open(LOG, "w").close()


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")


v3 = pd.read_parquet(os.path.join(BASE, "results_v3", "panel_v3.parquet"))
PRE, POST = [2017, 2018, 2019], [2021, 2022, 2023]
d = v3[v3.year.isin(PRE + POST)].copy()
d["y"] = d.hit_rate.to_numpy() * 100.0
log("stock-years %d, stocks %d (gem %d, main %d)"
    % (len(d), d.ticker.nunique(),
       d.loc[d.board == "gem"].ticker.nunique(),
       d.loc[d.board == "main"].ticker.nunique()))

ti_all = pd.Categorical(d.ticker).codes
yi_all = pd.Categorical(d.year).codes
NT, NY = ti_all.max() + 1, yi_all.max() + 1


def twoway_demean(v, ti, yi, nt, ny, iters=60, tol=1e-12):
    """交替投影做双向去均值（非平衡面板下的 FWL）。"""
    x = v.astype(float).copy()
    for _ in range(iters):
        old = x.copy()
        s = np.bincount(ti, weights=x, minlength=nt)
        c = np.bincount(ti, minlength=nt)
        x = x - (s / np.maximum(c, 1))[ti]
        s = np.bincount(yi, weights=x, minlength=ny)
        c = np.bincount(yi, minlength=ny)
        x = x - (s / np.maximum(c, 1))[yi]
        if np.max(np.abs(x - old)) < tol:
            break
    return x


def fwl(dd, ti, yi, nt, ny):
    y = twoway_demean(dd.y.to_numpy(), ti, yi, nt, ny)
    D = twoway_demean(dd.D.to_numpy(), ti, yi, nt, ny)
    den = float(D @ D)
    b = float(D @ y) / den
    u = y - b * D
    Z = D * u
    Zu = np.bincount(ti, weights=Z, minlength=nt)
    G = len(Zu)
    V = float(Zu @ Zu) / den ** 2
    V *= G / (G - 1.0)
    return b, float(np.sqrt(V))


d["D"] = ((d.board == "gem") & (d.year.isin(POST))).astype(float)
b, se = fwl(d, ti_all, yi_all, NT, NY)
log("FWL within (stock+year FE): beta = %+.4f pp, clustered SE %.4f, t = %.2f"
    % (b, se, b / se))

# 单元格均值（不加权，与回归同口径）
def cm(bd, yrs):
    s = d[(d.board == bd) & (d.year.isin(yrs))]
    return float(s.y.mean())


cells = dict(gem_pre=cm("gem", PRE), gem_post=cm("gem", POST),
             main_pre=cm("main", PRE), main_post=cm("main", POST))
cells["gem_chg"] = cells["gem_post"] - cells["gem_pre"]
cells["main_chg"] = cells["main_post"] - cells["main_pre"]
cells["did_2x2"] = cells["gem_chg"] - cells["main_chg"]
log("cells (unweighted stock-year): GEM %.3f -> %.3f (%+.3f) | MAIN %.3f -> %.3f (%+.3f)"
    % (cells["gem_pre"], cells["gem_post"], cells["gem_chg"],
       cells["main_pre"], cells["main_post"], cells["main_chg"]))
log("2x2 DiD = %+.3f ; FE regression = %+.3f (unbalanced panel)" % (cells["did_2x2"], b))

# 随机化推断：整体打乱板别标签，保持每只股票全部年份
rng = np.random.default_rng(20261026)
main_ids = d.loc[d.board == "main", "ticker"].unique()
ng = int(d.loc[d.board == "gem", "ticker"].nunique())
B = 999
betas = np.empty(B)
for i in range(B):
    pick = set(rng.choice(main_ids, ng, replace=False))
    dd = d.copy()
    dd["D"] = (dd.ticker.isin(pick) & dd.year.isin(POST)).astype(float).to_numpy()
    betas[i], _ = fwl(dd, ti_all, yi_all, NT, NY)
p = float((1 + np.sum(betas <= b)) / (1 + B))
log("randomisation %d draws: placebo mean %+.4f sd %.4f | p(lower) = %.4f"
    % (B, betas.mean(), betas.std(), p))

json.dump({"beta": b, "clustered_se": se, "t": b / se, "cells": cells,
           "rand_p": p, "rand_B": B,
           "n_stocks": int(d.ticker.nunique()),
           "n_gem": int(d.loc[d.board == "gem"].ticker.nunique()),
           "n_main": int(d.loc[d.board == "main"].ticker.nunique()),
           "n_stock_years": int(len(d)), "n_windows": int(d.n.sum())},
          open(os.path.join(OUT, "firststage.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
log("wrote firststage.json")
