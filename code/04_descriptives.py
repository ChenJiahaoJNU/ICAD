import os
# -*- coding: utf-8 -*-
"""从 444 只原始面板重算稿子里所有描述性数字（band 自行重建）。"""
import os, json
import numpy as np
import pandas as pd

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_verify")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "recount.log")
open(LOG, "w").close()
def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

REFORM = pd.Timestamp("2020-08-24")
raw = pd.read_parquet(os.path.join(BASE, "data", "panel_merged.parquet"))
raw["d"] = pd.to_datetime(raw["date"])
raw["board"] = np.where(raw.code.str.startswith("sz.30"), "gem", "main")

def band_of(df):
    b = np.full(len(df), 0.10)
    b[df.board.values == "gem"] = 0.10
    b[(df.board.values == "gem") & (df.d.values >= np.datetime64(REFORM))] = 0.20
    b[df.isST.values.astype(bool)] = 0.05
    return b

raw["band"] = band_of(raw)
raw["preclose"] = raw["preclose"].astype(float)
raw["close"] = raw["close"].astype(float)
raw["hi"] = np.round(raw.preclose * (1 + raw.band), 2)
raw["lo"] = np.round(raw.preclose * (1 - raw.band), 2)
raw["up"] = (np.abs(raw.close - raw.hi) < 0.005).astype(float)
raw["dn"] = (np.abs(raw.close - raw.lo) < 0.005).astype(float)
raw = raw[raw.tradestatus.astype(str).isin(["1", "1.0"]) & raw.preclose.gt(0)]
log(f"usable rows {len(raw):,}  stocks {raw.code.nunique()}  days {raw.d.nunique()}")

L = {}
L["stock_days"] = int(len(raw))
L["stocks"] = int(raw.code.nunique())
L["days"] = int(raw.d.nunique())
L["up_days"] = int(raw.up.sum())
L["up_rate_all"] = float(raw.up.mean() * 100)
L["dn_rate_all"] = float(raw.dn.mean() * 100)
log(f"stock-days {len(raw):,}; limit-up {L['up_days']:,} ({L['up_rate_all']:.3f}%); "
    f"limit-down {L['dn_rate_all']:.3f}%")
L["by_board"] = {"gem": int(raw[raw.board == "gem"].code.nunique()),
                 "main": int(raw[raw.board == "main"].code.nunique())}

# 年度 × 板别
raw["year"] = raw.d.dt.year
piv = raw.groupby(["year", "board"]).agg(up=("up", "mean"), dn=("dn", "mean"),
                                         n=("up", "size")).reset_index()
piv["up"] *= 100; piv["dn"] *= 100
tab = piv.pivot(index="year", columns="board", values=["up", "dn"])
L["censoring_table"] = {str(int(y)): {
    "main_up": float(tab[("up", "main")][y]), "main_dn": float(tab[("dn", "main")][y]),
    "gem_up": float(tab[("up", "gem")][y]), "gem_dn": float(tab[("dn", "gem")][y])}
    for y in tab.index}
log("\nyear | main_up main_dn | gem_up gem_dn")
for y, v in L["censoring_table"].items():
    log(f"{y} | {v['main_up']:6.3f} {v['main_dn']:6.3f} | {v['gem_up']:6.3f} {v['gem_dn']:6.3f}")

# 波动率十分位：按交易日横截面排名（与时点市场环境对齐，避免股票固定效应主导分位）
raw = raw.sort_values(["code", "d"])
raw["r"] = np.log(raw.close / raw.preclose)
raw["vol60"] = (raw.groupby("code")["r"].transform(
    lambda s: s.rolling(60, min_periods=40).std().shift(1)))
v = raw.dropna(subset=["vol60"]).copy()
v["dec"] = v.groupby("d")["vol60"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
vt = v.groupby("dec").agg(vol=("vol60", "mean"), up=("up", "mean"), dn=("dn", "mean"),
                          n=("up", "size"))
vt["vol"] *= 100; vt["up"] *= 100; vt["dn"] *= 100
L["vol_deciles"] = {int(k): {"vol": float(r.vol), "up": float(r.up), "dn": float(r.dn)}
                    for k, r in vt.iterrows()}
log("\ndecile | vol% | up% | dn%")
for k, r in L["vol_deciles"].items():
    log(f"{k:2d} | {r['vol']:5.2f} | {r['up']:5.3f} | {r['dn']:5.3f}")
L["vol_ratio_10_1"] = float(vt.up.iloc[-1] / vt.up.iloc[0])
log(f"decile10/decile1 limit-up ratio = {L['vol_ratio_10_1']:.2f}x")

# band 重建核对
sealed = raw[raw.up == 1]
L["pctchg_tail_5"] = float(np.nanpercentile(sealed.pctChg.astype(float), 5))
log(f"\nsealed limit-up days: {len(sealed):,}; 5th pct of pctChg = {L['pctchg_tail_5']:.2f}%")
bad = sealed[sealed.pctChg.astype(float) < (sealed.band * 100 - 0.5)]
L["sealed_far_below_band"] = int(len(bad))
log(f"sealed days more than 0.5pp below band: {len(bad)}")

# 首阶段（2017-19 vs 2021-23）
v3 = pd.read_parquet(os.path.join(BASE, "results_v3", "panel_v3.parquet"))
PRE, POST = [2017, 2018, 2019], [2021, 2022, 2023]
def cm(bd, yrs):
    s = v3[(v3.board == bd) & (v3.year.isin(yrs))]
    return float(np.average(s.hit_rate, weights=s.n) * 100)
fs = dict(pre_gem=cm("gem", PRE), post_gem=cm("gem", POST),
          pre_main=cm("main", PRE), post_main=cm("main", POST))
fs["did_2x2"] = (fs["post_gem"] - fs["pre_gem"]) - (fs["post_main"] - fs["pre_main"])
# 同样用 raw 直算（应一致，检验 v3 的窗口加权有无偏差）
def cm_raw(bd, yrs):
    s = raw[(raw.board == bd) & (raw.year.isin(yrs))]
    return float(s.up.mean() * 100)
fs["raw_pre_gem"] = cm_raw("gem", PRE); fs["raw_post_gem"] = cm_raw("gem", POST)
fs["raw_pre_main"] = cm_raw("main", PRE); fs["raw_post_main"] = cm_raw("main", POST)
fs["raw_did"] = ((fs["raw_post_gem"] - fs["raw_pre_gem"])
                 - (fs["raw_post_main"] - fs["raw_pre_main"]))
L["first_stage"] = fs
log(f"\nfirst stage (v3 window-weighted): GEM {fs['pre_gem']:.3f}->{fs['post_gem']:.3f} "
    f"MAIN {fs['pre_main']:.3f}->{fs['post_main']:.3f} DiD={fs['did_2x2']:+.3f}")
log(f"first stage (raw stock-day):     GEM {fs['raw_pre_gem']:.3f}->{fs['raw_post_gem']:.3f} "
    f"MAIN {fs['raw_pre_main']:.3f}->{fs['raw_post_main']:.3f} DiD={fs['raw_did']:+.3f}")

json.dump(L, open(os.path.join(OUT, "recount.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
log("\nwrote recount.json")
