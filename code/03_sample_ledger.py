import os
# -*- coding: utf-8 -*-
"""核查稿子里所有样本量数字，输出 sample construction 底账。"""
import os, json
import numpy as np
import pandas as pd

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_verify")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "verify.log")
open(LOG, "w").close()
def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

L = {}
raw = pd.read_parquet(os.path.join(BASE, "data", "panel_merged.parquet"))
raw["d"] = pd.to_datetime(raw["date"])
L["stocks_downloaded"] = int(raw.code.nunique())
L["trading_days"] = int(raw.d.nunique())
L["calendar_start"] = str(raw.d.min().date())
L["calendar_end"] = str(raw.d.max().date())
L["stock_days_raw"] = int(len(raw))
log(f"raw: {L['stocks_downloaded']} stocks, {L['trading_days']} days, "
    f"{L['stock_days_raw']:,} rows")

# 板别
def board(c):
    return "gem" if c.startswith("sz.30") else "main"
raw["board"] = raw.code.map(board)
L["raw_by_board"] = raw.groupby("board").code.nunique().to_dict()
log(f"raw by board: {L['raw_by_board']}")

# --- V3 面板（W=250, 年）---
v3 = pd.read_parquet(os.path.join(BASE, "results_v3", "panel_v3.parquet"))
L["v3_stocks"] = int(v3.ticker.nunique())
L["v3_by_board"] = v3.groupby("board").ticker.nunique().to_dict()
L["v3_rows"] = int(len(v3))
L["v3_windows"] = int(v3.n.sum())
log(f"panel_v3 (W=250): {L['v3_stocks']} stocks {L['v3_by_board']}, "
    f"{L['v3_windows']:,} admissible windows")

# --- SC 单元（W=125, 半年）---
u = pd.read_parquet(os.path.join(BASE, "results_sc", "units.parquet"))
L["units_stocks"] = int(u.ticker.nunique())
L["units_by_board"] = u.groupby("board").ticker.nunique().to_dict()
L["units_rows"] = int(len(u))
L["units_periods"] = int(u.t.nunique())
log(f"units (W=125): {L['units_stocks']} stocks {L['units_by_board']}, "
    f"{L['units_rows']:,} rows, {L['units_periods']} half-years")

# 复现 SC 的 65% 覆盖门槛
PRE_T = list(range(2014 * 2, 2020 * 2)); POST_T = list(range(2021 * 2, 2025 * 2))
board_of = u.groupby("ticker")["board"].first().to_dict()
keep = {}
for m in [2, 3, 5]:
    cell = u.groupby(["ticker", "t"]).agg(v=(f"vr{m}", "mean"), n=(f"vr{m}", "count")).reset_index()
    cell = cell[cell.n >= 4]
    wide = cell.pivot(index="t", columns="ticker", values="v").reindex(PRE_T + POST_T)
    pc = wide.loc[PRE_T].notna().mean(axis=0); qc = wide.loc[POST_T].notna().mean(axis=0)
    sel = wide.columns[(pc >= 0.65) & (qc >= 0.65)]
    nb = pd.Series([board_of.get(t) for t in sel]).value_counts().to_dict()
    keep[m] = {"total": int(len(sel)), "by_board": nb}
    log(f"  m={m}: SC-eligible {len(sel)} stocks {nb}")
L["sc_eligible"] = keep

# --- first stage：真实单元格均值 ---
FS_PRE = [2017, 2018, 2019]; FS_POST = [2021, 2022, 2023]
def cellmean(board_name, yrs):
    s = v3[(v3.board == board_name) & (v3.year.isin(yrs))]
    return float(np.average(s.hit_rate, weights=s.n) * 100)
fs = {"pre_gem": cellmean("gem", FS_PRE), "post_gem": cellmean("gem", FS_POST),
      "pre_main": cellmean("main", FS_PRE), "post_main": cellmean("main", FS_POST)}
fs["did_2x2"] = (fs["post_gem"] - fs["pre_gem"]) - (fs["post_main"] - fs["pre_main"])
L["first_stage"] = fs
log(f"first stage cell means: GEM {fs['pre_gem']:.3f}->{fs['post_gem']:.3f}  "
    f"MAIN {fs['pre_main']:.3f}->{fs['post_main']:.3f}  DiD(2x2)={fs['did_2x2']:+.3f}pp")

# 该子样本的股票数
sub = v3[v3.year.isin(FS_PRE + FS_POST)]
L["fs_stocks"] = int(sub.ticker.nunique())
L["fs_by_board"] = sub.groupby("board").ticker.nunique().to_dict()
L["fs_stock_years"] = int(len(sub))
L["fs_windows"] = int(sub.n.sum())
log(f"first-stage sample: {L['fs_stocks']} stocks {L['fs_by_board']}, "
    f"{L['fs_stock_years']} stock-years, {L['fs_windows']:,} windows")

# --- 全样本年份覆盖 ---
L["v3_years"] = sorted(int(y) for y in v3.year.unique())
L["units_halfyears"] = [f"{t//2}H{1 if t%2==0 else 2}" for t in sorted(u.t.unique())]

# --- limit-up 总体频率（全样本，按 stock-days）---
L["limitup_rate_all"] = float(raw.pctChg.abs().notna().mean())  # placeholder
hit = ((raw.close == raw.preclose * (1 + raw.get("band", 0)))).mean() if "band" in raw else None

# 用 v3 的 hit_rate 加权
L["limitup_rate_v3"] = float(np.average(v3.hit_rate, weights=v3.n) * 100)
log(f"limit-up rate (v3-weighted): {L['limitup_rate_v3']:.3f}%")

json.dump(L, open(os.path.join(OUT, "sample_ledger.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
log("\nwrote sample_ledger.json")
