# -*- coding: utf-8 -*-
"""
主设定：pathF（匹配全部改革前半年 VR 路径，Abadie 原始做法）+ vol + hit。
同一引擎做：聚合安慰剂推断（B=2000，加一修正）、规模稳健性、留出期验证、逐期缺口。
同时跑 segment 设定作为对照，两者都如实报告。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls

BASE = os.environ.get(
    "REPL_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results_sc")
LOG = os.path.join(OUT, "sc_final.log")
open(LOG, "w").close()


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")


PRE = list(range(2014 * 2, 2020 * 2))
POST = list(range(2021 * 2, 2025 * 2))
R = pd.read_parquet(os.path.join(OUT, "units.parquet"))
B = int(os.environ.get("SC_B", "2000"))
R_MODE = ["pathF", "segment"]


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


class Design:
    def __init__(self, m):
        cell = R.groupby(["ticker", "t"]).agg(v=(f"vr{m}", "mean"),
                                              n=(f"vr{m}", "count")).reset_index()
        cell = cell[cell.n >= 4]
        self.wide = (cell.pivot(index="t", columns="ticker", values="v")
                     .reindex(PRE + POST))
        pc = self.wide.loc[PRE].notna().mean(axis=0)
        qc = self.wide.loc[POST].notna().mean(axis=0)
        self.wide = self.wide.loc[:, (pc >= .65) & (qc >= .65)]
        self.Y = self.wide.to_numpy(float)
        self.col = {c: i for i, c in enumerate(self.wide.columns)}
        self.board = {c: R.loc[R.ticker == c, "board"].iloc[0]
                      for c in self.wide.columns}
        self.gem = [c for c in self.wide.columns if self.board[c] == "gem"]
        self.main = [c for c in self.wide.columns if self.board[c] == "main"]
        self.R = R
        self._c = {}

    def Z(self, mode, fitp):
        key = (mode, tuple(fitp))
        if key in self._c:
            return self._c[key]
        fitp = list(fitp)
        sub = self.R[self.R.t.isin(fitp)]
        vol = sub.groupby("ticker")["vol"].mean()
        hit = sub.groupby("ticker")["hit"].mean()
        cols = list(self.wide.columns)
        if mode == "segment":
            k = max(1, len(fitp) // 3)
            segs = [fitp[0:k], fitp[k:2 * k], fitp[2 * k:]]
            mat = [self.wide.loc[sg].mean(axis=0, skipna=True).to_numpy() for sg in segs]
        else:
            mat = [self.wide.loc[t].to_numpy() for t in fitp]
        X = np.vstack(mat).T
        cm = np.nanmean(X, axis=0)
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(cm, bad[1])
        X = np.column_stack([X, vol.reindex(cols).to_numpy(),
                             hit.reindex(cols).to_numpy()])
        out = pd.DataFrame(np.nan_to_num(X, nan=0.0), index=cols)
        self._c[key] = out
        return out

    def fit(self, treated, donors, mode, fitp, evalp):
        Z = self.Z(mode, fitp)
        z1 = Z.loc[treated].to_numpy().mean(axis=0)
        Zall = Z.to_numpy()
        rows = {t: i for i, t in enumerate(Z.index)}
        Zi = np.array([rows[d] for d in donors])
        Zr = np.array([rows[x] for x in treated])
        sd = np.abs(Zall[np.concatenate([Zi, Zr])].std(axis=0))
        sd[sd == 0] = 1.0
        w = _snnls(z1, Zall[Zi].T, np.diag(1.0 / sd ** 2))
        ci = np.array([self.col[t] for t in treated])
        cd = np.array([self.col[d] for d in donors])
        y = np.nanmean(self.Y[:, ci], axis=1)
        syn = _pred(self.Y[:, cd], w)
        gap = y - syn
        pi = {t: i for i, t in enumerate(self.wide.index)}
        fi = [pi[t] for t in fitp if t in pi]
        ei = [pi[t] for t in evalp if t in pi]
        qi = [pi[t] for t in POST]
        return dict(w=w, y=y, syn=syn, gap=gap, index=self.wide.index,
                    pre_rm=float(np.sqrt(np.mean(gap[fi] ** 2))),
                    out_rm=float(np.sqrt(np.mean(gap[ei] ** 2))),
                    att=float(np.mean(gap[qi])),
                    post_rm=float(np.sqrt(np.mean(gap[qi] ** 2))),
                    donors=donors, treated=treated)


def run(dsg, nsz, labels, mode, rng, Bp):
    o = dsg.fit(labels, dsg.main, mode, PRE, PRE)
    k = int(min(len(o["treated"]), len(dsg.main) - 12))
    ma = np.array(dsg.main)
    pa, pr = [], []
    for _ in range(Bp):
        pg = list(ma[rng.choice(len(ma), k, replace=False)])
        dn = [d for d in dsg.main if d not in set(pg)]
        oj = dsg.fit(pg, dn, mode, PRE, PRE)
        pa.append(oj["att"])
        pr.append(oj["post_rm"] / oj["pre_rm"] if oj["pre_rm"] > 0 else np.nan)
    pa = np.asarray(pa); pr = np.asarray(pr)
    ratio = o["post_rm"] / o["pre_rm"] if o["pre_rm"] > 0 else np.nan
    return dict(nt=len(o["treated"]), nd=len(o["donors"]), att=o["att"],
                pre_rm=o["pre_rm"], post_rm=o["post_rm"], ratio=ratio,
                pl_mean=float(pa.mean()), pl_sd=float(pa.std()),
                pl_p05=float(np.quantile(pa, .05)),
                p1=float((1 + np.sum(pa <= o["att"])) / (1 + len(pa))),
                p2=float((1 + np.sum(np.abs(pa) >= abs(o["att"]))) / (1 + len(pa))),
                pr=float((1 + np.sum(pr >= ratio)) / (1 + len(pr))),
                eff=1.0 / float(np.sum(o["w"] ** 2)), topw=float(o["w"].max()),
                nz=int((o["w"] > 1e-4).sum()), o=o)


main_rows, rob_rows = [], []
for mode in R_MODE:
    log("")
    log("=" * 104)
    log("Specification: %s  (B=%d aggregate placebos, +1 correction)" % (mode, B))
    log("=" * 104)
    for m in [2, 3, 5]:
        dsg = Design(m)
        log("")
        log("m=%d  eligible %d = %d ChiNext + %d main"
            % (m, len(dsg.wide.columns), len(dsg.gem), len(dsg.main)))
        for nsz in (60, 90, 125):
            if nsz > len(dsg.gem):
                continue
            rng = np.random.default_rng(7000 + nsz)
            lab = (list(dsg.gem) if nsz == len(dsg.gem)
                   else list(rng.choice(dsg.gem, nsz, replace=False)))
            r = run(dsg, nsz, lab, mode, rng, B)
            row = dict(m=m, spec=mode, nsz=nsz, **{k: r[k] for k in
                       ["nt", "nd", "att", "pre_rm", "post_rm", "ratio", "pl_mean",
                        "pl_sd", "pl_p05", "p1", "p2", "pr", "eff", "topw", "nz"]})
            rob_rows.append(row)
            log("  nsz=%3d ATT %+.4f preRM %.4f OOSratio %.2f | effN %.1f topw %.2f | "
                "placebo %+.4f sd %.4f | p1=%.4f p2=%.4f pRatio=%.4f"
                % (nsz, r["att"], r["pre_rm"], r["ratio"], r["eff"], r["topw"],
                   r["pl_mean"], r["pl_sd"], r["p1"], r["p2"], r["pr"]))
            if nsz == len(dsg.gem):
                main_rows.append(row)
                # 逐期缺口 + 留出检验
                o = r["o"]
                pi = {t: i for i, t in enumerate(o["index"])}
                g = [o["gap"][pi[t]] for t in POST]
                lead = int(np.argmax(np.abs(np.array(g) / len(POST))))
                wo = float(np.mean([g[i] for i in range(len(POST)) if i != lead]))
                log("         gaps: " + " ".join("%+.3f" % x for x in g))
                log("         drop worst period (%dH%d): ATT %+.4f -> %+.4f"
                    % (POST[lead] // 2, 1 if POST[lead] % 2 == 0 else 2, r["att"], wo))
                K = len(PRE) * 2 // 3
                ho = dsg.fit(dsg.gem, dsg.main, mode, PRE[:K], PRE[K:])
                log("         HOLDOUT OOS RMSE %.4f (in-sample %+.4f)"
                    % (ho["out_rm"], ho["pre_rm"]))
                row["att_ex_worst"] = wo
                row["holdout_oos"] = ho["out_rm"]

pd.DataFrame(main_rows).to_csv(os.path.join(OUT, "sc_final_main.csv"),
                               index=False, encoding="utf-8-sig")
pd.DataFrame(rob_rows).to_csv(os.path.join(OUT, "sc_final_robust.csv"),
                              index=False, encoding="utf-8-sig")
log("")
log("wrote sc_final_main.csv / sc_final_robust.csv")
