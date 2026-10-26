# Exhibit map

Every table and figure in the manuscript, the script that produces it, and the file it writes.
Table numbers follow the manuscript.

| Exhibit | Content | Script | Output |
|---|---|---|---|
| **Table 1** | Sample construction | `code/03_sample_ledger.py` | `results_verify/sample_ledger.json` |
| **Table 2** | Simulated variance ratios under truncation and deferral | `code/09_mechanism_calibration.py` | `results_verify/deferral_final.csv`, `deferral_final.json` |
| **Table 3** | Pre-trend diagnosis (holdout fit) | `code/08_holdout_pretrend.py` | `results_sc/sc_holdout_inference.csv` |
| **Table 4** | Estimates against the calibrated benchmark | `code/09_mechanism_calibration.py` + `code/07_synthetic_control.py` | `deferral_final.json`, `sc_final_main.csv` |
| **Table 5** | Descriptive statistics of VR(m) by board and period | `code/10_descriptive_hetero_tables.py` | `results_sc/tab_descriptive.csv` |
| **Table 6** | Limit-hit frequency by year and board | `code/04_descriptives.py` | `results_verify/recount.json` |
| **Table 7** | Limit-hit frequency by volatility decile | `code/04_descriptives.py` | `results_verify/recount.json` |
| **Table 8** | First stage | `code/05_first_stage.py` | `results_verify/firststage.json` |
| **Table 9** | Synthetic control estimates | `code/07_synthetic_control.py` | `results_sc/sc_final_main.csv` |
| **Table 10** | Robustness to treated-group size | `code/07_synthetic_control.py` | `results_sc/sc_final_robust.csv` |
| **Table 11** | Heterogeneity by pre-reform limit-hit intensity | `code/10_descriptive_hetero_tables.py` | `results_sc/tab_heterogeneity.csv` |
| **Table 12** | Difference-in-differences and the placebo that rejects it | `code/06_did_and_placebo.py` | `results_verify/did_all.json` |
| **Table 13** | Sensitivity to the pre-window | `code/06_did_and_placebo.py` | `results_verify/did_all.json` |
| **Figure 1** | DiD coefficients by horizon | `code/11_figures_1to5.py` | `figures/Fig1.pdf` |
| **Figure 2** | Limit-hit frequency by volatility decile | `code/11_figures_1to5.py` | `figures/Fig2.pdf` |
| **Figure 3** | VR term structure by board and period | `code/11_figures_1to5.py` | `figures/Fig3.pdf` |
| **Figure 4** | Placebo distribution and per-period gaps | `code/11_figures_1to5.py` | `figures/Fig4.pdf` |
| **Figure 5** | Treated against synthetic | `code/11_figures_1to5.py` | `figures/Fig5.pdf` |
| **Figure 6** | Event-time gaps and calibrated benchmark | `code/12_figure_6_event.py` | `figures/Fig6.pdf` |
| **Appendix A** | Estimation-protocol comparison | `code/04_descriptives.py` and a dedicated check | see below |

## Notes on individual exhibits

**Table 3 is the load-bearing exhibit.** `code/08_holdout_pretrend.py` fits the synthetic-control
weights on 2014H1–2017H2 only, with the matching variables drawn from those same periods, and
evaluates the four held-out pre-reform half-years (2018H1–2019H2) as a pre-trend diagnostic. It
reports, for each horizon, the post-reform gap, the held-out pre-reform gap, their difference,
and a one-sided placebo p-value for the difference. 2,000 aggregate placebo groups of matched
size are re-estimated under the identical procedure. Runtime is about 12 minutes.

**Appendix A** compares variance ratios computed from concatenated available returns against
those from complete calendar windows. The comparison is produced by a standalone check rather
than by `04_descriptives.py`; the numbers are 0.960 versus 0.656 at m = 60 and 1.078 versus
0.908 at m = 20.

**Table 1** is generated from the two shipped panels, not from the raw stock files, so it can be
checked without any download.

## Intermediate objects

| File | Built by | Used by |
|---|---|---|
| `data/panel_merged.parquet` | retrieval from Baostock (not shipped) | all scripts |
| `results_v3/panel_v3.parquet` | `code/01_build_annual_panel.py` | `03`, `05`, `06`, `10`, `11` |
| `results_sc/units.parquet` | `code/02_build_halfyear_units.py` | `07`, `08`, `09`, `10`, `11`, `12` |

Both intermediates are shipped, so the pipeline can be entered at any stage.
