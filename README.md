# Replication package

**Price limits and measured risk in China: invariance, calibration, and a pre-trend diagnosis**

Xiangdong Liu, Jiahao Chen, Zhenyan Liu, Jinfeng Pan

Submitted to *Empirical Economics*, special issue on Economic Transformation in China.

---

## What this package contains

| Path | Content |
|---|---|
| `code/` | Twelve Python scripts, numbered in execution order |
| `results/` | The expected output of every script, so a rerun can be checked against ours |
| `MANIFEST.md` | Map from manuscript exhibit to script to output file |
| `requirements.txt` | Environment |

The three data files are hosted separately, because they are large and binary:

**Data: <https://huggingface.co/datasets/Changahou/ICAD>**

| File | Content | Size |
|---|---|---|
| `panel_merged.parquet` | Analysis panel: 444 A-share stocks, 2,676 trading days (2014-01-02 to 2024-12-31), 1,188,144 stock-day rows | 39 MB |
| `panel_v3.parquet` | Stock-year panel of variance ratios on rolling 250-session windows | 1.2 MB |
| `units.parquet` | Stock-half-year panel on rolling 125-session windows | 7.8 MB |

Download them into `data/`, `results_v3/` and `results_sc/` before running anything:

```bash
mkdir -p data results_v3 results_sc
# panel_merged.parquet -> data/
# panel_v3.parquet     -> results_v3/
# units.parquet        -> results_sc/
```

The 444 individual stock CSV files retrieved from the source interface (188 MB) are not
distributed; they are redundant with `panel_merged.parquet` and can be regenerated from the
public interface.

---

## Requirements

```
Python 3.10
numpy 1.26.4, pandas 2.3.3, scipy 1.15.3, matplotlib, pyarrow
```

```bash
pip install -r requirements.txt
```

Paths are resolved relative to the package root, so the package can be unzipped anywhere and
run without editing. To point at a different data location, set the environment variable
`REPL_BASE` to the package root.

---

## How to reproduce

Run from the package root. Every script is seeded; no result depends on an unseeded draw.

### Stage 1 — rebuild the two estimation panels (optional)

`results_v3/panel_v3.parquet` and `results_sc/units.parquet` are shipped with the package, so
this stage is optional. Skipping it saves roughly 30 minutes.

```bash
python code/01_build_annual_panel.py      # -> results_v3/panel_v3.parquet  (~25 min)
python code/02_build_halfyear_units.py    # -> results_sc/units.parquet     (~5 min)
```

### Stage 2 — reproduce every table and figure

```bash
python code/03_sample_ledger.py            # Table 1
python code/09_mechanism_calibration.py    # Tables 2 and 4  (~3 min)
python code/10_descriptive_hetero_tables.py # Tables 3 and 4: descriptive and heterogeneity data (~4 min)
python code/04_descriptives.py             # Tables 5 and 6, Figure 2  (~2 min)
python code/05_first_stage.py              # Table 7  (~1 min)
python code/07_synthetic_control.py        # Tables 8 and 9  (~8 min)
python code/08_holdout_pretrend.py         # Table 3, the pre-trend diagnosis  (~12 min)
python code/06_did_and_placebo.py          # Tables 11 and 12  (~2 min)
python code/11_figures_1to5.py             # Figures 1 to 5  (~3 min)
python code/12_figure_6_event.py           # Figure 6  (~2 min)
```

`08_holdout_pretrend.py` is the one that matters most: it produces the estimate we treat as
the paper's surviving result, together with the pre-trend diagnostic that removes the
five-day estimate.

---

## Where the underlying data come from

Raw prices are retrieved from the **Baostock** open interface (<https://www.baostock.com>),
which returns unadjusted daily OHLC, the exchange-provided previous close, trading status, and
the delisting-risk-warning flag for Shanghai and Shenzhen listed securities. Nothing in the
package is proprietary or restricted.

To rebuild `panel_merged.parquet` from scratch you need `baostock` and network access. The
retrieval script used is not part of this package because it embeds a stock-selection step that
we document in the manuscript's sample construction table (Table 1): the universe is 444
stocks with complete histories over the sample window, 280 on the main board and 164 on
ChiNext. Limit prices are reconstructed from the previous close and the applicable band
(5% for delisting-warning stocks, 20% for ChiNext from 24 August 2020, 10% otherwise) and
rounded to the nearest fen; the manuscript reports two validation checks on that
reconstruction.

---

## Table and figure map

See `MANIFEST.md` for the table-by-table and figure-by-figure mapping from manuscript exhibit
to script to output file.

---

## A note on what the shipped numbers are

The paper's headline empirical claim is deliberately narrow. Of the three horizons examined:

- The five-day estimate of −0.063 from a conventional synthetic control (full pre-reform path
  fit) does **not** survive the pre-trend diagnostic. Fitting weights on 2014H1–2017H2 only,
  the treated unit runs below its synthetic counterpart by −0.081 in the four held-out
  pre-reform half-years, which exceeds the post-reform gap of −0.077.
- The three-day estimate of −0.042 (one-sided p = 0.055) is the one that survives, and it is
  labelled exploratory in the manuscript.
- The two-day estimate does not survive.

`code/08_holdout_pretrend.py` reproduces all three numbers together with the 2,000 aggregate
placebos behind each p-value.
