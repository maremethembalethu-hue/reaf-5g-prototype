# CICIoT2023 Dataset Comparability & Feature-Validity Investigation

Why does a 46-feature raw CICIoT2023 export ("wataiData") score ~99.95% macro-F1 on
DDoS-vs-DoS classification while a 39-feature merged dataset ("merged") only scores
~73.6% on the same task? This investigation, not just the conclusion five scripts, each building on the last, that narrow the cause down from "maybe it's
the missing features" to "one specific column fails basic physical-consistency
checks.

## TL,DR


The original hypothesis that wataiData's extra engineered features (`Srate`,
`Drate`, etc.) explain its higher score is **wrong**. The real cause traces to
wataiData's `IAT` (Inter-Arrival Time) column: across 20.4 million rows it takes only
23,932 distinct values, repeating ~854 times each at an exact, evenly-spaced gap
not what genuine per-event timestamp subtraction should produce, and a sharp contrast
with merged's 344,434 distinct values over 8.8 million rows. `IAT` also doesn't
survive a file-grouped train/test split test, ruling out ordinary leakage as an
alternative explanation the problem is in how the column was computed, not in how
the data was split. Once `IAT` is excluded, wataiData's honest score (~79.7%) lands
close to merged's (~73.6%).
 

## Repository structure

```
 - experiment_file.py                    # full 5-experiment production-style pipeline
 - experiment_dataset.py   # Srate/Drate ablation + decomposition + LOO
 - diff_dataset.py            # 10-section statistical dataset comparison
 - test_leakage.py                          # group-aware split leakage test
 - investigate_lat.py                         # IAT physical-consistency forensics
 - data/MERGED_CSV/MERGED_CSV/              # Merged01.csv ... Merged63.csv (39 features)
 - data/wataiData/csv/CICIoT2023/                # part-00000-*.csv ... part-001683-*.csv (46 features)
 - outputs/                                 # every script's results land here, see below
    - experiments_<run_id>/
    - exp3_missing_features_<run_id>/
    - dataset_diff_<run_id>/
    - leakage_test_<run_id>/
    - iat_forensics_<run_id>/
```

`<run_id>` is a `YYYYMMDD_HHMMSS` timestamp generated fresh each run, so re-running a
script never overwrites a previous result set.

## Requirements

```
pip install pandas numpy scikit-learn "xgboost>=2.0" lightgbm catboost optuna matplotlib scipy
```

All plotting uses matplotlib's `Agg` backend; nothing opens a window or blocks
waiting for input; every figure is saved straight to disk.

## Datasets

Each script has `WATAI_ROOT` / `MERGED_ROOT` constants near the top, point these at
your local copies before running:

- **wataiData**: `part-00000-<uuid>-c000.csv` … `part-001683-<uuid>-c000.csv`, 46 columns.
- **merged**: `Merged01.csv` … `Merged63.csv`, 39 columns.

The two schemas overlap but aren't a clean subset/superset: 37 columns are shared, 2
(`Time_To_Live`, `IGMP`) exist only in merged, and 9 (`flow_duration`, `Duration`,
`Srate`, `Drate`, `urg_count`, `Magnitude`, `Radius`, `Covariance`, `Weight`) exist
only in wataiData. All scripts also silently fix a known raw-data column typo
(`Magnitue` to `Magnitude`).

Every script downcasts numeric columns to `float32` on load and prints a memory
estimate, since wataiData at full volume is tens of millions of rows. Start with a
low `sample_frac_watai` / `sample_frac_merged` in each script's `main()` and raise it
once you've confirmed the pipeline runs cleanly on your machine, **all of the
results in this repo's `outputs/` folders were generated with
`sample_frac_watai = sample_frac_merged = 0.5`** (a uniform 50% sample of each
source), not the lower defaults hardcoded in the scripts themselves.

---

## The scripts

### 1. `experiment_file.py`

The full production-style pipeline: five experiments in one run against both
datasets with all available models (XGBoost, LightGBM, CatBoost, Random Forest,
Decision Tree, and others).

| Experiment | What it does |
|---|---|
| 1. Dataset comparison | Every model, all available features, on each dataset, flat 8-class target |
| 2. Feature selection | Pearson-filter to Gini-rank to top-25 "Heavy" / top-15 "Lite" feature sets, computed per dataset, plus a feature-count sweep to justify the cutoff |
| 3. Srate/Drate ablation | wataiData-only ablation: all-46 vs common-37 vs 46-minus-Srate/Drate |
| 4. Data-volume scaling | Same model/features at 10/25/50/75/100% of the training split |
| 5. Flat vs 3-stage cascade | Justifies the production 3-stage architecture against a single flat classifier |

**Output:** `outputs/experiments_<run_id>/results/master_results.csv` (one row per
model x experiment x feature-set) and `outputs/experiments_<run_id>/plots/*.png`
(confusion matrices, feature-importance bars, comparison charts all descriptively
named, e.g. `exp2__watai__gini_feature_ranking.png`).

### 2. `experiment_dataset.py`

Standalone, focused version of Experiment 3, this is where the Srate/Drate
hypothesis first got tested and rejected. Three parts:

- **Part 1**: controlled ablation, wataiData rows only: all-46 vs minus-the-9-
  wataiData-only-features vs minus-Srate/Drate-only.
- **Part 2**: cross-dataset decomposition: `A` = wataiData-46, `B` = wataiData
  restricted to the 37 shared features, `C` = merged real data. Splits the observed
  gap into a *feature effect* (`A−B`) and a *residual effect* (`B−C`), this is
  where it became clear the gap is ~100% residual on the DDoS-vs-DoS subtask.
- **Part 3**: leave-one-out over the 9 wataiData-only features, ranked by macro-F1
  drop when each is individually removed, plus Gini importance.

**Output:** `outputs/exp3_missing_features_<run_id>/results/`
- `exp3_master_results.csv`: every model/variant/subtask result
- `part2_decomposition_summary.csv`: the A/B/C decomposition table
- `part3_leave_one_out_summary.csv`, `part3_gini_importance_watai_only.csv`

and `outputs/exp3_missing_features_<run_id>/plots/`: confusion matrices per variant,
macro-F1 bar comparisons, per-feature drop charts.

### 3. `diff_dataset.py`

Pure statistics, no model training, compares the two datasets directly so findings
aren't confounded by any particular classifier. Ten sections, each its own CSV:

| # | File | Contents |
|---|---|---|
| 1 | `1_dataset_overview.csv` | row counts, per-family proportions |
| 2 | `2_ddos_dos_subtype_counts.csv` | raw attack-subtype breakdown within DDoS/DoS |
| 3 | `3_duplicate_and_zero_diagnostics.csv` | duplicate-row fraction, per-feature zero-fraction |
| 4a/b | `4a_feature_distribution_all_rows.csv`, `4b_feature_distribution_ddos_dos_rows.csv` | mean/std/median/skew + KS-test per shared feature |
| 4c | `4c_protocol_type_comparison.csv` | categorical Protocol Type comparison (chi-square) |
| 5 | `5_univariate_auc_ddos_vs_dos.csv` | how well each feature *alone* separates DDoS/DoS, per dataset, **this is where `IAT`'s outsized AUC first showed up** |
| 6 | `6_univariate_auc_overall.csv` | same, averaged across all 8 families |
| 7 | `7_class_separability.csv` | centroid-distance / intra-class-spread ratio, all class pairs |
| 8 | `8_correlation_structure_diff_top_pairs.csv` | biggest feature-pair correlation differences between datasets |
| 9 | `9_merged_only_columns_diagnostics.csv` | is `Time_To_Live`/`IGMP` actually informative |
| 10 | `10_investigation_summary.csv` | one-line-per-metric scannable summary with a LIKELY/UNLIKELY flag |

**Output:** `outputs/dataset_diff_<run_id>/results/` (the CSVs above) and
`outputs/dataset_diff_<run_id>/plots/` (histograms for the most-different features, a
correlation-diff heatmap, AUC comparison bars).

### 4. `test_leakage.py`

Tests whether wataiData's near-perfect DDoS-vs-DoS score is a train/test leakage
artifact, specifically, whether rows from the same source `part-*.csv` file leaking
across a random split explains it. Tags every row with its source filename, then
compares a conventional random split against a `StratifiedGroupKFold` split (no
file's rows ever appear in both train and test), crossed with three feature sets
(full 46 / minus IAT / minus IAT & Number).

**Output:** `outputs/leakage_test_<run_id>/results/`
- `leakage_test_master_results.csv`: all 6 watai conditions x 2 models + the merged
  reference run
- `1_per_file_leakage_diagnostic.csv`: per-file label purity and IAT stats
- `2_leakage_summary.csv`: headline numbers (label purity, share of IAT variance
  explained by source file)

and `outputs/leakage_test_<run_id>/plots/`: confusion matrices per condition, and
`3__leakage_test_summary.png` (the condition-by-condition bar chart).

**Result:** random vs. group-aware split are statistically indistinguishable, file
leakage is ruled out. `IAT`'s signal is real at the row level, which sent the
investigation to the next script.

### 5. `investigate_lat.py`

The decisive script. If `IAT` isn't leaking across files, is it even a valid
statistic? Four checks, all internal to each dataset (no cross-dataset scale
assumptions required):

| # | File | Contents |
|---|---|---|
| 1 | `1_physical_consistency.csv` | does `IAT x (Number−1) ≈ Duration`? does `IAT ≈ 1/Rate`? |
| 2 | `2_quantization.csv` | unique-value count, spacing between values, top-value mass |
| 3 | `3_class_gap.csv` | do the DDoS/DoS IAT ranges even overlap, per dataset |
| 4 | `4_row_order.csv` | correlation between IAT and row position within its source file |

**Output:** `outputs/iat_forensics_<run_id>/results/` (the four CSVs above) and
`outputs/iat_forensics_<run_id>/plots/` (a Duration-vs-derived-duration scatter, IAT
histograms by class for both datasets).

**Result:** merged's `IAT` correlates with `1/Rate` at 0.99999, essentially exact.
wataiData's correlates at −0.0002 with `1/Rate` and −0.006 with `Duration`,
essentially zero on every check. Combined with the quantization result (23,932
unique values across 20.4M rows, vs. 344,434 unique values across 8.8M rows in
merged), this is the closest thing this investigation has to a smoking gun: `IAT` in
wataiData isn't inter-arrival time in any coherent sense.

---

## `outputs/` common structure

Every script follows the same layout:

```
outputs/<script_name>_<run_id>/
- results/     # CSVs every numeric finding
- plots/       # PNGs 
```

Plots are saved automatically. CSV filenames are numbered where a script has a natural
read order (e.g. `01_...csv`, `02_...csv` in `dataset_diff_investigation.py` and
`iat_forensics.py`) so you can work through a results folder top to bottom.

## Recommendations that came out of this

- Exclude `IAT` from wataiData's feature set for any model meant to generalize or be
  compared against merged.
- Treat merged's ~73–74% (or wataiData-minus-`IAT`'s ~79.7%) as the credible ceiling
  for DDoS-vs-DoS on this feature schema, not wataiData's original ~99.95%.
- The `8_correlation_structure_diff_top_pairs.csv` results are the
  leading candidate for the ~6-point residual gap that remains even after removing
  `IAT`, and are the natural next thing to dig into if this investigation continues.

See the full report for the complete methodology, all figures, limitations, and
detailed recommendations.
