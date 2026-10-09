# Supplementary Material S1: The DDoS-vs-DoS Investigation



---

## 1. Background and Research Question

Our production pipeline classifies CICIoT2023 traffic in three stages:

1. Benign vs. attack.
2. A 6-way family classification, where DDoS and DoS are merged into a single "Flood" class.
3. DDoS vs. DoS, run on the Flood rows only.

Each stage uses two model tracks: a **Heavy** track (XGBoost, LightGBM, CatBoost, Random Forest) and a **Lite** track (Decision Tree and related models).

Two candidate data sources exist for the third stage:

- **wataiData**: the raw CICIoT2023 export, with 46 features [28].
- **merged**: a dataset with 39 features [27].

The two schemas turn out not to be a clean subset/superset of each other (Table 7).

**Table 7: Schema differences between the two datasets**

| | Count | Columns |
|---|---:|---|
| Common to both | 37 | — |
| Merged-only | 2 | `Time_To_Live`, `IGMP` |
| wataiData-only | 9 | `flow_duration`, `Duration`, `Srate`, `Drate`, `urg_count`, `Magnitude`, `Radius`, `Covariance`, `Weight` |

**Our starting hypothesis:** the nine wataiData-only engineered features, especially `Srate` and `Drate` (which are literally rate-of-attack statistics), explain why wataiData does so much better than merged at DDoS-vs-DoS classification.

Every experiment below was designed to test that hypothesis. When it failed, the goal became finding the real explanation.

---

## 2. Method

All the scripts behind these experiments were run with `sample_frac_watai = sample_frac_merged = 0.5`, meaning a uniform 50% random sample of each source.

Sampling both sources at the same rate matters. Otherwise merged could end up exhaustively loaded while wataiData was only partly sampled, which would distort duplicate-rate comparisons, row-count-dependent statistics, and any claim that one dataset is more diverse or better populated than the other.

Unless stated otherwise, every row count, proportion, and duplicate rate below reflects this 0.5/0.5 sampling scheme.

---

## Experiment 1: Controlled Feature Removal (Testing the Srate/Drate Hypothesis)

Using wataiData rows only, trained and evaluated the same models on three feature sets.

**Table 8: Effect of removing feature subsets from the 46-feature wataiData set**

| Variant | Features | Overall macro-F1 (XGBoost) | DDoS-vs-DoS macro-F1 (XGBoost) |
|---|---:|---:|---:|
| All 46 | 46 | 0.8253 | 0.9995 |
| Minus all 9 wataiData-only features | 37 | 0.7975 (−0.028) | 0.9995 (−0.00004) |
| Minus only `Srate`/`Drate` | 44 | 0.8238 (−0.0015) | 0.9995 (+0.00001) |

**Finding:** Taking out `Srate`/`Drate`, or even all nine wataiData-only engineered columns, barely changes DDoS-vs-DoS separability. The idea that these rate features drive wataiData's near-perfect score isn't supported. Whatever produces the 99.95% figure, it survives their complete removal.

---

## Experiment 2: Cross-Dataset Decomposition

**Design:** Wanted to separate "the effect of missing features" from "everything else that differs between the two sources." To do that, compared three conditions:

- **A:** wataiData, all 46 features.
- **B:** wataiData, restricted to the 37 features shared with merged (same rows as A).
- **C:** merged, real data, all 39 features.

The observed gap, `total_drop = A − C`, splits into two parts:

- `feature_effect = A − B`: caused purely by which columns are available.
- `residual_effect = B − C`: everything else (different rows, different source, different computation).

**Table 9: Decomposing the A-to-C gap into feature effect and residual effect**

| Subtask | Model | A | B | C | Total drop | Feature effect | Residual effect |
|---|---|---:|---:|---:|---:|---:|---:|
| Overall (8-class) | XGBoost | 0.825 | 0.798 | 0.665 | 0.160 | 0.028 (17%) | 0.132 (83%) |
| Overall (8-class) | Decision Tree | 0.735 | 0.735 | 0.623 | 0.112 | 0.0002 (0.2%) | 0.112 (99.8%) |
| DDoS-vs-DoS | XGBoost | 0.9995 | 0.9995 | 0.7356 | 0.264 | 0.00004 (0.02%) | 0.264 (~100%) |
| DDoS-vs-DoS | Decision Tree | 0.9997 | 0.9997 | 0.7228 | 0.277 | ~0 | 0.277 (~100%) |

**Finding:** For DDoS-vs-DoS specifically, essentially 100% of the gap is residual, meaning it has nothing to do with which columns are present. B and C have nearly the same feature width, the same task, and the same class definitions. Yet B still scores near-perfectly while C collapses to about 73%. Feature availability is definitively ruled out as the cause for this subtask.

---

## Experiment 3: Per-Feature Leave-One-Out and Importance Ranking

**Design:** Removed each of the nine wataiData-only features one at a time from the full 46-feature set, re-evaluated with XGBoost, and ranked the features by how much macro-F1 dropped.

**Table 10: Macro-F1 change when each wataiData-only feature is removed individually (XGBoost), ranked by drop**

| Feature removed | Overall macro-F1 drop | DDoS-vs-DoS macro-F1 drop |
|---|---:|---:|
| `flow_duration` | 0.0172 | ~0 |
| `urg_count` | 0.0063 | ~0 |
| `Duration` | 0.0052 | ~0 |
| `Drate` | 0.0037 | ~0 |
| `Srate` | 0.0024 | ~0 |
| `Magnitude` (spelled "Magnitue" in the raw export) | 0.0021 | ~0 |
| `Radius` | 0.0020 | ~0 |
| `Covariance` | 0.0011 | ~0 |
| `Weight` | −0.0004 (removing it helps slightly) | ~0 |

A Gini-importance ranking from a single decision tree (all 46 features) put `Magnitude` at the top (0.191), even though its leave-one-out effect is modest. That fits with it being collinear with other statistics that stay in the model.

**Finding:** No single wataiData-only feature, `Srate` and `Drate` included, has a meaningful effect on DDoS-vs-DoS in isolation. `flow_duration` is the most useful of the nine for the overall 8-class task, roughly 5–7x more impactful than `Srate` or `Drate`, but none of the nine move the DDoS-vs-DoS number.

---

## Experiment 4: Comprehensive Dataset-Difference Investigation

With feature availability ruled out, ran a purely statistical, model-free comparison of the two sources across ten diagnostic categories. The most informative ones are below.

### Volume and class balance

**Table 11: Class volume and balance, including the share of exact-duplicate rows before deduplication**

| | wataiData | merged |
|---|---:|---:|
| Total rows (post-clean) | 15,011,042 | 11,137,735 |
| DDoS | 63.7% | 59.9% |
| DoS | 22.3% | 19.5% |
| Mirai | 7.36% | 10.84% |
| Benign | 3.66% | 4.70% |
| Recon | 1.18% | 2.94% |
| Spoofing | 1.62% | 1.97% |
| Web | 0.08% | 0.11% |
| BruteForce | 0.04% | 0.06% |
| Duplicate row fraction (raw, pre-dedup) | 35.7% | 50.5% |

Class proportions are broadly similar, though merged carries proportionally more Mirai and Recon. Both sources contain a lot of exact-duplicate rows before deduplication, and merged has more. That compounds the loss of DDoS/DoS training diversity in merged (about 60% of its rows are DDoS-labelled), but it's a secondary factor compared with what follows.

### Feature-level scale mismatch: IAT and Number

**Table 12: Scale comparison of `IAT` and `Number`**

| Feature | wataiData mean (DDoS/DoS rows) | merged mean (DDoS/DoS rows) | KS statistic |
|---|---:|---:|---:|
| `IAT` | 83,115,680 | 0.000176 | 0.9997 |
| `Number` | 9.498 | 99.896 | 0.9998 |

`IAT` differs by roughly eight orders of magnitude between the two sources, which is far too large to be a unit-conversion difference. `Number` (packet-window size) differs by about 10x, which fits the two sources capturing DDoS/DoS traffic at different effective window sizes.

### Single-feature discriminative power for DDoS-vs-DoS

**Table 13: Single-feature AUC for DDoS-vs-DoS across the shared features**

| Feature | wataiData AUC | merged AUC | Δ |
|---|---:|---:|---:|
| `IAT` | 0.9973 | 0.5758 | 0.4216 |
| `Header_Length` | 0.6446 | 0.5482 | 0.0964 |
| `ack_flag_number` | 0.5420 | 0.5212 | 0.0208 |
| All other shared features | ≤ 0.57 | ≤ 0.58 | ≤ 0.01 |

`IAT` alone accounts for almost all of the discriminative signal wataiData has for this task. No other shared feature comes close in either dataset.

### Correlation-structure divergence

Several feature pairs relate to each other in opposite ways in the two sources.

**Table 14: Feature pairs whose correlation differs in direction or strength between the datasets**

| Pair | wataiData r | merged r |
|---|---:|---:|
| `IAT` × `Number` | 0.996 | −0.068 |
| `fin_flag_number` × `ack_count` | 0.960 | −0.095 |
| `rst_flag_number` × `rst_count` | −0.032 | 0.990 |
| `Header_Length` × `Tot sum` | 0.403 | −0.448 (sign flip) |

This pattern spans several unrelated feature pairs, not just `IAT`. That suggests the two sources were produced by different feature-extraction implementations or versions, with at least some columns not sharing a definition despite sharing a name.

### Merged-only columns (`Time_To_Live`, `IGMP`)

**Table 15: Zero-fraction, unique values, and average one-vs-rest AUC for the two merged-only columns**

| Feature | Zero-fraction | N unique | Avg one-vs-rest AUC (8-class) |
|---|---:|---:|---:|
| `Time_To_Live` | 0.002% | 4,612 | 0.672 (moderately informative) |
| `IGMP` | 99.86% | 8 | 0.501 (uninformative) |

Neither is a significant confound. `Time_To_Live` carries some modest general signal, and `IGMP` is almost always zero and contributes nothing.

### Model-free class-separability check

Computed a standardized centroid-distance to average intra-class-spread ratio for the DDoS/DoS pair in each dataset: **wataiData = 2.01, merged = 2.48** (merged slightly higher).

This crude, aggregate metric did *not* show a structural separability collapse. That's a useful negative result, because it tells us the effect is concentrated almost entirely in one feature (`IAT`) rather than spread across the 37-dimensional space, where a whole-space geometric metric would simply dilute it.

---

## Experiment 5: Leakage Test

The extreme, tightly clustered per-file `IAT` values first suggested a session/file-identity leak: that `IAT` might act as a proxy for which capture file a row came from, and that a random row-level train/test split would let that proxy leak across the split. tested this directly.

**Design:** Tagged wataiData rows with their source `part-*.csv` filename. Used a `StratifiedGroupKFold` split, which guarantees no file's rows ever appear in both train and test, and compared it against the conventional random row-level split, both with and without `IAT`/`Number`.

### Leakage diagnostic

**Table 16: Per-file label purity, single-class file fraction, and share of `IAT` variance explained by source file**

| Metric | Value |
|---|---:|
| Mean label purity per source file | 0.8034 (σ = 0.0014) |
| Fraction of files that are 100% one class | 0.0% |
| Share of `IAT` variance explained by file | −0.0000011 (~0) |

Every one of the 169 source files is about 80.3% DDoS and 19.7% DoS, with almost no file-to-file variation. That's consistent with these files being Spark-shuffled output partitions, not session- or capture-run-homogeneous blocks. `IAT`'s variance sits almost entirely *within* files, so there's no file-level structure for a group-aware split to neutralize.

### Results

These figures come after correcting how duplicate rows were handled. The earlier handling had inflated wataiData's row count and shifted its "minus IAT" numbers. Merged was unaffected: its numbers are bit-for-bit identical to the pre-fix run (0.736093 and 0.725797), which independently confirms nothing was distorted there. The "full 46" (`IAT` included) figures barely moved, because `IAT`'s effect is so large it swamps the duplicate-row question either way.

**Table 17: DDoS-vs-DoS macro-F1 with and without `IAT`, after correcting duplicate-row handling**

| Split | Features | XGBoost macro-F1 | Decision Tree macro-F1 |
|---|---|---:|---:|
| Random | Full 46 | 0.99967 | 0.99986 |
| Group-aware | Full 46 | 0.99967 | 0.99985 |
| Random | Minus `IAT` | 0.83842 | 0.83740 |
| Group-aware | Minus `IAT` | 0.83802 | 0.83356 |
| Random | Minus `IAT` & `Number` | 0.83801 | 0.83739 |
| Group-aware | Minus `IAT` & `Number` | 0.83811 | 0.83357 |
| Merged (reference) | All 39, random | 0.73609 | 0.72580 |
| Merged (reference) | All 39, group-aware | 0.73607 | 0.72538 |

**Finding:** Random and group-aware splits are statistically indistinguishable, with or without `IAT`, and for both datasets. File/session-identity leakage is ruled out as the mechanism for wataiData and for merged. (Merged's own per-file check shows mean label purity of 0.753 across all 63 files, with only 2×10⁻⁶ of `IAT`'s variance explained by source file.)

This is informative: it means `IAT`'s signal is a genuine per-row property of the wataiData export, not an artifact of how rows happen to be grouped into files.

Removing `IAT` alone drops XGBoost macro-F1 from about 99.97% to about 83.8% under either split. That closes roughly 61% of the original 26-percentage-point gap to the merged benchmark. The remaining gap of about 10 points may come from the correlation-structure differences in Table 14, but didn't test that. `Number` contributes nothing measurable once `IAT` is removed.

---

## Experiment 6: IAT Quantization and Distributional Forensics

With ordinary leakage ruled out, the last question was whether wataiData's `IAT` is computed from genuine per-event timestamps at all.

> **Caveat, added on review:** The correlation checks below were first presented as a validity test. But a near-perfect correlation like merged's 0.99999 with `1/Rate` is more simply read as `IAT` being arithmetically derived from `Rate` in that pipeline (the same quantity encoded twice) than as proof that merged's `IAT` reflects independently observed timestamps.

### Relationship with Rate and Duration

**Table 18: Correlation between `IAT` and Rate- and Duration-derived quantities**

| Check | Correlation |
|---|---:|
| wataiData: `IAT` × (`Number` − 1) vs. `Duration` | −0.0057 |
| wataiData: `IAT` × (`Number` − 1) vs. `flow_duration` | −0.0003 |
| wataiData: `IAT` vs. `1/Rate` | −0.0002 (log-space: −0.0071) |
| merged: `IAT` vs. `1/Rate` | 0.99999 (log-space: 0.9906) |

wataiData's `IAT` shows no measurable relationship with `Duration`, `flow_duration`, or `1/Rate`, while merged's shows an almost exact one with `1/Rate`. Neither result proves or disproves validity on its own. Merged's near-1.0 correlation is more likely arithmetic redundancy between two columns than confirmation of genuine timestamp-derived data. Keep these numbers as a descriptive observation, not as the basis for any conclusion in this report. Section 4.4.5 of the report carries the conclusion.

### Quantization

**Table 19: `IAT` quantization in wataiData and merged**

| | wataiData | merged |
|---|---:|---:|
| Rows | 20,427,487 | 8,840,528 |
| Unique `IAT` values | 23,932 | 344,434 |
| Fraction unique | 0.12% | 3.90% |
| Median gap between sorted unique values | 24.0 (exact) | 7.2×10⁻⁹ |
| Share of rows at the single most common value | 0.16% | 0.01% |

Each unique `IAT` value in wataiData repeats about 854 times on average, with a suspiciously exact spacing of 24 between consecutive distinct values. That isn't how a continuously computed floating-point average behaves. It looks much more like a coarse, discretized code or counter than a measured statistic. Merged's `IAT`, by contrast, shows fine-grained, continuous variation, which is what genuine computation would look like.

### Class-range overlap and the AUC-vs-summary-statistics discrepancy

**Table 20: `IAT` class-range overlap between DDoS and DoS**

| | DDoS mean | DDoS σ | DoS mean | DoS σ | Range gap |
|---|---:|---:|---:|---:|---:|
| wataiData | 83,188,540 | 1,408,120 | 82,971,980 | 1,333,974 | −99,639,944 (near-total overlap) |
| merged | 0.047 | 54.2 | 0.0057 | 3.0 | −3,287 (near-total overlap) |

At face value, this seems to contradict the 0.997 AUC in Table 13: the class means are less than 0.2 standard deviations apart, and the raw ranges overlap almost completely.

The quantization result resolves it. With only about 24,000 distinct, heavily repeated values, plus a long tail of rare extreme outliers (implied by the inflated variance), naive mean, standard-deviation, and range statistics get dominated by those outliers. The rank-based AUC, computed on a representative subsample and robust to outliers, correctly picks up that the bulk of each class's repeated values is cleanly separated.

A feature that needs outlier-robust statistics to reveal a signal that naive statistics completely hide is itself further evidence against treating it as a trustworthy, well-behaved measurement.