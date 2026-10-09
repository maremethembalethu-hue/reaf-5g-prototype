# Supplementary Material S2: Model Development, the Full Story

---

## 1. The Story in One Page

The aim was a model small enough to run on an edge node, one that decides in real time whether traffic is an attack and which kind, so that evidence acquisition can fire. Four investigations followed, in order:

1. **Which feature schema?** The official CICIoT2023 files carry 39 features, the dataset paper describes 46, and a Kaggle copy has all 46 and scores much higher. The 46-feature version was rejected because the live extraction code could not reproduce those columns faithfully.
2. **Which architecture?** A flat 34-class classifier plateaued at about 70–78% accuracy for every algorithm and barely improved with 46× more data. Four structural problems explained the plateau, and a three-stage gated pipeline was built in response.
3. **Why was DDoS-vs-DoS so unstable across data sources?** Stage 3 scored about 65–78% on one source and near 100% on another. A long investigation (S1) showed that the near-perfect score came from a single suspect feature, `IAT`, and not from the engineered rate features first suspected. Stage 3 therefore abstains when unsure.
4. **Why did live accuracy diverge from training accuracy?** [[Paste the text for Sections 4.4.6 and 4.4.7 here; not yet supplied.]]

---

## 2. Feature Schema: 39 vs. 46

The CICIoT2023 paper documents 46 features [2]. The official distribution [1] ships 39. Nine features, `Srate` and `Drate` (same-source and same-destination packet rates) among them, are computed by the official extraction script but never written to the output rows. A complete 46-feature distribution [3] was located and tested.

**Table 1: Heavy-tier results on the two feature schemas**

| Model | 39 features, official (deployed) | 46 features, Kaggle (not deployed) |
|---|---|---|
| XGBoost (Heavy) | 75.6–78.4% acc, 0.60–0.63 macro-F1 | 87.83% acc, 0.677 macro-F1 |
| LightGBM (Heavy) | 70.1–74.4% acc, 0.55–0.62 macro-F1 | 99.30% acc, 0.794 macro-F1 |
| CatBoost | 73.9–76.3% acc, 0.58–0.59 macro-F1 | 86.48% acc, 0.618 macro-F1 |
| Random Forest | 73.0–75.7% acc, 0.57–0.61 macro-F1 | 86.61% acc, 0.617 macro-F1 |

On the Lite tier, LightGBM reached 98.80% accuracy and 0.746 macro-F1 with 46 features.

**Decision: the 46-feature schema was not adopted.** A feature-parity audit of the extraction code needed to reproduce those 46 columns live turned up confirmed differences between what several column names claim and what is actually computed. The 39-feature schema is the official distribution, and its main limitation is characterised in this document.

[[Add the audit details and the 46× training-row experiment (3.37-point gain) from the Feature-Schema Selection write-up.]]

---

## 3. Flat-Model Baseline

Four Heavy-tier and up to six Lite-tier candidates were trained on one undersampled, stratified split of the 34-class task, then evaluated once on a held-out test set.

### Heavy tier

On the 39-feature schema, XGBoost led at every data scale by a small but stable margin. On the 46-feature schema, LightGBM led XGBoost by about 11.5 accuracy points. That gap did not override the schema decision, but it is reported for completeness.

### Lite tier

Results mix the two schemas, so rows are **not** comparable across schemas.

**Table 2: Lite-tier candidates on the flat 34-class task**

| Model | Accuracy | Macro-F1 | Schema | Notes |
|---|---:|---:|:---:|---|
| Tiny LightGBM | 98.8% | 0.746 | 46 | Best Lite score; not deployed because the 46-feature schema was not adopted |
| Decision Tree | 46.1–52.0% | 0.40–0.41 | 39 | Deployed candidate |
| Random Forest | 73.0% | 0.472 | 46 | ONNX export risk [[explain what the risk is]] |
| MLP | 61.2% (fp32) | 0.310 | 46 | Fell to 53.9% after INT8 quantisation; several classes dropped to exactly 0.00 recall |
| Extra Trees | 62.2% | 0.376 | 46 | — |
| Logistic Regression | 57.8% | 0.264 | 46 | — |

### Convergence

On the 39-feature data, all four Heavy algorithms landed at a similar, mediocre macro-F1 regardless of algorithm or data volume. A 46× increase in training rows moved accuracy by only 3.37 points.

> Convergence across different model families, unmoved by more data, points to an **information-limited** problem: performance is capped by what the model can see, not by how well the model learns.

---

## 4. Why a Flat Classifier Was the Wrong Shape

1. **One hard subtask dragged down 33 others.** DDoS-vs-DoS is hard in this dataset, with an estimated ceiling of roughly 65–84% depending on which artefact-corrected estimate is used (S1). In a flat model, errors on that subtask are averaged into the same macro-F1 as every other class.
2. **No single window size.** The CICIoT2023 methodology captures DDoS, DoS, and Mirai subtypes with 100-packet windows, and all other classes (Benign included) with 10-packet windows. A flat model accepts one feature-vector shape, so that split cannot be represented.
3. **Decisions differ in importance and cost.** Attack vs. benign is the most important decision and should be the most reliable. A flat model gives every decision the same single pass.

---

## 5. The Three-Stage Pipeline

**Table 3: Design response to each structural problem**

| Problem | Design response |
|---|---|
| DDoS-vs-DoS drags down everything else | Isolate the subtask as the last stage |
| No single window size fits all classes | Run two windows in parallel; the 100-packet result overrides only for Flood and Mirai |
| Decisions differ in cost and confidence | Use a sequence of increasingly specific questions, each with a confidence threshold of its own |

### The stages

- **Stage 1: Attack vs. Benign.** Always at 10 packets. Determines whether evidence acquisition fires.
- **Stage 2: Six families.** BruteForce, Flood (DDoS + DoS merged), Mirai, Recon, Spoofing, and Web. Runs at 10 packets by default, overridden by the 100-packet evaluation when that evaluation confidently predicts Flood or Mirai.
- **Stage 3: DDoS vs. DoS.** Runs only on rows that Stage 2 called Flood. If the confidence margin falls below `MIN_MARGIN = 0.20`, the pipeline reports `Flood_uncertain` instead of guessing.

Each stage runs only on the rows routed to it (Stage 3 never sees non-Flood traffic). Each stage's confidence decides whether the next answer is trusted or a lower-confidence fallback is reported.

**Table 4: Per-stage accuracy (isolated evaluation)**

| Stage | Task | Accuracy |
|:---:|---|---:|
| 1 | Attack vs. Benign | 95.4–95.7% |
| 2 | Six families, all rows (DDoS + DoS merged) | 87.4–89.4% |
| 2 | Six families, attack rows only | 89.2–91.5% |
| 3 | DDoS vs. DoS | 65.5–78% |

---

## 6. Where to Find the Rest

| Topic | Location |
|---|---|
| DDoS-vs-DoS investigation | S1, `S1_DDoS_vs_DoS_Investigation.md` |
| Live vs. training divergence | [[Section 4.4.6–4.4.7 text]] |
| Code and results | [[GitHub URL, commit SHA]] |

---

## References

- **[1]** CICIoT2023 official distribution.
- **[2]** E. C. P. Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors*, 2023.
- **[3]** Kaggle 46-feature CICIoT2023 distribution.