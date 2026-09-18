

import glob
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

GPU_AVAILABLE = False
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass
if not GPU_AVAILABLE:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
        GPU_AVAILABLE = out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        pass
XGB_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
print(f"XGBoost device: {XGB_DEVICE}")

WATAI_ROOT = Path("data/wataiData/csv/CICIoT2023")
MERGED_ROOT = Path("data/MERGED_CSV/MERGED_CSV")

OUTPUT_DIR = Path(f"outputs/exp3_missing_features_{RUN_ID}")
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
for _d in (OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
MASTER_CSV = RESULTS_DIR / "exp3_master_results.csv"
_rows = []


def save_row(row):
    row = {"run_id": RUN_ID, "timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    _rows.append(row)
    pd.DataFrame(_rows).to_csv(MASTER_CSV, index=False)


def save_plot(fig, name):
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    path = PLOTS_DIR / f"{safe}.png"
    i = 1
    while path.exists():
        path = PLOTS_DIR / f"{safe}_{i}.png"
        i += 1
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"P {path}")


#  feature schema 
MERGED_39_FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]
WATAI_46_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]
_m, _w = set(MERGED_39_FEATURES), set(WATAI_46_FEATURES)
COMMON_FEATURES = [f for f in MERGED_39_FEATURES if f in _w]           # 37
MERGED_ONLY_FEATURES = [f for f in MERGED_39_FEATURES if f not in _w]  # Time_To_Live, IGMP
WATAI_ONLY_FEATURES = [f for f in WATAI_46_FEATURES if f not in _m]    # the 9 missing-in-39 features
SRATE_DRATE = [f for f in ("Srate", "Drate") if f in _w]
WATAI_MINUS_MISSING_37 = [f for f in WATAI_46_FEATURES if f not in WATAI_ONLY_FEATURES]  # == COMMON_FEATURES
WATAI_MINUS_SRATE_DRATE_44 = [f for f in WATAI_46_FEATURES if f not in SRATE_DRATE]
print(f"common={len(COMMON_FEATURES)} merged_only={MERGED_ONLY_FEATURES} watai_only={WATAI_ONLY_FEATURES}")

ATTACK_FAMILY_MAP = {
    "Backdoor_Malware": "Web", "BrowserHijacking": "Web", "CommandInjection": "Web",
    "SqlInjection": "Web", "Uploading_Attack": "Web", "XSS": "Web",
    "DDoS-ACK_Fragmentation": "DDoS", "DDoS-HTTP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS", "DDoS-PSHACK_FLOOD": "DDoS", "DDoS-RSTFINFLOOD": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS", "DDoS-UDP_Fragmentation": "DDoS",
    "DictionaryBruteForce": "BruteForce",
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    "DoS-HTTP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-UDP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    "Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon", "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon", "VulnerabilityScan": "Recon",
    "Benign_Final": "Benign", "BENIGN": "Benign", "BenignTraffic": "Benign",
}
FLOOD_MERGE = {"DDoS": "Flood", "DoS": "Flood"}
FLOOD_CLASSES = ("DDoS", "DoS")


#  loading 
def find_label_col(df):
    for c in ("label", "Label", "LABEL"):
        if c in df.columns:
            return c
    raise KeyError("no label column found")


def _downcast(df):
    fcols = df.select_dtypes(include=["float64"]).columns
    if len(fcols):
        df[fcols] = df[fcols].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def load_watai(root=WATAI_ROOT, sample_frac=None, chunksize=500_000):
    files = sorted(glob.glob(str(root / "part-*.csv")))
    if not files:
        print(f"[WARN] no part-*.csv under {root}")
        return pd.DataFrame()
    frames = []
    for i, fpath in enumerate(files):
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            frames.append(_downcast(chunk))
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  watai: {i + 1}/{len(files)} files")
    full = pd.concat(frames, ignore_index=True)
    lc = find_label_col(full)
    if lc != "label":
        full = full.rename(columns={lc: "label"})
    print(f"watai: {len(full):,} rows, ~{full.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return full


def load_merged(root=MERGED_ROOT, start=1, end=63, sample_frac=None, chunksize=500_000):
    files = [f for f in (root / f"Merged{i:02d}.csv" for i in range(start, end + 1)) if f.exists()]
    if not files:
        print(f"[WARN] no Merged*.csv under {root}")
        return pd.DataFrame()
    frames = []
    for i, fpath in enumerate(files):
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            frames.append(_downcast(chunk))
        print(f"  merged: {i + 1}/{len(files)} files ({fpath.name})")
    full = pd.concat(frames, ignore_index=True)
    lc = find_label_col(full)
    if lc != "label":
        full = full.rename(columns={lc: "label"})
    print(f"merged: {len(full):,} rows, ~{full.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return full


#  preprocessing 
def clean(df):
    before = len(df)
    df = df.drop_duplicates()
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"clean: {before:,} -> {len(df):,} rows")
    if "label" in df.columns:
        df["label"] = df["label"].replace({"BenignTraffic": "Benign_Final", "BENIGN": "Benign_Final"})
    return df.reset_index(drop=True)


def encode_protocol(df, col="Protocol Type"):
    if col not in df.columns:
        return df, None
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df, le


def map_family(df, label_col="label"):
    nm = {k.upper(): v for k, v in ATTACK_FAMILY_MAP.items()}
    mapped = df[label_col].astype(str).str.upper().map(nm)
    unmapped = df.loc[mapped.isna(), label_col].unique().tolist()
    if unmapped:
        print(f"[WARN] unmapped labels: {unmapped[:10]}")
    df[label_col] = mapped.fillna(df[label_col])
    return df


def cap_rows(df, label_col, max_rows, rs=RANDOM_STATE):
    if max_rows is None or len(df) <= max_rows:
        return df
    sampled, _ = train_test_split(df, train_size=max_rows, stratify=df[label_col], random_state=rs)
    return sampled.reset_index(drop=True)


def undersample(df, label_col="y", min_per_class=1000, rs=RANDOM_STATE):
    counts = df[label_col].value_counts()
    cap = max(min_per_class, int(counts.median()))
    parts = []
    for cls, n in counts.items():
        cdf = df[df[label_col] == cls]
        parts.append(cdf.sample(n=cap, random_state=rs) if n > cap else cdf)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=rs).reset_index(drop=True)
    print(f"undersample: {len(df):,} -> {len(out):,} (cap={cap:,})")
    return out


def prepare(raw_df, min_per_class=1000, max_eval_rows=300_000):
    df = clean(raw_df)
    df, _ = encode_protocol(df)
    df = map_family(df)
    train_full, test = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=RANDOM_STATE)
    train, val = train_test_split(train_full, test_size=0.125, stratify=train_full["label"], random_state=RANDOM_STATE)
    train, val, test = (x.reset_index(drop=True) for x in (train, val, test))
    val, test = cap_rows(val, "label", max_eval_rows), cap_rows(test, "label", max_eval_rows)
    le = LabelEncoder()
    train["y"] = le.fit_transform(train["label"])
    val["y"] = le.transform(val["label"])
    test["y"] = le.transform(test["label"])
    train = undersample(train, "y", min_per_class)
    return {"train": train, "val": val, "test": test, "label_encoder": le}


def fit_scale(train, feats):
    s = StandardScaler()
    s.fit(train[feats])
    return s


def apply_scale(df, feats, scaler):
    extra = [c for c in ("label", "y") if c in df.columns and c not in feats]
    out = df[list(feats) + extra].copy()
    out[feats] = scaler.transform(out[feats])
    return out


def sample_weights(y, max_ratio=40.0):
    counts = np.bincount(np.asarray(y))
    w = len(y) / (len(counts) * counts)
    w = np.clip(w, w.min(), w.min() * max_ratio)
    return w[np.asarray(y)]


#  models 
def train_xgb(Xtr, ytr, Xval, yval, num_class):
    obj = {"objective": "binary:logistic", "eval_metric": "logloss"} if num_class == 2 else \
          {"objective": "multi:softprob", "num_class": num_class, "eval_metric": "mlogloss"}
    model = xgb.XGBClassifier(**obj, n_estimators=300, max_depth=8, learning_rate=0.08,
                               subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                               reg_lambda=2.0, tree_method="hist", device=XGB_DEVICE,
                               random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(Xtr, ytr, sample_weight=sample_weights(ytr), eval_set=[(Xval, yval)], verbose=False)
    return model


def train_dt(Xtr, ytr):
    model = DecisionTreeClassifier(criterion="gini", max_depth=15, min_samples_split=10,
                                    min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    return model


MODEL_BUILDERS = {
    "xgboost_heavy": lambda Xtr, ytr, Xval, yval, nc: train_xgb(Xtr, ytr, Xval, yval, nc),
    "decision_tree_lite": lambda Xtr, ytr, Xval, yval, nc: train_dt(Xtr, ytr),
}


def evaluate(model, X, y, name):
    t0 = time.perf_counter()
    pred = model.predict(X)
    lat = (time.perf_counter() - t0) / max(len(X), 1) * 1e6
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    roc = None
    try:
        proba = model.predict_proba(X)
        roc = roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2 else \
              roc_auc_score(y, proba, multi_class="ovr", average="macro")
    except Exception:
        pass
    return {"model": name, "n": len(X), "accuracy": acc, "macro_precision": p, "macro_recall": r,
            "macro_f1": f1, "roc_auc_macro_ovr": roc, "latency_us": lat}, pred


def plot_bar(labels, series, ylabel, title, name):
    x = np.arange(len(labels))
    w = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    for i, (k, v) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, v, w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    plt.tight_layout()
    save_plot(fig, name)


def plot_confusion(y, pred, classes, title, name):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    save_plot(fig, name)


#  shared train/eval helper 
def run_variant(prepared, feats, tag, part, extra_tags=None):
    # Trains every model in MODEL_BUILDERS on `feats`, evaluates overall
    # (flat) and on the DDoS-vs-DoS Flood-only subset. 
    train, val, test = prepared["train"], prepared["val"], prepared["test"]
    le = prepared["label_encoder"]
    num_class = len(le.classes_)
    scaler = fit_scale(train, feats)
    train_s, val_s, test_s = apply_scale(train, feats, scaler), apply_scale(val, feats, scaler), \
                              apply_scale(test, feats, scaler)

    flood_train = train_s[train_s["label"].isin(FLOOD_CLASSES)]
    flood_val = val_s[val_s["label"].isin(FLOOD_CLASSES)]
    flood_test = test_s[test_s["label"].isin(FLOOD_CLASSES)]
    flood_le = LabelEncoder().fit(flood_train["label"])
    fy_tr = pd.Series(flood_le.transform(flood_train["label"]))
    fy_va = pd.Series(flood_le.transform(flood_val["label"]))
    fy_te = pd.Series(flood_le.transform(flood_test["label"]))

    out = {}
    for mname, builder in MODEL_BUILDERS.items():
        model = builder(train_s[feats], train_s["y"], val_s[feats], val_s["y"], num_class)
        m_overall, pred_overall = evaluate(model, test_s[feats], test_s["y"], f"{tag}_{mname}")
        row = {"experiment": part, "tag": tag, "model": mname, "subtask": "overall_8class",
               "n_features": len(feats)}
        row.update(m_overall)
        if extra_tags:
            row.update(extra_tags)
        save_row(row)

        fmodel = builder(flood_train[feats], fy_tr, flood_val[feats], fy_va, 2)
        m_flood, pred_flood = evaluate(fmodel, flood_test[feats], fy_te, f"{tag}_{mname}_flood")
        frow = {"experiment": part, "tag": tag, "model": mname, "subtask": "ddos_vs_dos",
                "n_features": len(feats)}
        frow.update(m_flood)
        if extra_tags:
            frow.update(extra_tags)
        save_row(frow)

        out[mname] = {"overall": m_overall, "flood": m_flood,
                       "overall_pred": pred_overall, "flood_pred": pred_flood,
                       "flood_test_y": fy_te, "flood_classes": list(flood_le.classes_)}
    return out


#  PART 1 
def part1_feature_removal_ablation(watai_raw, min_per_class=1000, max_eval_rows=300_000):
    print("PART 1: remove the 39-feature-missing columns from the 46-feature data" )
    prepared = prepare(watai_raw, min_per_class, max_eval_rows)
    variants = {
        "watai_all46": WATAI_46_FEATURES,
        "watai_minus_9missing_37": WATAI_MINUS_MISSING_37,
        "watai_minus_srate_drate_44": WATAI_MINUS_SRATE_DRATE_44,
    }
    results = {}
    for tag, feats in variants.items():
        results[tag] = run_variant(prepared, feats, tag, "part1_feature_removal")
        r = results[tag]["xgboost_heavy"]
        plot_confusion(r["flood_test_y"], r["flood_pred"], r["flood_classes"],
                        f"Part 1: {tag} — DDoS vs DoS (XGBoost)", f"part1__{tag}__flood_confusion")

    for subtask in ("overall", "flood"):
        series = {m: [results[t][m][subtask]["macro_f1"] for t in variants] for m in MODEL_BUILDERS}
        plot_bar(list(variants), series, "macro F1", f"Part 1: feature removal — {subtask}",
                  f"part1__{subtask}__macro_f1_by_variant")
    return results


#  PART 2 
def part2_cross_dataset_decomposition(watai_raw, merged_raw, min_per_class=1000, max_eval_rows=300_000):
    print("PART 2: decompose the real 46 vs 39 gap (feature effect vs residual)")
    prepared_w = prepare(watai_raw, min_per_class, max_eval_rows)
    prepared_m = prepare(merged_raw, min_per_class, max_eval_rows)

    A = run_variant(prepared_w, WATAI_46_FEATURES, "A_watai_all46", "part2_decomposition")
    B = run_variant(prepared_w, WATAI_MINUS_MISSING_37, "B_watai_common37", "part2_decomposition")
    C = run_variant(prepared_m, MERGED_39_FEATURES, "C_merged_real39", "part2_decomposition")

    decomp_rows = []
    for subtask in ("overall", "flood"):
        for mname in MODEL_BUILDERS:
            fa, fb, fc = A[mname][subtask]["macro_f1"], B[mname][subtask]["macro_f1"], C[mname][subtask]["macro_f1"]
            total_drop = fa - fc
            feature_effect = fa - fb
            residual_effect = fb - fc
            decomp_rows.append({"subtask": subtask, "model": mname,
                                 "macro_f1_A_watai46": fa, "macro_f1_B_watai_common37": fb,
                                 "macro_f1_C_merged_real39": fc,
                                 "total_drop_A_minus_C": total_drop,
                                 "feature_effect_A_minus_B": feature_effect,
                                 "residual_effect_B_minus_C": residual_effect,
                                 "feature_effect_share_of_total": feature_effect / total_drop if total_drop else np.nan})
            print(f"[{subtask}|{mname}] A={fa:.4f} B={fb:.4f} C={fc:.4f} | "
                  f"total_drop={total_drop:.4f} feature_effect={feature_effect:.4f} "
                  f"residual_effect={residual_effect:.4f}")

    decomp_df = pd.DataFrame(decomp_rows)
    decomp_path = RESULTS_DIR / "part2_decomposition_summary.csv"
    decomp_df.to_csv(decomp_path, index=False)
    print(f"csv {decomp_path}")

    for subtask in ("overall", "flood"):
        sub = decomp_df[decomp_df["subtask"] == subtask]
        plot_bar(sub["model"].tolist(),
                  {"A_watai_all46": sub["macro_f1_A_watai46"].tolist(),
                   "B_watai_common37": sub["macro_f1_B_watai_common37"].tolist(),
                   "C_merged_real39": sub["macro_f1_C_merged_real39"].tolist()},
                  "macro F1", f"Part 2: {subtask} — A vs B vs C", f"part2__{subtask}__ABC_macro_f1")
        plot_bar(sub["model"].tolist(),
                  {"feature_effect (A-B)": sub["feature_effect_A_minus_B"].tolist(),
                   "residual_effect (B-C)": sub["residual_effect_B_minus_C"].tolist()},
                  "macro F1 drop", f"Part 2: {subtask} — decomposition of the total drop",
                  f"part2__{subtask}__decomposition")

    for mname in MODEL_BUILDERS:
        r = C[mname]
        plot_confusion(r["flood_test_y"], r["flood_pred"], r["flood_classes"],
                        f"Part 2: merged real 39-feature — DDoS vs DoS ({mname})",
                        f"part2__merged_real39__{mname}__flood_confusion")
    return decomp_df


#  PART 3 
def part3_per_feature_loo(watai_raw, min_per_class=1000, max_eval_rows=300_000):
    print("PART 3: leave-one-out over the 9 missing-in-39 features")
    prepared = prepare(watai_raw, min_per_class, max_eval_rows)
    baseline = run_variant(prepared, WATAI_46_FEATURES, "baseline_all46", "part3_loo")

    loo_rows = []
    for feat in WATAI_ONLY_FEATURES:
        feats = [f for f in WATAI_46_FEATURES if f != feat]
        res = run_variant(prepared, feats, f"minus_{feat}", "part3_loo", extra_tags={"removed_feature": feat})
        for subtask in ("overall", "flood"):
            for mname in MODEL_BUILDERS:
                base_f1 = baseline[mname][subtask]["macro_f1"]
                new_f1 = res[mname][subtask]["macro_f1"]
                loo_rows.append({"removed_feature": feat, "subtask": subtask, "model": mname,
                                  "baseline_macro_f1": base_f1, "after_removal_macro_f1": new_f1,
                                  "drop": base_f1 - new_f1})

    loo_df = pd.DataFrame(loo_rows)
    loo_path = RESULTS_DIR / "part3_leave_one_out_summary.csv"
    loo_df.to_csv(loo_path, index=False)
    print(f"csv {loo_path}")

    for subtask in ("overall", "flood"):
        for mname in MODEL_BUILDERS:
            sub = loo_df[(loo_df["subtask"] == subtask) & (loo_df["model"] == mname)].sort_values(
                "drop", ascending=False)
            plot_bar(sub["removed_feature"].tolist(), {"macro_f1 drop when removed": sub["drop"].tolist()},
                      "macro F1 drop", f"Part 3: {subtask} ({mname}) — per-feature removal impact",
                      f"part3__{subtask}__{mname}__per_feature_drop")

    # Gini importance of the 9 watai-only features inside the full-46 XGBoost model
    train_s = apply_scale(prepared["train"], WATAI_46_FEATURES, fit_scale(prepared["train"], WATAI_46_FEATURES))
    dt = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    dt.fit(train_s[WATAI_46_FEATURES], train_s["y"])
    importances = pd.Series(dt.feature_importances_, index=WATAI_46_FEATURES)
    watai_only_importances = importances[WATAI_ONLY_FEATURES].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(watai_only_importances.index[::-1], watai_only_importances.values[::-1])
    ax.set_xlabel("Gini importance"); ax.set_title("Part 3: Gini importance of the 9 missing-in-39 features")
    plt.tight_layout()
    save_plot(fig, "part3__gini_importance_watai_only_features")
    watai_only_importances.to_csv(RESULTS_DIR / "part3_gini_importance_watai_only.csv")

    return loo_df


#  main 
def main(sample_frac_watai=0.5, sample_frac_merged=0.5, min_per_class=1000, max_eval_rows=300_000):
    print(f"Run ID: {RUN_ID} | output: {OUTPUT_DIR.resolve()}")
    watai_raw = load_watai(sample_frac=sample_frac_watai)
    merged_raw = load_merged(sample_frac=sample_frac_merged)
    if not len(watai_raw):
        raise RuntimeError("watai dataset could not be loaded — check WATAI_ROOT")

    part1_feature_removal_ablation(watai_raw, min_per_class, max_eval_rows)
    if len(merged_raw):
        part2_cross_dataset_decomposition(watai_raw, merged_raw, min_per_class, max_eval_rows)
    else:
        print("S Part 2 — merged dataset not available")
    part3_per_feature_loo(watai_raw, min_per_class, max_eval_rows)

    print(f"\nDone. Master CSV: {MASTER_CSV.resolve()}")
    print(f"Plots: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main(sample_frac_watai=0.5 ,sample_frac_merged=0.5, min_per_class=1000, max_eval_rows=300_000)