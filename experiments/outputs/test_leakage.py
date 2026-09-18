
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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
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
XGB_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"

WATAI_ROOT = Path("data/wataiData/csv/CICIoT2023")
MERGED_ROOT = Path("data/MERGED_CSV/MERGED_CSV")

OUTPUT_DIR = Path(f"outputs/leakage_test_{RUN_ID}")
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
for _d in (OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
_rows = []


def save_row(row):
    row = {"run_id": RUN_ID, "timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    _rows.append(row)
    pd.DataFrame(_rows).to_csv(RESULTS_DIR / "leakage_test_master_results.csv", index=False)


def save_csv(df, name):
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[csv] {path} ({len(df)} rows)")


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
    "Magnitude", "Radius", "Covariance", "Variance", "Weight",
]
WATAI_MINUS_IAT = [f for f in WATAI_46_FEATURES if f != "IAT"]
WATAI_MINUS_IAT_NUMBER = [f for f in WATAI_46_FEATURES if f not in ("IAT", "Number")]

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
FLOOD_CLASSES = ("DDoS", "DoS")
COLUMN_RENAME = {"Magnitue": "Magnitude"}


def _norm_key(name):
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def normalize_columns(df, expected):
    df = df.rename(columns=COLUMN_RENAME, copy=False)
    lookup = {_norm_key(c): c for c in df.columns}
    rename_map = {lookup[_norm_key(f)]: f for f in expected
                  if f not in df.columns and _norm_key(f) in lookup and lookup[_norm_key(f)] != f}
    if rename_map:
        df = df.rename(columns=rename_map, copy=False)
    return df


def validate_columns(df, required, name):
    missing = [f for f in required if f not in df.columns]
    if missing:
        raise KeyError(f"{name} missing expected columns: {missing}; actual: {sorted(df.columns)}")


def _find_label_col(df):
    for c in ("label", "Label", "LABEL"):
        if c in df.columns:
            return c
    raise KeyError("no label column")


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
        fname = Path(fpath).name
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk.rename(columns=COLUMN_RENAME, copy=False)
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            chunk = _downcast(chunk)
            chunk["source_file"] = fname
            frames.append(chunk)
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  watai: {i + 1}/{len(files)} files")
    full = pd.concat(frames, ignore_index=True)
    full = normalize_columns(full, WATAI_46_FEATURES)
    validate_columns(full, WATAI_46_FEATURES, "watai")
    lc = _find_label_col(full)
    if lc != "label":
        full = full.rename(columns={lc: "label"}, copy=False)
    print(f"watai: {len(full):,} rows, {full['source_file'].nunique()} source files")
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
            chunk = chunk.rename(columns=COLUMN_RENAME, copy=False)
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            frames.append(_downcast(chunk))
        print(f"  merged: {i + 1}/{len(files)} files ({fpath.name})")
    full = pd.concat(frames, ignore_index=True)
    full = normalize_columns(full, MERGED_39_FEATURES)
    validate_columns(full, MERGED_39_FEATURES, "merged")
    lc = _find_label_col(full)
    if lc != "label":
        full = full.rename(columns={lc: "label"}, copy=False)
    print(f"merged: {len(full):,} rows")
    return full


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


def add_family(df):
    nm = {k.upper(): v for k, v in ATTACK_FAMILY_MAP.items()}
    mapped = df["label"].astype(str).str.upper().map(nm)
    df["family"] = mapped.fillna(df["label"])
    return df


#  leakage diagnostic 
def leakage_diagnostic(flood_df):
    grp = flood_df.groupby("source_file")
    rows = []
    for fname, sub in grp:
        dom_frac = sub["family"].value_counts(normalize=True).iloc[0]
        rows.append({"source_file": fname, "n_rows": len(sub), "dominant_family": sub["family"].mode().iloc[0],
                     "label_purity": dom_frac, "iat_mean": sub["IAT"].mean(), "iat_std": sub["IAT"].std()})
    per_file = pd.DataFrame(rows)
    save_csv(per_file, "01_per_file_leakage_diagnostic")

    global_var = flood_df["IAT"].var()
    within_var = per_file["iat_std"].pow(2).mul(per_file["n_rows"] - 1).sum() / (len(flood_df) - len(per_file))
    eta_sq_like = 1 - (within_var / global_var) if global_var else np.nan
    summary = pd.DataFrame([{
        "metric": "mean label purity across source files (1.0 = every file is 100% one class)",
        "value": per_file["label_purity"].mean(),
    }, {
        "metric": "fraction of files with label_purity == 1.0",
        "value": (per_file["label_purity"] == 1.0).mean(),
    }, {
        "metric": "global IAT variance", "value": global_var,
    }, {
        "metric": "average within-file IAT variance", "value": within_var,
    }, {
        "metric": "share of IAT variance explained by source_file (eta-squared-like; near 1.0 = IAT is basically a file ID)",
        "value": eta_sq_like,
    }])
    save_csv(summary, "02_leakage_summary")
    print(summary.to_string(index=False))
    return per_file, summary


#  splitting 
def group_aware_split(df, label_col, group_col, test_size=0.2, rs=RANDOM_STATE):
    n_splits = max(2, round(1 / test_size))
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        train_idx, test_idx = next(sgkf.split(df, df[label_col], groups=df[group_col]))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] StratifiedGroupKFold failed ({e}); falling back to manual per-class group split")
        train_parts, test_parts = [], []
        rng = np.random.default_rng(rs)
        for cls in df[label_col].unique():
            sub = df[df[label_col] == cls]
            groups = sub[group_col].unique()
            rng.shuffle(groups)
            n_test = max(1, int(len(groups) * test_size))
            test_groups = set(groups[:n_test])
            mask = sub[group_col].isin(test_groups)
            test_parts.append(sub[mask])
            train_parts.append(sub[~mask])
        return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def random_split(df, label_col, test_size=0.2, rs=RANDOM_STATE):
    train, test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=rs)
    return train.reset_index(drop=True), test.reset_index(drop=True)


#  models 
def sample_weights(y, max_ratio=40.0):
    counts = np.bincount(np.asarray(y))
    w = len(y) / (len(counts) * counts)
    w = np.clip(w, w.min(), w.min() * max_ratio)
    return w[np.asarray(y)]


def train_xgb(Xtr, ytr):
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=300, max_depth=8,
                               learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                               reg_lambda=2.0, tree_method="hist", device=XGB_DEVICE, random_state=RANDOM_STATE,
                               n_jobs=-1)
    model.fit(Xtr, ytr, sample_weight=sample_weights(ytr))
    return model


def train_dt(Xtr, ytr):
    model = DecisionTreeClassifier(criterion="gini", max_depth=15, min_samples_split=10, min_samples_leaf=5,
                                    class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    return model


MODELS = {"xgboost_heavy": train_xgb, "decision_tree_lite": train_dt}


def evaluate(model, X, y, name):
    t0 = time.perf_counter()
    pred = model.predict(X)
    lat = (time.perf_counter() - t0) / max(len(X), 1) * 1e6
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    try:
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = np.nan
    return {"model": name, "n": len(X), "accuracy": acc, "macro_f1": f1, "auc": auc, "latency_us": lat}, pred


def fit_scale(train, feats):
    s = StandardScaler()
    s.fit(train[feats])
    return s


def apply_scale(df, feats, scaler):
    out = df[list(feats)].copy()
    out[feats] = scaler.transform(out[feats])
    return out


def plot_confusion(y, pred, classes, title, name):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    save_plot(fig, name)


def run_condition(train, test, feats, label_encoder, split_tag, feature_tag):
    for mname, builder in MODELS.items():
        scaler = fit_scale(train, feats)
        Xtr, Xte = apply_scale(train, feats, scaler), apply_scale(test, feats, scaler)
        ytr, yte = train["y"], test["y"]
        model = builder(Xtr, ytr)
        m, pred = evaluate(model, Xte, yte, f"{split_tag}__{feature_tag}__{mname}")
        m.update({"split": split_tag, "features": feature_tag, "n_features": len(feats), "model_kind": mname})
        save_row(m)
        if mname == "xgboost_heavy":
            plot_confusion(yte, pred, list(label_encoder.classes_),
                            f"{split_tag} split, {feature_tag} — DDoS vs DoS", f"{split_tag}__{feature_tag}__confusion")
        print(f"[{split_tag}|{feature_tag}|{mname}] acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} auc={m['auc']:.4f}")


def leakage_test(flood_df):
    le = LabelEncoder().fit(flood_df["family"])
    flood_df = flood_df.copy()
    flood_df["y"] = le.transform(flood_df["family"])

    conditions = {"full_46": WATAI_46_FEATURES, "minus_IAT": WATAI_MINUS_IAT,
                  "minus_IAT_Number": WATAI_MINUS_IAT_NUMBER}

    rand_train, rand_test = random_split(flood_df, "family")
    group_train, group_test = group_aware_split(flood_df, "family", "source_file")
    print(f"random split: train={len(rand_train):,} test={len(rand_test):,}")
    print(f"group split:  train={len(group_train):,} test={len(group_test):,} "
          f"(train files={group_train['source_file'].nunique()}, test files={group_test['source_file'].nunique()})")

    for feat_tag, feats in conditions.items():
        run_condition(rand_train, rand_test, feats, le, "random", feat_tag)
        run_condition(group_train, group_test, feats, le, "group", feat_tag)


def merged_reference(merged_raw):
    merged = add_family(clean(merged_raw))
    flood = merged[merged["family"].isin(FLOOD_CLASSES)].reset_index(drop=True)
    le = LabelEncoder().fit(flood["family"])
    flood["y"] = le.transform(flood["family"])
    train, test = random_split(flood, "family")
    for mname, builder in MODELS.items():
        scaler = fit_scale(train, MERGED_39_FEATURES)
        Xtr, Xte = apply_scale(train, MERGED_39_FEATURES, scaler), apply_scale(test, MERGED_39_FEATURES, scaler)
        model = builder(Xtr, train["y"])
        m, pred = evaluate(model, Xte, test["y"], f"merged_reference__{mname}")
        m.update({"split": "random", "features": "merged_all39", "n_features": 39, "model_kind": mname})
        save_row(m)
        print(f"[merged reference|{mname}] acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} auc={m['auc']:.4f}")


def summary_chart():
    df = pd.read_csv(RESULTS_DIR / "leakage_test_master_results.csv")
    xgb_rows = df[df["model_kind"] == "xgboost_heavy"]
    labels, values = [], []
    for split in ("random", "group"):
        for feat in ("full_46", "minus_IAT", "minus_IAT_Number"):
            r = xgb_rows[(xgb_rows["split"] == split) & (xgb_rows["features"] == feat)]
            if len(r):
                labels.append(f"{split}\n{feat}")
                values.append(r["macro_f1"].iloc[0])
    ref = xgb_rows[xgb_rows["features"] == "merged_all39"]
    if len(ref):
        labels.append("merged\nreference")
        values.append(ref["macro_f1"].iloc[0])

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5))
    colors = ["#4c72b0"] * 3 + ["#dd8452"] * 3 + ["#55a868"] * (len(labels) - 6)
    ax.bar(labels, values, color=colors[:len(labels)])
    ax.axhline(values[-1] if len(ref) else 0, linestyle="--", color="gray", linewidth=1)
    ax.set_ylabel("macro F1 (DDoS vs DoS)")
    ax.set_title("Leakage test: random vs group-aware split, with/without IAT")
    plt.tight_layout()
    save_plot(fig, "leakage_test_summary")


def main(sample_frac_watai=0.5, sample_frac_merged=0.5):
    print(f"Run ID: {RUN_ID} | output: {OUTPUT_DIR.resolve()}")
    watai_raw = load_watai(sample_frac=sample_frac_watai)
    if not len(watai_raw):
        raise RuntimeError("watai dataset required")
    watai = add_family(clean(watai_raw))
    flood = watai[watai["family"].isin(FLOOD_CLASSES)].reset_index(drop=True)

    leakage_diagnostic(flood)
    leakage_test(flood)

    merged_raw = load_merged(sample_frac=sample_frac_merged)
    if len(merged_raw):
        merged_reference(merged_raw)
    else:
        print("S merged reference — dataset not available")

    summary_chart()
    print(f"\nDone. Results: {RESULTS_DIR.resolve()}")
    print(f"Plots: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main(sample_frac_watai=0.5, sample_frac_merged=0.5)