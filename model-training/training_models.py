import os, re, glob, json, time, hashlib, subprocess, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, roc_auc_score, classification_report)
import xgboost as xgb
import optuna
import psutil
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

GPU_AVAILABLE = False
GPU_INFO = "No GPU detected"

try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_INFO = torch.cuda.get_device_name(0)
except ImportError:
    pass

if not GPU_AVAILABLE:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            GPU_AVAILABLE = True
            GPU_INFO = out.stdout.strip().splitlines()[0]
    except Exception:
        pass


# XGBoost >= 2.0 GPU config. If you're on xgboost < 2.0, use tree_method="gpu_hist" 
XGB_TREE_METHOD = "hist"
XGB_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"


# Opt-in only see markdown above for why this defaults to False.
USE_GPU_FOR_LITE = False
CUML_AVAILABLE = False
if USE_GPU_FOR_LITE and GPU_AVAILABLE:
    try:
        import cudf
        from cuml.tree import DecisionTreeClassifier as cuDecisionTreeClassifier
        CUML_AVAILABLE = True
    except ImportError:
        print("[WARN] cuML/cuDF not installed — falling back to CPU Decision Tree for the Lite model.")

print(f"GPU available:        {GPU_AVAILABLE}")
print(f"GPU device:           {GPU_INFO}")
print(f"XGBoost, Heavy will train on: {XGB_DEVICE}")
print(f"Decision Tree, Lite will train on: {'GPU cuML' if CUML_AVAILABLE else 'CPU (scikit-learn)'}")

# Paths & output directory
CICIOT_ROOT = Path("data/CICIoT2023")
OUTPUT_DIR = Path("outputs/three_stage_reshuffle")
CICIOT_COMBINED_ROOT = Path("data/MERGED_CSV/MERGED_CSV")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

import os
from pathlib import Path

# root = CICIOT_ROOT
# print("Top-level folders found on disk:")
# for p in sorted(root.iterdir()):
#     if p.is_dir():
#         print(" ", p.name)

# folder_name : (base_filename_without_suffix, num_files)
CICIOT_MANIFEST = {
    "Backdoor_Malware":            ("Backdoor_Malware", 1),
    "Benign_Final":                ("BenignTraffic", 4),
    "BrowserHijacking":            ("BrowserHijacking", 1),
    "CommandInjection":            ("CommandInjection", 1),
    "DDoS-ACK_Fragmentation":      ("DDoS-ACK_Fragmentation", 13),
    "DDoS-HTTP_Flood":             ("DDoS-HTTP_Flood-", 1),
    "DDoS-ICMP_Flood":             ("DDoS-ICMP_Flood", 27),
    "DDoS-ICMP_Fragmentation":     ("DDoS-ICMP_Fragmentation", 20),
    "DDoS-PSHACK_FLOOD":           ("DDoS-PSHACK_Flood", 16),
    "DDoS-RSTFINFLOOD":            ("DDoS-RSTFINFlood", 15),
    "DDoS-SlowLoris":              ("DDoS-SlowLoris", 1),
    "DDoS-SYN_Flood":              ("DDoS-SYN_Flood", 16),
    "DDoS-SynonymousIP_Flood":     ("DDoS-SynonymousIP_Flood", 14),
    "DDoS-TCP_Flood":              ("DDoS-TCP_Flood", 18),
    "DDoS-UDP_Flood":              ("DDoS-UDP_Flood", 21),
    "DDoS-UDP_Fragmentation":      ("DDoS-UDP_Fragmentation", 13),
    "DictionaryBruteForce":        ("DictionaryBruteForce", 1),
    "DNS_Spoofing":                ("DNS_Spoofing", 1),
    "DoS-HTTP_Flood":              ("DoS-HTTP_Flood", 2),
    "DoS-SYN_Flood":               ("DoS-SYN_Flood", 8),
    "DoS-TCP_Flood":               ("DoS-TCP_Flood", 11),
    "DoS-UDP_Flood":               ("DoS-UDP_Flood", 17),
    "Mirai-greeth_flood":          ("Mirai-greeth_flood", 29),
    "Mirai-greip_flood":           ("Mirai-greip_flood", 22),
    "Mirai-udpplain":              ("Mirai-udpplain", 25),
    "MITM-ArpSpoofing":            ("MITM-ArpSpoofing", 2),
    "Recon-HostDiscovery":         ("Recon-HostDiscovery", 1),
    "Recon-OSScan":                ("Recon-OSScan", 1),
    "Recon-PingSweep":             ("Recon-PingSweep", 1),
    "Recon-PortScan":              ("Recon-PortScan", 1),
    "SqlInjection":                ("SqlInjection", 1),
    "Uploading_Attack":            ("Uploading_Attack", 1),
    "VulnerabilityScan":           ("VulnerabilityScan", 1),
    "XSS":                         ("XSS", 1),
}

# 8-class attack-family grouping  this is now used as the actual classification target for both Heavy and Lite tracks
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
    "Benign_Final": "Benign",
}

# 3-stage architecture: Stage 1 (attack/benign) to Stage 2 (6-way family,
# DDoS+DoS merged into "Flood") to Stage 3 (DDoS vs DoS, Flood rows only).
# Every stage trains a Heavy (XGBoost) and a Lite (Decision Tree) model .
FLOOD_MERGE = {"DDoS": "Flood", "DoS": "Flood"}
FLOOD_CLASSES = ("DDoS", "DoS")

def list_ciciot_files(class_folder, base_name, n_files, root=CICIOT_ROOT):
    # Reproduces the CICIoT2023 naming convention: base.pcap.csv, base1.pcap.csv, base2.pcap.csv ... base(n-1).pcap.csv
    folder = root / class_folder
    files = []
    for i in range(n_files):
        suffix = "" if i == 0 else str(i)
        fname = f"{base_name}{suffix}.pcap.csv"
        files.append(folder / fname)
    return files

def verify_manifest(manifest=CICIOT_MANIFEST, root=CICIOT_ROOT):
    # Run this first against your real CICIOT_ROOT to catch path/naming mismatches before kicking off a long load.
    missing = []
    for cls, (base, n) in manifest.items():
        for f in list_ciciot_files(cls, base, n, root):
            if not f.exists():
                missing.append(str(f))
    if missing:
        print(f"[WARN] {len(missing)} expected files not found. First 10:")
        for m in missing[:10]:
            print("   ", m)
    else:
        print("All manifest files found.")
    return missing

verify_manifest()  # CICIOT_ROOT points at real data

def load_ciciot_class(class_folder, label, base_name, n_files, root=CICIOT_ROOT,
                       usecols=None, sample_frac=None, chunksize=500_000):
    frames = []
    for fpath in list_ciciot_files(class_folder, base_name, n_files, root):
        if not fpath.exists():
            continue
        for chunk in pd.read_csv(fpath, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if sample_frac:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["Label"] = label
  #  df["AttackFamily"] = ATTACK_FAMILY_MAP.get(class_folder, "Unknown")
    return df

def load_ciciot2023(manifest=CICIOT_MANIFEST, root=CICIOT_ROOT, usecols=None, sample_frac=None):
    all_frames = []
    for cls, (base, n) in manifest.items():
        print(f"Loading class: {cls:30s} ({n} file(s))")
        df_cls = load_ciciot_class(cls, cls, base, n, root, usecols, sample_frac)
        all_frames.append(df_cls)
    full = pd.concat(all_frames, ignore_index=True)
    print(f"Total rows loaded: {len(full):,}")
    return full

def load_ciciot2023_floored(manifest=CICIOT_MANIFEST, root=CICIOT_ROOT, usecols=None,
                             sample_frac=0.02, min_per_class=1000):
    # Same loading logic as load_ciciot2023, but guarantees at least min_per_class rows survive sampling for classes small enough that sample_frac alone would gut them
    all_frames = []
    for cls, (base, n) in manifest.items():
        print(f"Loading class: {cls:30s} ({n} file(s))")
        df_cls = load_ciciot_class(cls, cls, base, n, root, usecols, sample_frac=None)
        if len(df_cls) == 0:
            print(f"[WARN] 0 rows loaded for class {cls} — check manifest/files")
            continue
        target_n = max(min_per_class, int(len(df_cls) * sample_frac))
        target_n = min(target_n, len(df_cls))
        df_cls = df_cls.sample(n=target_n, random_state=RANDOM_STATE)
        all_frames.append(df_cls)
    full = pd.concat(all_frames, ignore_index=True)
    print(f"Total rows loaded: {len(full):,}")
    return full


COLUMN_RENAME = {"Magnitue": "Magnitude"}  # fixes the CICIoT2023 raw-column typo

def normalize_columns(df):
    return df.rename(columns=COLUMN_RENAME)

DERIVED_CICIOT_COLUMNS = ["flow_duration", "Srate", "Drate", "Magnitude", "Radius", "Covariance", "Weight"]
 

def clean_dataframe(df, drop_cols=None):
    #  Data Cleaning module: remove ID/index columns, remove missing/empty records, remove duplicates, remove non-finite numeric values.
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df):,} rows with NaN/Inf")

    if "Label" in df.columns:
        df["Label"] = df["Label"].replace({"BenignTraffic": "Benign_Final"})

    return df.reset_index(drop=True)

def map_to_attack_family(df, label_col="Label", family_map=ATTACK_FAMILY_MAP):
     # The 34 raw CICIoT2023 classes into the 8-class attack-family grouping used as the classification target for both Heavy and Lite models.
     df = df.copy()
     mapped = df[label_col].map(family_map)
     unmapped = df.loc[mapped.isna(), label_col].unique().tolist()
     if unmapped:
         print(f"[WARN] {len(unmapped)} label(s) had no attack-family mapping and were left unchanged: {unmapped}")
     df[label_col] = mapped.fillna(df[label_col])
     print("8-class label distribution after family mapping:")
     print(df[label_col].value_counts())
     return df



def encode_protocol(df, col="Protocol Type"):
    if col not in df.columns:
        return df, None
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df, le

def pearson_filter(df, feature_cols, threshold=0.90):
    if len(feature_cols) < 2:
        return feature_cols, []
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in feature_cols if c not in to_drop]
    print(f"Pearson filter: dropped {len(to_drop)} of {len(feature_cols)} "
          f"features (|r| > {threshold})")
    return kept, to_drop


# Literature-derived, FIXED feature sets:
#  HEAVY_FEATURES from Almahaqeri et al. (2026), LightGBM gain-based selection, 23 features
#  LITE_FEATURES  from Dzaki et al. (2025), Gini Impurity Tree-based selection, 10 features

LITE_FEATURES = [
    "Tot size", "Protocol Type", 
    "fin_flag_number", "syn_flag_number",
     "Header_Length","UDP",
      "Min", "Max", "AVG",
      "Number","Std","TCP",]

HEAVY_FEATURES = [
    "Header_Length", "Protocol Type", 
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", 
    "cwr_flag_number", "ack_count",
     "HTTP", "HTTPS", "IAT",
    "SSH", "IRC", "TCP", "UDP",  "ICMP",
       "Tot sum", "Min", "Max", "AVG",
      "Number", "Variance",]





# Synthetic w=100 samples for the w=10-native families (Benign, Web, Recon,
# Spoofing, BruteForce). DDoS/DoS/
# Mirai are captured at Number=100 in the real dataset; every other class is
# captured at Number=10. Stage 2's w=100 buffer (see model_engine.py) only
# ever saw genuine Flood/Mirai rows during training, so it has no way to
# recognize "not Flood/Mirai" at w=100 and confidently guesses wrong instead.

W10_NATIVE_CLASSES = {
    cls: spec for cls, spec in CICIOT_MANIFEST.items()
    if ATTACK_FAMILY_MAP.get(cls) not in ("DDoS", "DoS", "Mirai")
}

_W100_MEAN_COLS = [
    "Header_Length", "Time_To_Live", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
]
_W100_SUM_COLS = ["ack_count", "syn_count", "fin_count", "rst_count", "Tot sum"]


def list_ciciot_merged_files(root=CICIOT_COMBINED_ROOT, start=1, end=63):
    files = []
    for i in range(start, end + 1):
        fpath = root / f"Merged{i:02d}.csv"
        if fpath.exists():
            files.append(fpath)
        else:
            print(f"[WARN] File not found: {fpath}")
    if not files:
        print(f"[WARN] No Merged files found under {root}")
    return files

def load_ciciot2023_merged(root=CICIOT_COMBINED_ROOT, start=1, end=63, usecols=None,
                            sample_frac=None, chunksize=500_000):
    files = list_ciciot_merged_files(root=root, start=start, end=end)
    if not files:
        return pd.DataFrame()
    frames = []
    for i, fpath in enumerate(files):
        print(f"Loading {fpath.name} ({i + 1}/{len(files)})...")
        for chunk in pd.read_csv(fpath, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    if "Label" in full.columns and "Label" not in full.columns:
        full = full.rename(columns={"Label": "Label"})
    elif "LABEL" in full.columns and "Label" not in full.columns:
        full = full.rename(columns={"LABEL": "Label"})
    print(f"\nTotal rows loaded: {len(full):,}")
    if "Label" in full.columns:
        print("\nClass distribution:")
        print(full["Label"].value_counts())
    return full


def _reconstruct_w100_row(group):
    # group: a 10-row DataFrame slice, each row a genuine w=10 sample from
    # the SAME class, consecutive in original file order.
    row = {}

    for col in _W100_MEAN_COLS:
        if col in group.columns:
            row[col] = float(group[col].mean())

    for col in _W100_SUM_COLS:
        if col in group.columns:
            row[col] = float(group[col].sum())

    if "Protocol Type" in group.columns:
        row["Protocol Type"] = group["Protocol Type"].mode().iloc[0]

    if "Min" in group.columns:
        row["Min"] = float(group["Min"].min())
    if "Max" in group.columns:
        row["Max"] = float(group["Max"].max())

    row["Number"] = 100.0

    tot_sum = row.get("Tot sum")
    if tot_sum is not None:
        row["AVG"] = tot_sum / 100.0
        row["Tot size"] = row["AVG"]  # matches the real data: Tot size == AVG exactly

    if "Rate" in group.columns:
        rates = group["Rate"].replace(0, np.nan)
        durations = 10.0 / rates  # Number_i=10 for every w10 row
        total_duration = durations.sum()
        row["Rate"] = (100.0 / total_duration) if total_duration and not np.isnan(total_duration) else 0.0
        if "IAT" not in row and "IAT" in group.columns:
            iat_total_duration = (9.0 * group["IAT"]).sum()  # ~9 gaps per 10-packet sub-window
            row["IAT"] = iat_total_duration / 99.0            # ~99 gaps over the combined 100

    if "Std" in group.columns or "Variance" in group.columns:
        sub_means = group["AVG"] if "AVG" in group.columns else None
        sub_var_sample = group["Variance"] if "Variance" in group.columns else (group["Std"] ** 2)
        n = 10
        sub_var_pop = sub_var_sample * (n - 1) / n           # ddof=1 -> ddof=0
        overall_mean = sub_means.mean() if sub_means is not None else 0.0
        between_group = ((sub_means - overall_mean) ** 2).mean() if sub_means is not None else 0.0
        pooled_pop_var = sub_var_pop.mean() + between_group
        N = 100
        pooled_sample_var = pooled_pop_var * N / (N - 1)      # ddof=0 -> ddof=1
        row["Variance"] = float(pooled_sample_var)
        row["Std"] = float(np.sqrt(max(pooled_sample_var, 0.0)))

    return row


def _safe_transform_protocol(series, proto_encoder):
    # proto_encoder is fit on the (often small) SAMPLED main dataset
    known = set(proto_encoder.classes_)
    as_str = series.astype(str)
    unseen = ~as_str.isin(known)
    if unseen.any():
        print(f"  [WARN] {unseen.sum()} Protocol Type value(s) unseen by proto_encoder "
              f"during synthesis — mapped to {proto_encoder.classes_[0]!r}")
        as_str = as_str.where(~unseen, proto_encoder.classes_[0])
    return proto_encoder.transform(as_str)


def synthesize_w100_for_class(class_folder, base_name, n_files, root, label,
                                proto_encoder, group_size=10, max_groups=2000):
    df = load_ciciot_class(class_folder, label, base_name, n_files, root)
    if len(df) == 0:
        return pd.DataFrame()
    df = normalize_columns(df)
    df = clean_dataframe(df)  # drop dupes/NaN/Inf BEFORE grouping — keeps groups clean
    if "Protocol Type" in df.columns and proto_encoder is not None:
        df["Protocol Type"] = _safe_transform_protocol(df["Protocol Type"], proto_encoder)

    n_groups = min(len(df) // group_size, max_groups)
    if n_groups == 0:
        return pd.DataFrame()

    rows = []
    for g in range(n_groups):
        group = df.iloc[g * group_size:(g + 1) * group_size]
        row = _reconstruct_w100_row(group)
        row["Label"] = label
        rows.append(row)

    synth = pd.DataFrame(rows)
    print(f"  synthesized {len(synth):,} w=100 rows for {label} "
          f"(from {n_groups * group_size:,}/{len(df):,} w=10 rows)")
    return synth


def generate_synthetic_w100_dataset(proto_encoder, root=CICIOT_ROOT, max_groups_per_class=2000):
    print("Synthesizing w=100 negative samples for w=10-native classes...")
    frames = []
    for cls, (base, n) in W10_NATIVE_CLASSES.items():
        synth = synthesize_w100_for_class(cls, base, n, root, cls, proto_encoder,
                                           max_groups=max_groups_per_class)
        if len(synth):
            frames.append(synth)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total synthetic w=100 rows: {len(combined):,}")
    return combined


def resolve_feature_set(df, requested_features):
    available = [c for c in requested_features if c in df.columns]
    missing = [c for c in requested_features if c not in df.columns]
    print(f" {len(requested_features)} | available: {len(available)} | missing: {len(missing)}")
    if missing:
        print(f"NOT PRESENT in this data pull (dropped): {missing}")
    return available




# Splitting, train-only resampling & scaling 

def split_ciciot(df, label_col="Label", test_size=0.20, val_frac_of_train=0.125, random_state=RANDOM_STATE):
    #  Stratified 80/20 train/test split  A validation slice is then carved out of the 80%
    train_full, test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=random_state)
    train, val = train_test_split(train_full, test_size=val_frac_of_train,
                                   stratify=train_full[label_col], random_state=random_state)
    print(f"CICIoT2023 stratified 80/20 split -> train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def undersample_train(df, label_col="y", min_per_class=1000, target_total=None,
                                  random_state=RANDOM_STATE):
    # Resampling module, Almahaqeri-style stratified undersampling, applied to the TRAIN partition only
    counts = df[label_col].value_counts()
    if target_total is not None:
        per_class_cap = max(min_per_class, target_total // df[label_col].nunique())
    else:
        per_class_cap = max(min_per_class, int(counts.median()))

    parts = []
    for cls, n in counts.items():
        cls_df = df[df[label_col] == cls]
        if n > per_class_cap:
            cls_df = cls_df.sample(n=per_class_cap, random_state=random_state)
        parts.append(cls_df)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"Stratified undersampling (TRAIN only): {len(df):,} -> {len(out):,} rows "
          f"(per-class cap = {per_class_cap:,})")
    print(out[label_col].value_counts())
    return out

def gini_based_selection(X_train, y_train, top_k=6):
    model = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False).head(top_k).index.tolist()


def select_features_for_track(train_df, candidate_features, lite_top_k=6):
    pearson_kept, _ = pearson_filter(train_df, candidate_features)
    if len(pearson_kept) < 2:
        pearson_kept = candidate_features
    heavy_features = pearson_kept
    if len(pearson_kept) > lite_top_k:
        lite_features = gini_based_selection(
            train_df[pearson_kept], train_df["y"], top_k=lite_top_k)
    else:
        lite_features = pearson_kept
    return heavy_features, lite_features

def fit_scaler(train_df, feature_cols):
    # StandardScaler fitted on TRAIN only (post-resampling); TEST/VAL reuse these statistics.
    if isinstance(feature_cols, tuple):
        feature_cols = feature_cols[0]
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler

def apply_scaler(df, feature_cols, scaler):
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df

#  Heavy model (XGBoost, GPU, sample-weighted, narrow Optuna search)

def compute_sample_weights(y):
    counts = np.bincount(y)
    N, K = len(y), len(counts)
    class_weights = N / (K * counts)
    return class_weights[y]


def _xgb_objective_params(num_class):
    # Stage 3 is always binary (DDoS vs DoS) this exists so the objective
    # is still correct if this script's target ever stops being exactly 2
    # classes for some reason, rather than silently assuming binary.
    if num_class == 2:
        return {"objective": "binary:logistic", "eval_metric": "logloss"}
    return {"objective": "multi:softprob", "num_class": num_class, "eval_metric": "mlogloss"}


def build_xgb_objective(X_train, y_train, X_val, y_val, num_class):
    def objective(trial):
        params = {
            **_xgb_objective_params(num_class),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "tree_method": XGB_TREE_METHOD,
            "device": XGB_DEVICE,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
        return f1
    return objective

def tune_heavy_model(X_train, y_train, X_val, y_val, num_class, n_trials=15):
    # n_trials kept small 
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(build_xgb_objective(X_train, y_train, X_val, y_val, num_class), n_trials=n_trials)
    print("Best macro-F1 (val):", study.best_value)
    #print("Best params:", study.best_params)
    return study.best_params

def train_heavy_model(X_train, y_train, X_val, y_val, num_class, best_params):
    eval_metric = ["logloss", "error"] if num_class == 2 else ["mlogloss", "merror"]
    params = {**best_params, **_xgb_objective_params(num_class),
              "tree_method": XGB_TREE_METHOD, "device": XGB_DEVICE,
              "random_state": RANDOM_STATE, "n_jobs": -1, "eval_metric": eval_metric}
    sample_weights = compute_sample_weights(y_train.values)
    t0 = time.perf_counter()
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
   # print(f"Heavy[{tag}] trained on device='{XGB_DEVICE}' in {time.perf_counter()-t0:.1f}s")
    return model, model.evals_result()

# Lite model kept as a low-complexity, minimally-tuned edge baseline 
LITE_PARAMS = dict(
    criterion="gini",
    splitter="best",
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

def train_lite_model(X_train, y_train, params=LITE_PARAMS):
    if CUML_AVAILABLE:
        # cuML's DecisionTreeClassifier has only max_depth/random_state are honored here, this path is an experimental accuracy check, not the deployment model.
        cu_model = cuDecisionTreeClassifier(max_depth=params["max_depth"], random_state=params["random_state"])
        X_gpu = cudf.DataFrame.from_pandas(X_train.reset_index(drop=True))
        y_gpu = cudf.Series(y_train.reset_index(drop=True))
        t0 = time.perf_counter()
        cu_model.fit(X_gpu, y_gpu)
        print(f"Lite model trained on GPU (cuML) in {time.perf_counter()-t0:.1f}s [experimental]")
        cu_model._is_gpu = True
        return cu_model
    model = DecisionTreeClassifier(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"Lite model trained on CPU in {time.perf_counter()-t0:.1f}s")
    model._is_gpu = False
    return model

def evaluate_model(model, X_test, y_test, label_names=None, model_name="model"):
    # TEST is only ever scored here, once per model, after training/tuning is fully frozen.
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0
    per_sample_latency_us = (elapsed / len(X_test)) * 1e6

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"[WARN] ROC-AUC not computed: {e}")

    results = {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro,
        "weighted_precision": p_w, "weighted_recall": r_w, "weighted_f1": f1_w,
        "roc_auc_macro_ovr": roc_auc,
        "inference_latency_us_per_sample": per_sample_latency_us,
        "confusion_matrix": cm.tolist(),
    }
    print(json.dumps({k: v for k, v in results.items() if k != "confusion_matrix"}, indent=2))
    if label_names is not None:
        print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    return results

def model_size_bytes(model, path):
    import joblib
    joblib.dump(model, path)
    return os.path.getsize(path)


# XGBoost does have boosting rounds, and can track train-vs-validation loss across them this is to show did the model behave during training, plot can get for a gradient-boosted model.
def plot_xgb_training_curve(evals_result, model_name="HeavyNet (XGBoost) — Stage 3"):
        loss_key = "logloss" if "logloss" in evals_result["validation_0"] else "mlogloss"
        err_key = "error" if "error" in evals_result["validation_0"] else "merror"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        rounds = range(len(evals_result["validation_0"][loss_key]))

        axes[0].plot(rounds, evals_result["validation_0"][loss_key], label="training")
        axes[0].plot(rounds, evals_result["validation_1"][loss_key], label="validation")
        axes[0].set_xlabel("boosting round"); axes[0].set_ylabel(loss_key)
        axes[0].set_title(f"{model_name} — log loss per round")
        axes[0].legend()

        axes[1].plot(rounds, evals_result["validation_0"][err_key], label="training")
        axes[1].plot(rounds, evals_result["validation_1"][err_key], label="validation")
        axes[1].set_xlabel("boosting round"); axes[1].set_ylabel(err_key)
        axes[1].set_title(f"{model_name} — error rate per round")
        axes[1].legend()
        plt.tight_layout()
        plt.show()


from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(model, X_test, y_test, class_names, model_name="model"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True)
    ax.set_title(f"{model_name} — confusion matrix (held-out test set)")
    plt.tight_layout()
    plt.show()
# HEAVY_FEATURES/LITE_FEATURES lists, if a feature you kept shows near-zero importance on your actual data pull, that's worth noting in the writeup
def plot_feature_importance(model, feature_names, model_name="model", top_n=None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Model has no feature_importances_ attribute")
    order = np.argsort(importances)[::-1]
    if top_n:
        order = order[:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.3)))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    ax.set_xlabel("importance")
    ax.set_title(f"{model_name} — feature importance")
    plt.tight_layout()
    plt.show()
from sklearn.model_selection import learning_curve
# The substitute for the models behavour during training is a learning curve, train the same model architecture on increasing fractions of the training data and watch how train/validation accuracy converge
def plot_learning_curve(estimator, X, y, model_name="model", cv=3):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 6), random_state=RANDOM_STATE, n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="training")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="cross-validation")
    ax.set_xlabel("training examples")
    ax.set_ylabel("macro F1")
    ax.set_title(f"{model_name} — learning curve")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Since REAF-5G's whole premise is a Heavy/Lite accuracy-vs-efficiency tradeoff, a side-by-side bar chart of the metrics evaluate_model already returns
def plot_model_comparison(heavy_results, lite_results):
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    heavy_vals = [heavy_results[m] for m in metrics]
    lite_vals = [lite_results[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width/2, heavy_vals, width, label=f"Heavy (XGBoost) ")
    ax1.bar(x + width/2, lite_vals, width, label=f"Lite (Decision Tree) ")
    ax1.set_xticks(x); ax1.set_xticklabels(metrics)
    ax1.set_ylabel("score")
    ax1.set_title(f"Heavy vs. Lite — accuracy metrics ")
    ax1.legend()
    plt.tight_layout()
    plt.show()

    fig, ax2 = plt.subplots(figsize=(6, 5))
    ax2.bar(["Heavy", "Lite"],
            [heavy_results["inference_latency_us_per_sample"], lite_results["inference_latency_us_per_sample"]],
            color=["steelblue", "orange"])
    ax2.set_ylabel("inference latency (µs/sample)")
    ax2.set_title(f"Heavy  vs. Lite  — inference latency")
    plt.tight_layout()
    plt.show()


def export_heavy_to_onnx(xgb_model, n_features, out_path):
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    booster = xgb_model.get_booster()
    original_feature_names = booster.feature_names   # remember the real names
    booster.feature_names = None   # forces XGBoost's dump to use the f0, f1, ... convention
                                    # onnxmltools' converter actually expects

    try:
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)
        with open(out_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Heavy model exported to {out_path}")
    finally:
        booster.feature_names = original_feature_names  # restore, so heavy_model still
                                                          # works normally for predict()/
                                                          # plot_feature_importance() afterward
def export_lite_to_onnx(dt_model, n_features, out_path):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(dt_model, initial_types=initial_type)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Lite model exported to {out_path}")

def get_unique_path(path):
    # Returns `path` unchanged if it doesn't exist yet. If it does, appends
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

def train_and_evaluate_stage(train_h, val_h, test_h, heavy_features,
                              train_l, test_l, lite_features,
                              y_col, num_class, class_names, optuna_trials, tag):
    # Trains and evaluates the Heavy(XGBoost)/Lite(DecisionTree) pair for
    # ONE stage. Returns everything the caller needs to export and report.
    best_params = tune_heavy_model(train_h[heavy_features], train_h[y_col],
                                    val_h[heavy_features], val_h[y_col],
                                    num_class, n_trials=optuna_trials)
    heavy_model, heavy_evals = train_heavy_model(train_h[heavy_features], train_h[y_col],
                                                  val_h[heavy_features], val_h[y_col],
                                                  num_class, best_params)
    lite_model = train_lite_model(train_l[lite_features], train_l[y_col])

    print(f"\nHeavy model — {tag}, CICIoT2023 test:")
    heavy_results = evaluate_model(heavy_model, test_h[heavy_features], test_h[y_col],
                                    label_names=class_names, model_name=f"heavy_{tag}_xgboost")
    print(f"\nLite model — {tag}, CICIoT2023 test:")
    lite_results = evaluate_model(lite_model, test_l[lite_features], test_l[y_col],
                                   label_names=class_names, model_name=f"lite_{tag}_decision_tree")

    plot_xgb_training_curve(heavy_evals, model_name=f"HeavyNet (XGBoost) — {tag}")
    plot_confusion_matrix(heavy_model, test_h[heavy_features], test_h[y_col], class_names,
                           f"HeavyNet (XGBoost) — {tag}")
    plot_confusion_matrix(lite_model, test_l[lite_features], test_l[y_col], class_names,
                           f"LiteNet (Decision Tree) — {tag}")
    plot_feature_importance(heavy_model, heavy_features, f"HeavyNet (XGBoost) — {tag}")
    plot_feature_importance(lite_model, lite_features, f"LiteNet (Decision Tree) — {tag}")
    plot_model_comparison(heavy_results, lite_results)

    return heavy_model, lite_model, heavy_results, lite_results


def print_stage3_subtype_breakdown(model, flood_test, features, stage3_encoder, tag):
     if "RawLabel" not in flood_test.columns:
         print(f"[{tag}] RawLabel not available — skipping per-subtype breakdown")
         return
     preds = model.predict(flood_test[features])
     pred_labels = stage3_encoder.inverse_transform(preds)
     true_labels = flood_test["Label"].values
     subtypes = flood_test["RawLabel"].values

     print(f"\n--- Stage 3 ({tag}) accuracy by original subtype ---")
     for subtype in sorted(set(subtypes)):
         mask = subtypes == subtype
         n = mask.sum()
         if n == 0:
             continue
         acc = (pred_labels[mask] == true_labels[mask]).mean()
         print(f"  {subtype:28s} n={n:6d}  accuracy={acc:.3f}")


def run_pipeline(sample_frac_ciciot=0.5, optuna_trials=7, min_per_class=1000,
                  undersample_min_per_class=1000, undersample_target_total=None):
    # Full 3-stage architecture. Every stage trains a Heavy (XGBoost) and a
    # Lite (Decision Tree) model only models total, 6 ONNX exports total.

    # Flow Feature Extraction Engine data loading itself is unchanged.
    ciciot_raw = load_ciciot2023_floored(sample_frac=sample_frac_ciciot, min_per_class=min_per_class)
   
    #ciciot_raw = load_ciciot2023_merged(sample_frac=sample_frac_ciciot)
    ciciot_raw = normalize_columns(ciciot_raw)

    # Data Cleaning & Normalization Module + Common Feature Processing Layer
    ciciot = clean_dataframe(ciciot_raw)
    ciciot, proto_encoder = encode_protocol(ciciot)

  
    ciciot["RawLabel"] = ciciot["Label"]
    ciciot = map_to_attack_family(ciciot)

    # Stratified 80/20 split (validation carved out of TRAIN only; TEST held out untouched)
    train, val, test = split_ciciot(ciciot)
    # Full 8-class target used only for stratified undersampling, so every
    # family (including the ones stage 1/2/3 don't directly train on) stays
    # balanced going into the per-stage target derivations below.
    full_label_encoder = LabelEncoder()
    train = train.copy(); val = val.copy(); test = test.copy()
    train["y"] = full_label_encoder.fit_transform(train["Label"])
    val["y"] = full_label_encoder.transform(val["Label"])
    test["y"] = full_label_encoder.transform(test["Label"])

    train = undersample_train(
        train, label_col="y",
        min_per_class=undersample_min_per_class,
        target_total=undersample_target_total,
    )

    # Augment TRAIN ONLY with synthetic w=100 rows for the w=10-native
    # families val/test stay pure, genuine per-CSV rows for honest
    # evaluation. 
    synthetic_w100 = generate_synthetic_w100_dataset(proto_encoder, max_groups_per_class=2000)
    if len(synthetic_w100):
        synthetic_w100 = map_to_attack_family(synthetic_w100)
        synthetic_w100["y"] = full_label_encoder.transform(synthetic_w100["Label"])
        train = pd.concat([train, synthetic_w100], ignore_index=True)
        print(f"Train set after adding synthetic w=100 rows: {len(train):,}")

    heavy_features = resolve_feature_set(train, HEAVY_FEATURES)
    lite_features = resolve_feature_set(train, LITE_FEATURES)

    heavy_scaler = fit_scaler(train, heavy_features)
    lite_scaler = fit_scaler(train, lite_features)

    train_h = apply_scaler(train, heavy_features, heavy_scaler)
    val_h = apply_scaler(val, heavy_features, heavy_scaler)
    test_h = apply_scaler(test, heavy_features, heavy_scaler)
    train_l = apply_scaler(train, lite_features, lite_scaler)
    val_l = apply_scaler(val, lite_features, lite_scaler)
    test_l = apply_scaler(test, lite_features, lite_scaler)

    #  STAGE 1: attack vs benign 
    for df in (train_h, val_h, test_h, train_l, val_l, test_l):
        df["y_stage1"] = (df["Label"] != "Benign").astype(int)

    stage1_heavy, stage1_lite, stage1_heavy_res, stage1_lite_res = train_and_evaluate_stage(
        train_h, val_h, test_h, heavy_features, train_l, test_l, lite_features,
        y_col="y_stage1", num_class=2, class_names=["Benign", "Attack"],
        optuna_trials=optuna_trials, tag="stage1",
    )

    #  STAGE 2: 6-way family, ATTACK rows only 
    attack_train_h = train_h[train_h["Label"] != "Benign"].reset_index(drop=True)
    attack_val_h = val_h[val_h["Label"] != "Benign"].reset_index(drop=True)
    attack_test_h = test_h[test_h["Label"] != "Benign"].reset_index(drop=True)
    attack_train_l = train_l[train_l["Label"] != "Benign"].reset_index(drop=True)
    attack_test_l = test_l[test_l["Label"] != "Benign"].reset_index(drop=True)

    stage2_encoder = LabelEncoder().fit(attack_train_h["Label"].replace(FLOOD_MERGE))
    stage2_classes = list(stage2_encoder.classes_)
    for df in (attack_train_h, attack_val_h, attack_test_h, attack_train_l, attack_test_l):
        df["y_stage2"] = stage2_encoder.transform(df["Label"].replace(FLOOD_MERGE))

    stage2_heavy, stage2_lite, stage2_heavy_res, stage2_lite_res = train_and_evaluate_stage(
        attack_train_h, attack_val_h, attack_test_h, heavy_features,
        attack_train_l, attack_test_l, lite_features,
        y_col="y_stage2", num_class=len(stage2_classes), class_names=stage2_classes,
        optuna_trials=optuna_trials, tag="stage2",
    )

    #  STAGE 3: DDoS vs DoS, FLOOD rows only 
    flood_train_h = attack_train_h[attack_train_h["Label"].isin(FLOOD_CLASSES)].reset_index(drop=True)
    flood_val_h = attack_val_h[attack_val_h["Label"].isin(FLOOD_CLASSES)].reset_index(drop=True)
    flood_test_h = attack_test_h[attack_test_h["Label"].isin(FLOOD_CLASSES)].reset_index(drop=True)
    flood_train_l = attack_train_l[attack_train_l["Label"].isin(FLOOD_CLASSES)].reset_index(drop=True)
    flood_test_l = attack_test_l[attack_test_l["Label"].isin(FLOOD_CLASSES)].reset_index(drop=True)

    stage3_encoder = LabelEncoder().fit(flood_train_h["Label"])
    stage3_classes = list(stage3_encoder.classes_)
    for df in (flood_train_h, flood_val_h, flood_test_h, flood_train_l, flood_test_l):
        df["y_stage3"] = stage3_encoder.transform(df["Label"])

    stage3_heavy, stage3_lite, stage3_heavy_res, stage3_lite_res = train_and_evaluate_stage(
        flood_train_h, flood_val_h, flood_test_h, heavy_features,
        flood_train_l, flood_test_l, lite_features,
        y_col="y_stage3", num_class=len(stage3_classes), class_names=stage3_classes,
        optuna_trials=optuna_trials, tag="stage3",
    )

    # Per-subtype breakdown the aggregate DDoS-vs-DoS number above blends
    print_stage3_subtype_breakdown(stage3_heavy, flood_test_h, heavy_features, stage3_encoder, "Heavy")
    print_stage3_subtype_breakdown(stage3_lite, flood_test_l, lite_features, stage3_encoder, "Lite")

    #  Export, ONLY the 6 stage models, XGBoost/DecisionTree only 
    onnx_paths = {}
    stage_models = {
        "stage1": (stage1_heavy, stage1_lite),
        "stage2": (stage2_heavy, stage2_lite),
        "stage3": (stage3_heavy, stage3_lite),
    }
    for stage_tag, (heavy_model, lite_model) in stage_models.items():
        heavy_onnx_path = get_unique_path(OUTPUT_DIR / f"heavy_{stage_tag}_xgboost.onnx")
        lite_onnx_path = get_unique_path(OUTPUT_DIR / f"lite_{stage_tag}_decision_tree.onnx")
        try:
            export_heavy_to_onnx(heavy_model, len(heavy_features), str(heavy_onnx_path))
            onnx_paths[f"heavy_{stage_tag}"] = heavy_onnx_path.name
        except Exception as e:
            print(f"[WARN] {stage_tag} Heavy ONNX export failed: {e}")
            onnx_paths[f"heavy_{stage_tag}"] = None
        try:
            export_lite_to_onnx(lite_model, len(lite_features), str(lite_onnx_path))
            onnx_paths[f"lite_{stage_tag}"] = lite_onnx_path.name
        except Exception as e:
            print(f"[WARN] {stage_tag} Lite ONNX export failed: {e}")
            onnx_paths[f"lite_{stage_tag}"] = None

    joblib.dump(heavy_scaler, get_unique_path(OUTPUT_DIR / "heavy_scaler.pkl"))
    joblib.dump(lite_scaler, get_unique_path(OUTPUT_DIR / "lite_scaler.pkl"))
    joblib.dump(full_label_encoder, get_unique_path(OUTPUT_DIR / "full_label_encoder.pkl"))
    joblib.dump(stage2_encoder, get_unique_path(OUTPUT_DIR / "stage2_encoder.pkl"))
    joblib.dump(stage3_encoder, get_unique_path(OUTPUT_DIR / "stage3_encoder.pkl"))
    joblib.dump(proto_encoder, OUTPUT_DIR / "protocol_encoder.joblib")

    with open(get_unique_path(OUTPUT_DIR / "feature_lists.json"), "w") as f:
        json.dump({
            "heavy_features": heavy_features,
            "lite_features": lite_features,
            "stage2_classes": stage2_classes,
            "stage3_classes": stage3_classes,
            "onnx_files": onnx_paths,
        }, f, indent=2)

    print(f"\nAll 6 stage models and artifacts written to: {OUTPUT_DIR.resolve()}")
    return {
        "stage1": {"heavy": stage1_heavy_res, "lite": stage1_lite_res},
        "stage2": {"heavy": stage2_heavy_res, "lite": stage2_lite_res},
        "stage3": {"heavy": stage3_heavy_res, "lite": stage3_lite_res},
    }


if __name__ == "__main__":
    run_pipeline(sample_frac_ciciot=0.5, optuna_trials=7, min_per_class=1000)