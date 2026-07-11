# pip install pandas numpy scikit-learn xgboost lightgbm optuna psutil onnxmltools skl2onnx joblib torch
# Optional (GPU-accelerated Lite model, experimentation only): pip install cudf-cu12 cuml-cu12 --extra-index-url https://pypi.nvidia.com
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

# XGBoost >= 2.0 GPU config. If you're on xgboost < 2.0, use tree_method="gpu_hist" instead and drop the "device" key.
XGB_TREE_METHOD = "hist"
XGB_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"

# Opt-in only — see markdown above for why this defaults to False.
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
print(f"XGBoost (Heavy) will train on: {XGB_DEVICE}")
print(f"Decision Tree (Lite) will train on: {'GPU (cuML)' if CUML_AVAILABLE else 'CPU (scikit-learn)'}")

# Paths & output directory
CICIOT_ROOT = Path("data/CICIoT2023")   
IDS2018_ROOT = Path("data/CSE-CIC-IDS2018")     
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# folder_name : (base_filename_without_suffix, num_files)
CICIOT_MANIFEST = {
    "Backdoor_Malware":            ("Backdoor_Malware", 1),
    "Benign_Final":                ("BenignTraffic", 5),
    "BrowserHijacking":            ("BrowserHijacking", 1),
    "CommandInjection":            ("CommandInjection", 13),
    "DDOS-ACK_Fragmentation":      ("DDOS-ACK_Fragmentation", 1),
    "DDOS-HTTP_Flood":             ("DDOS-HTTP_Flood", 1),
    "DDOS-ICMP_Flood":             ("DDOS-ICMP_Flood", 27),
    "DDOS-ICMP_Fragmentation":     ("DDOS-ICMP_Fragmentation", 20),
    "DDOS-PSHACK_FLOOD":           ("DDOS-PSHACK_FLOOD", 16),
    "DDOS-RSTFINFLOOD":            ("DDOS-RSTFINFLOOD", 15),
    "DDOS-SlowLoris":              ("DDOS-SlowLoris", 1),
    "DDOS-SYN_Flood":              ("DDOS-SYN_Flood", 16),
    "DDOS-SynonymousIP_Flood":     ("DDOS-SynonymousIP_Flood", 14),
    "DDOS-TCP_Flood":              ("DDOS-TCP_Flood", 18),
    "DDOS-UDP_Flood":              ("DDOS-UDP_Flood", 21),
    "DDOS-UDP_Fragmentation":      ("DDOS-UDP_Fragmentation", 13),
    "DictionaryBruteForce":        ("DictionaryBruteForce", 1),
    "DNS Spoofing":                ("DNS Spoofing", 1),
    "DoS-HTTP_Flood":              ("DoS-HTTP_Flood", 2),
    "DOS-SYN_Flood":               ("DOS-SYN_Flood", 8),
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
    "Uploading Attack":            ("Uploading Attack", 1),
    "VulnerabilityScan":           ("VulnerabilityScan", 1),
    "XSS":                         ("XSS", 1),
}

# 8-class attack-family grouping used later for evidence prioritisation (Section 15)
# and the coarse-grained classification task, following CICIoT2023 documentation.
ATTACK_FAMILY_MAP = {
    "Backdoor_Malware": "Web", "BrowserHijacking": "Web", "CommandInjection": "Web",
    "SqlInjection": "Web", "Uploading Attack": "Web", "XSS": "Web",
    "DDOS-ACK_Fragmentation": "DDoS", "DDOS-HTTP_Flood": "DDoS", "DDOS-ICMP_Flood": "DDoS",
    "DDOS-ICMP_Fragmentation": "DDoS", "DDOS-PSHACK_FLOOD": "DDoS", "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SlowLoris": "DDoS", "DDOS-SYN_Flood": "DDoS", "DDOS-SynonymousIP_Flood": "DDoS",
    "DDOS-TCP_Flood": "DDoS", "DDOS-UDP_Flood": "DDoS", "DDOS-UDP_Fragmentation": "DDoS",
    "DictionaryBruteForce": "BruteForce",
    "DNS Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    "DoS-HTTP_Flood": "DoS", "DOS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-UDP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    "Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon", "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon", "VulnerabilityScan": "Recon",
    "Benign_Final": "Benign",
}

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
    df["AttackFamily"] = ATTACK_FAMILY_MAP.get(class_folder, "Unknown")
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

IDS2018_FILES = [
    "Bot.csv", "Brute Force -Web.csv", "Brute Force -XSS.csv",
    "DDOS attack-HOIC.csv", "DDOS attack-LOIC-UDP.csv", "DDoS attacks-LOIC-HTTP.csv",
    "DoS attacks-GoldenEye.csv", "DoS attacks-Hulk.csv", "DoS attacks-SlowHTTPTest.csv",
    "DoS attacks-Slowloris.csv", "FTP-BruteForce.csv", "Infilteration.csv",
    "SQL Injection.csv", "SSH-Bruteforce.csv",
]

def load_ids2018(root=IDS2018_ROOT, files=IDS2018_FILES, sample_frac=None, chunksize=300_000):
    frames = []
    for fname in files:
        fpath = root / fname
        if not fpath.exists():
            print(f"[WARN] missing {fpath}")
            continue
        print(f"Loading {fname}")
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if sample_frac:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    print(f"IDS2018 total rows: {len(df):,}")
    return df

def clean_dataframe(df, drop_cols=None):
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

    return df.reset_index(drop=True)

def encode_protocol(df, col="Protocol Type"):
    if col not in df.columns:
        return df, None
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df, le

COLUMN_RENAME = {"Magnitue": "Magnitude"}  # fixes the CICIoT2023 raw-column typo

def normalize_columns(df):
    return df.rename(columns=COLUMN_RENAME)

def pearson_filter(df, feature_cols, threshold=0.90):
    # Drops one feature from every pair with |correlation| > threshold, keeping the first-seen feature.
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in feature_cols if c not in to_drop]
    print(f"Pearson filter: dropped {len(to_drop)} of {len(feature_cols)} features (|r| > {threshold})")
    return kept, to_drop

HEAVY_FEATURES = [
     "flow_duration", "Rate", "Srate", "Drate", "Tot sum", "Number", "Tot size",
     "IAT", "Header_Length", "Min", "Max", "AVG", "Std",
     "Magnitude", "Radius", "Covariance", "Variance", "Weight",
     "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
     "rst_count", "Protocol Type",
 ] 

LITE_FEATURES = [
     "IAT", "Magnitude", "Protocol Type", "Header_Length", "Min",
     "flow_duration", "fin_count", "rst_count", "Srate", "urg_count",
 ]

def resolve_feature_set(df, requested_features, set_name="feature set"):
    available = [c for c in requested_features if c in df.columns]
    missing = [c for c in requested_features if c not in df.columns]
    print(f"[{set_name}] requested: {len(requested_features)} | available: {len(available)} | missing: {len(missing)}")
    if missing:
        print(f"[{set_name}] NOT PRESENT in this data pull (dropped): {missing}")
    return available, missing

import lightgbm as lgb

def gain_based_selection(X_train, y_train, median_rule=True, top_k=None):
    model = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    gains = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                       index=X_train.columns).sort_values(ascending=False)
    selected = gains[gains > gains.median()].index.tolist() if median_rule else gains.head(top_k).index.tolist()
    return selected, gains

def gini_based_selection(X_train, y_train, top_k=10):
    model = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return importances.head(top_k).index.tolist(), importances

def split_ciciot(df, label_col="Label"):
    train, temp = train_test_split(df, test_size=0.30, stratify=df[label_col], random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp[label_col], random_state=RANDOM_STATE)
    print(f"CICIoT2023 split train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def split_ids2018(df, label_col="Label"):
    finetune, temp = train_test_split(df, test_size=0.95, stratify=df[label_col], random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=(0.80/0.95), stratify=temp[label_col], random_state=RANDOM_STATE)
    print(f"IDS2018 split fine-tune: {len(finetune):,} | val: {len(val):,} | test: {len(test):,}")
    return finetune.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


CICIOT_TO_IDS2018 = {
    "flow_duration":    "Flow Duration",
    "Header_Length":    None,   # IDS2018 splits Fwd/Bwd Header Length instead
    "Protocol Type":    "Protocol",
    "Min":              "Pkt Len Min",
    "Max":              "Pkt Len Max",
    "AVG":              "Pkt Len Mean",
    "Std":              "Pkt Len Std",
    "IAT":              "Flow IAT Mean",
    "Rate":             "Flow Pkts/s",
    "Srate":            "Fwd Pkts/s",
    "Drate":            "Bwd Pkts/s",
    "syn_flag_number":  "SYN Flag Cnt",
    "rst_flag_number":  "RST Flag Cnt",
    "psh_flag_number":  "PSH Flag Cnt",
    "ack_flag_number":  "ACK Flag Cnt",
    "urg_count":        "URG Flag Cnt",
    "fin_count":        "FIN Flag Cnt",
    "rst_count":        None,
    "Magnitude":        None,
    "Radius":           None,
    "Covariance":       None,
    "Weight":           None,
    "Tot sum":          "TotLen Fwd Pkts",
    "Tot size":         None,
    "Number":           None,
}

def harmonized_feature_set(feature_list):
    usable = [f for f in feature_list if CICIOT_TO_IDS2018.get(f)]
    dropped = [f for f in feature_list if not CICIOT_TO_IDS2018.get(f)]
    print(f"Harmonizable: {len(usable)}/{len(feature_list)}  |  dropped: {dropped}")
    return usable

def project_ids2018_to_ciciot_names(df, feature_list):
    # Renames the usable IDS2018 columns to their CICIoT2023 equivalents so the same model trained on CICIoT2023 columns can score IDS2018 rows without retraining.
    usable = harmonized_feature_set(feature_list)
    rename_map = {CICIOT_TO_IDS2018[f]: f for f in usable}
    df = df.rename(columns=rename_map)
    missing = [f for f in usable if f not in df.columns]
    if missing:
        print(f"[WARN] still missing after rename: {missing}")
    return df[[f for f in usable if f in df.columns]], usable

def fit_scaler(train_df, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler

def apply_scaler(df, feature_cols, scaler):
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df

def build_xgb_objective(X_train, y_train, X_val, y_val, num_class):
    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": num_class,
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.10),
            "max_depth": trial.suggest_int("max_depth", 6, 8),
            "n_estimators": trial.suggest_int("n_estimators", 300, 500),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "gamma": trial.suggest_float("gamma", 0.0, 0.2),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.1),
            "tree_method": XGB_TREE_METHOD,
            "device": XGB_DEVICE,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
        return f1
    return objective

def tune_heavy_model(X_train, y_train, X_val, y_val, num_class, n_trials=30):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(build_xgb_objective(X_train, y_train, X_val, y_val, num_class), n_trials=n_trials)
    print("Best macro-F1 (val):", study.best_value)
    print("Best params:", study.best_params)
    return study.best_params

def train_heavy_model(X_train, y_train, X_val, y_val, num_class, best_params):
    params = {**best_params, "objective": "multi:softprob", "num_class": num_class,
              "tree_method": XGB_TREE_METHOD, "device": XGB_DEVICE,
              "random_state": RANDOM_STATE, "n_jobs": -1,
              "eval_metric": ["mlogloss", "merror"]}
    t0 = time.perf_counter()
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    print(f"Heavy model trained on device='{XGB_DEVICE}' in {time.perf_counter()-t0:.1f}s")
    return model, model.evals_result()

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
def plot_xgb_training_curve(evals_result, model_name="HeavyNet (XGBoost)"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        rounds = range(len(evals_result["validation_0"]["mlogloss"]))

        axes[0].plot(rounds, evals_result["validation_0"]["mlogloss"], label="training")
        axes[0].plot(rounds, evals_result["validation_1"]["mlogloss"], label="validation")
        axes[0].set_xlabel("boosting round"); axes[0].set_ylabel("mlogloss")
        axes[0].set_title(f"{model_name} — log loss per round")
        axes[0].legend()

        axes[1].plot(rounds, evals_result["validation_0"]["merror"], label="training")
        axes[1].plot(rounds, evals_result["validation_1"]["merror"], label="validation")
        axes[1].set_xlabel("boosting round"); axes[1].set_ylabel("multiclass error rate")
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
    ax1.bar(x - width/2, heavy_vals, width, label="Heavy (XGBoost)")
    ax1.bar(x + width/2, lite_vals, width, label="Lite (Decision Tree)")
    ax1.set_xticks(x); ax1.set_xticklabels(metrics)
    ax1.set_ylabel("score")
    ax1.set_title("Heavy vs. Lite — accuracy metrics")
    ax1.legend()
    plt.tight_layout()
    plt.show()

    fig, ax2 = plt.subplots(figsize=(6, 5))
    ax2.bar(["Heavy", "Lite"],
            [heavy_results["inference_latency_us_per_sample"], lite_results["inference_latency_us_per_sample"]],
            color=["steelblue", "orange"])
    ax2.set_ylabel("inference latency (µs/sample)")
    ax2.set_title("Heavy vs. Lite — inference latency")
    plt.tight_layout()
    plt.show()

def export_heavy_to_onnx(xgb_model, n_features, out_path):
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Heavy model exported to {out_path}")

def export_lite_to_onnx(dt_model, n_features, out_path):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(dt_model, initial_types=initial_type)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Lite model exported to {out_path}")
    

def run_pipeline(sample_frac_ciciot=0.05, sample_frac_ids2018=0.3, optuna_trials=20):
    # [Flow Feature Extraction Engine]
    ciciot_raw = load_ciciot2023(sample_frac=sample_frac_ciciot)
    ciciot_raw = normalize_columns(ciciot_raw)
    ids2018_raw = load_ids2018(sample_frac=sample_frac_ids2018)

    # [Data Cleaning & Normalization Module] + [Common Feature Processing Layer]
    ciciot = clean_dataframe(ciciot_raw)
    ciciot, proto_encoder = encode_protocol(ciciot)
    ids2018 = clean_dataframe(ids2018_raw)

    # Splits
    train, val, test = split_ciciot(ciciot)
    ft, ext_val, ext_test = split_ids2018(ids2018)
    # print(sorted(ciciot.columns.tolist()))
    # print(sorted(ids2018.columns.tolist()))
    # Target encoding
    label_encoder = LabelEncoder()
    train = train.copy(); val = val.copy(); test = test.copy()
    train["y"] = label_encoder.fit_transform(train["Label"])
    val["y"] = label_encoder.transform(val["Label"])
    test["y"] = label_encoder.transform(test["Label"])
    num_class = len(label_encoder.classes_)

    # Scaling (train-only fit)
    # heavy_scaler = fit_scaler(train, HEAVY_FEATURES)
    # lite_scaler = fit_scaler(train, LITE_FEATURES)

    heavy_features, heavy_missing = resolve_feature_set(train, HEAVY_FEATURES, "HEAVY_FEATURES")
    lite_features, lite_missing = resolve_feature_set(train, LITE_FEATURES, "LITE_FEATURES")
    
    if heavy_missing or lite_missing:
        print("\n[RESEARCH NOTE] This data pull does not match the full 47-column CICIoT2023 "
              "schema referenced in the design." )
    
    heavy_scaler = fit_scaler(train, heavy_features)
    lite_scaler = fit_scaler(train, lite_features)

    train_h = apply_scaler(train, heavy_features, heavy_scaler)
    val_h = apply_scaler(val, heavy_features, heavy_scaler)
    test_h = apply_scaler(test, heavy_features, heavy_scaler)

    train_l = apply_scaler(train, lite_features, lite_scaler)
    val_l = apply_scaler(val, lite_features, lite_scaler)
    test_l = apply_scaler(test, lite_features, lite_scaler)

    best_params = tune_heavy_model(train_h[heavy_features], train_h["y"],
                                    val_h[heavy_features], val_h["y"],
                                    num_class, n_trials=optuna_trials)
    heavy_model, heavy_evals = train_heavy_model(train_h[heavy_features], train_h["y"],
                                                  val_h[heavy_features], val_h["y"],
                                                  num_class, best_params)

    lite_model = train_lite_model(train_l[lite_features], train_l["y"])

    print("\n Heavy model (CICIoT2023 test):")
    evaluate_model(heavy_model, test_h[heavy_features], test_h["y"],
                    label_names=label_encoder.classes_, model_name="heavy_xgboost")
    print("\nLite model (CICIoT2023 test):")
    evaluate_model(lite_model, test_l[lite_features], test_l["y"],
                    label_names=label_encoder.classes_, model_name="lite_decision_tree")

    heavy_usable = harmonized_feature_set(heavy_features)
    lite_usable = harmonized_feature_set(lite_features)

    plot_xgb_training_curve(heavy_evals)
    plot_confusion_matrix(heavy_model, test_h[heavy_features], test_h["y"], label_encoder.classes_, "HeavyNet (XGBoost)")
    plot_confusion_matrix(lite_model, test_l[lite_features], test_l["y"], label_encoder.classes_, "LiteNet (Decision Tree)")
    plot_feature_importance(heavy_model, heavy_features, "HeavyNet (XGBoost)")
    plot_feature_importance(lite_model, lite_features, "LiteNet (Decision Tree)")
    plot_learning_curve(DecisionTreeClassifier(**LITE_PARAMS), train_l[lite_features], train_l["y"], "LiteNet (Decision Tree)")
    heavy_results = evaluate_model(heavy_model, test_h[heavy_features], test_h["y"], label_encoder.classes_, "heavy_xgboost")
    lite_results = evaluate_model(lite_model, test_l[lite_features], test_l["y"], label_encoder.classes_, "lite_decision_tree")
    plot_model_comparison(heavy_results, lite_results)

    return {
        "heavy_model": heavy_model, "lite_model": lite_model,
        "label_encoder": label_encoder,
        "heavy_scaler": heavy_scaler, "lite_scaler": lite_scaler,
        "heavy_features": heavy_features, "lite_features": lite_features,   # <- add these
        "test_h": test_h, "test_l": test_l,
        "ext_val": ext_val, "ext_test": ext_test, "ft": ft,
        "heavy_usable": heavy_usable, "lite_usable": lite_usable,
    }


results = run_pipeline(sample_frac_ciciot=0.02, sample_frac_ids2018=0.2, optuna_trials=10)
