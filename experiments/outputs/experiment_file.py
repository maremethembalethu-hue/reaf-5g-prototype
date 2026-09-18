
import glob
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # never opens a window, never blocks, never prompts
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

 
# GPU detection 
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
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            GPU_AVAILABLE = True
            GPU_INFO = out.stdout.strip().splitlines()[0]
    except Exception:
        pass

XGB_TREE_METHOD = "hist"
XGB_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
print(f"GPU available: {GPU_AVAILABLE} ({GPU_INFO}) | XGBoost device: {XGB_DEVICE}")
 
# Paths: EDIT THESE to your real locations

WATAI_ROOT = Path("data/wataiData/csv/CICIoT2023")          # part-00000...part-001683 *.csv
MERGED_ROOT = Path("data/MERGED_CSV/MERGED_CSV")        # Merged01.csv ... Merged63.csv

OUTPUT_DIR = Path(f"outputs/experiments_{RUN_ID}")
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
for _d in (OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MASTER_RESULTS_CSV = RESULTS_DIR / "master_results.csv"
master_rows = []


def save_result_row(row: dict):
    # Append one result row and rewrite the master CSV immediately, so no
    # results are lost if a later experiment crashes
    row = {"run_id": RUN_ID, "timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    master_rows.append(row)
    pd.DataFrame(master_rows).to_csv(MASTER_RESULTS_CSV, index=False)


def save_plot(fig, descriptive_name: str):
    """Save + close a matplotlib figure under a descriptive, collision-free
    filename. Never calls plt.show() and never prompts."""
    safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", descriptive_name).strip("_")
    path = PLOTS_DIR / f"{safe_name}.png"
    counter = 1
    while path.exists():
        path = PLOTS_DIR / f"{safe_name}_{counter}.png"
        counter += 1
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot saved] {path}")
    return path
 
# Feature schema

# The merged dataset's exact 39 features
MERGED_39_FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]
assert len(MERGED_39_FEATURES) == 39

# The raw wataiData CICIoT2023 export's exact 46 features.
WATAI_46_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitude", "Radius", "Covariance", "Variance", "Weight",
]
assert len(WATAI_46_FEATURES) == 46

# Derived relationship between the two schemas, used by the Experiment 3
# ablation below
merged_set, watai_setData = set(MERGED_39_FEATURES), set(WATAI_46_FEATURES)
COMMON_FEATURES = [f for f in MERGED_39_FEATURES if f in watai_setData]            # present in both (37)
MERGED_ONLY_FEATURES = [f for f in MERGED_39_FEATURES if f not in watai_setData]    # e.g. Time_To_Live, IGMP
WATAI_ONLY_FEATURES = [f for f in WATAI_46_FEATURES if f not in merged_set]     # e.g. Srate, Drate, Magnitude...
SRATE_DRATE_FEATURES = [f for f in ("Srate", "Drate") if f in watai_setData]
print(f"Feature schema: {len(COMMON_FEATURES)} common | "
      f"{len(MERGED_ONLY_FEATURES)} merged-only {MERGED_ONLY_FEATURES} | "
      f"{len(WATAI_ONLY_FEATURES)} watai-only {WATAI_ONLY_FEATURES}")

# 8-class attack-family grouping 
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


 
# Data loading

def find_label_column(df):
    for cand in ("label", "Label", "LABEL"):
        if cand in df.columns:
            return cand
    raise KeyError(f"No label column found among columns: {list(df.columns)[:15]}...")


def downcast_numeric(df):
    # Halves numeric memory footprint float64 to float32, int64 to smallest
    # safe int dtype on each CHUNK right after it's read, before it's ever
    # concatenated or copied.
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(np.float32)
    int_cols = df.select_dtypes(include=["int64"]).columns
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def report_memory(df, name):
    gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"{name}: {len(df):,} rows x {df.shape[1]} cols  ~{gb:.2f} GB in memory)")
    if gb > 8:
        print(f"W {name} is large ~{gb:.1f} GB. Every downstream. ")


def load_watai_dataset(root=WATAI_ROOT, sample_frac=None, chunksize=500_000):
    
    # wataiData/csv/CICIoT2023 holds Spark-style partition files named
    # part-00000-<uuid>-c000.csv ... part-001683-<uuid>-c000.csv. Only the
    # numeric part id is fixed.
    files = sorted(glob.glob(str(root / "part-*.csv")))
    if not files:
        print(f"W No part-*.csv files found under {root}")
        return pd.DataFrame()
    frames = []
    for i, fpath in enumerate(files):
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            chunk = downcast_numeric(chunk)
            frames.append(chunk)
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  wataiData: loaded {i + 1}/{len(files)} part files")
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    label_col = find_label_column(full)
    if label_col != "label":
        full = full.rename(columns={label_col: "label"})
    report_memory(full, "wataiData")
    return full


def load_merged_dataset(root=MERGED_ROOT, start=1, end=63, sample_frac=None, chunksize=500_000):
    files = [root / f"Merged{i:02d}.csv" for i in range(start, end + 1)]
    files = [f for f in files if f.exists()]
    if not files:
        print(f"W No Merged*.csv files found under {root}")
        return pd.DataFrame()
    frames = []
    for i, fpath in enumerate(files):
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            chunk = downcast_numeric(chunk)
            frames.append(chunk)
        print(f"  merged: loaded {i + 1}/{len(files)} files ({fpath.name})")
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    label_col = find_label_column(full)
    if label_col != "label":
        full = full.rename(columns={label_col: "label"})
    report_memory(full, "merged dataset")
    return full

# Shared preprocessing
def clean_dataframe(df):
    # No leading df.copy() here on purpose: drop_duplicates() below already
    # returns an independent frame, it does not mutate in place unless
    # inplace=True is passed, so an extra full-frame copy before it just
    # doubles peak memory for no benefit on a full-volume wataiData load
    # tens of millions of rows that redundant copy alone is enough to
    # exhaust RAM.
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df):,} rows with NaN/Inf")
    if "label" in df.columns:
        df["label"] = df["label"].replace({"BenignTraffic": "Benign_Final", "BENIGN": "Benign_Final"})
    return df.reset_index(drop=True)


def encode_protocol(df, col="Protocol Type"):
    if col not in df.columns:
        return df, None
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df, le


def map_to_attack_family(df, label_col="label", family_map=ATTACK_FAMILY_MAP):
    # No copy here either this is only ever called (via prepare_dataset)
    # on the output of encode_protocol(clean_dataframe(raw_df)), which is
    # already an independent frame, never the original raw_df/raw_merged
    # that gets reused across experiments.
    normalized_map = {k.upper(): v for k, v in family_map.items()}
    mapped = df[label_col].astype(str).str.upper().map(normalized_map)
    unmapped = df.loc[mapped.isna(), label_col].unique().tolist()
    if unmapped:
        print(f"W {len(unmapped)} label(s) unmapped, left unchanged: {unmapped[:10]}")
    df[label_col] = mapped.fillna(df[label_col])
    return df


def resolve_feature_set(df, requested_features):
    available = [c for c in requested_features if c in df.columns]
    missing = [c for c in requested_features if c not in df.columns]
    if missing:
        print(f"W {len(missing)} requested feature(s) not present, dropped: {missing}")
    return available



# Data-driven Heavy/Lite feature selection: Pearson correlation filter,
# then rank the survivors by Gini-impurity feature importance single
# DecisionTreeClassifier fit the same method the production pipeline
# uses see pearson_filter/gini_based_selection/select_features_for_track
# in the hierarchical variant. 
def pearson_filter(df, feature_cols, threshold=0.90):
    if len(feature_cols) < 2:
        return feature_cols, []
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in feature_cols if c not in to_drop]
    print(f"Pearson filter: dropped {len(to_drop)} of {len(feature_cols)} features (|r| > {threshold}): {to_drop}")
    return kept, to_drop


def gini_rank_features(train_df, target_col, candidate_features):
    # Fits one Gini-criterion decision tree on the candidates and returns
    # them ranked most-important first, alongside their importances.
    model = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    model.fit(train_df[candidate_features], train_df[target_col])
    importances = pd.Series(model.feature_importances_, index=candidate_features).sort_values(ascending=False)
    return importances.index.tolist(), importances


def select_heavy_lite_features(train_df, candidate_features, target_col="y",
                                heavy_top_k=25, lite_top_k=15, pearson_threshold=0.90):
   
    # Pearson-filter the candidate list, rank the survivors by Gini importance,
    # then take the top heavy_top_k as the Heavy set and the top lite_top_k
    # a further-narrowed subset of Heavy as the Lite set. 
    pearson_kept, pearson_dropped = pearson_filter(train_df, candidate_features, threshold=pearson_threshold)
    if len(pearson_kept) < 2:
        pearson_kept = candidate_features

    ranked_features, importances = gini_rank_features(train_df, target_col, pearson_kept)

    heavy_features = ranked_features[:heavy_top_k]
    lite_features = ranked_features[:lite_top_k]
    print(f"Selected Heavy: {len(heavy_features)} | Lite: {len(lite_features)} "
          f"(from {len(pearson_kept)} Pearson survivors of {len(candidate_features)} candidates)")
    diagnostics = {
        "pearson_kept": pearson_kept, "pearson_dropped": pearson_dropped,
        "gini_ranked": ranked_features, "gini_importances": importances,
    }
    return heavy_features, lite_features, diagnostics


def split_dataset(df, label_col="label", test_size=0.20, val_frac_of_train=0.125, random_state=RANDOM_STATE):
    train_full, test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=random_state)
    train, val = train_test_split(train_full, test_size=val_frac_of_train,
                                   stratify=train_full[label_col], random_state=random_state)
    print(f"Stratified 80/20 split -> train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def cap_rows_stratified(df, label_col, max_rows, random_state=RANDOM_STATE):
    # If df has more rows than max_rows, take a stratified sample down to
    # max_rows. Val/test splits are NOT touched by undersample_train (that only caps TRAIN), so on a full-volume load
    # they can still be tens of millions of rows capping them here is what
    # stops every later fit_scaler/apply_scaler/evaluate_model call in every
    # experiment from repeatedly copying a huge frame. A few hundred thousand
    # rows is already far more than needed for a stable macro-F1 estimate.
    if max_rows is None or len(df) <= max_rows:
        return df
    sampled, _ = train_test_split(df, train_size=max_rows, stratify=df[label_col], random_state=random_state)
    print(f"Capped eval split: {len(df):,} -> {len(sampled):,} rows (stratified, max_eval_rows={max_rows:,})")
    return sampled.reset_index(drop=True)


def undersample_train(df, label_col="y", min_per_class=1000, target_total=None, random_state=RANDOM_STATE):
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
    return out


def fit_scaler(train_df, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df, feature_cols, scaler):
    # Slim to just what's actually used downstream instead of copying every original column
    # this function is called many times per experiment (once per
    # feature-list x heavy/lite track), and on an uncapped multi-million
    # row frame a full-width copy each time adds up fast.
    feature_cols = list(feature_cols)
    extra_cols = [c for c in ("label", "y") if c in df.columns and c not in feature_cols]
    slim = df[feature_cols + extra_cols].copy()
    slim[feature_cols] = scaler.transform(slim[feature_cols])
    return slim


def compute_sample_weights(y, max_ratio=40.0):
    counts = np.bincount(np.asarray(y))
    N, K = len(y), len(counts)
    class_weights = N / (K * counts)
    class_weights = np.clip(class_weights, class_weights.min(), class_weights.min() * max_ratio)
    return class_weights[np.asarray(y)]


def prepare_dataset(raw_df, min_per_class=1000, undersample_target_total=None, max_eval_rows=300_000):
    # Clean to encode protocol to map to 8-class family to stratified split
    # to cap val/test to max_eval_rows to label-encode to stratified
    # undersample TRAIN only. 
    df = clean_dataframe(raw_df)
    df, proto_encoder = encode_protocol(df)
    df = map_to_attack_family(df)
    train, val, test = split_dataset(df)
    val = cap_rows_stratified(val, "label", max_eval_rows)
    test = cap_rows_stratified(test, "label", max_eval_rows)

    full_label_encoder = LabelEncoder()
    train["y"] = full_label_encoder.fit_transform(train["label"])
    val["y"] = full_label_encoder.transform(val["label"])
    test["y"] = full_label_encoder.transform(test["label"])
    train = undersample_train(train, label_col="y", min_per_class=min_per_class,
                               target_total=undersample_target_total)
    return {"train": train, "val": val, "test": test,
            "proto_encoder": proto_encoder, "label_encoder": full_label_encoder}



 
# Model zoo
def xgb_objective_params(num_class):
    if num_class == 2:
        return {"objective": "binary:logistic", "eval_metric": "logloss"}
    return {"objective": "multi:softprob", "num_class": num_class, "eval_metric": "mlogloss"}


def tune_xgb(X_train, y_train, X_val, y_val, num_class, n_trials=8):
    def objective(trial):
        params = {
            **xgb_objective_params(num_class),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "tree_method": XGB_TREE_METHOD, "device": XGB_DEVICE,
            "random_state": RANDOM_STATE, "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
        return f1
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def train_heavy_xgboost(X_train, y_train, X_val, y_val, num_class, n_trials=8):
    best_params = tune_xgb(X_train, y_train, X_val, y_val, num_class, n_trials=n_trials)
    params = {**best_params, **xgb_objective_params(num_class),
              "tree_method": XGB_TREE_METHOD, "device": XGB_DEVICE,
              "random_state": RANDOM_STATE, "n_jobs": -1}
    sample_weights = compute_sample_weights(y_train)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def train_heavy_lightgbm(X_train, y_train, X_val=None, y_val=None, num_class=2):
    is_binary = num_class == 2
    kwargs = {} if is_binary else {"num_class": num_class}
    model = lgb.LGBMClassifier(objective="binary" if is_binary else "multiclass", **kwargs,
                                n_estimators=300, learning_rate=0.08, num_leaves=31,
                                min_child_samples=50, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    sample_weights = compute_sample_weights(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def train_heavy_catboost(X_train, y_train, X_val=None, y_val=None, num_class=2):
    model = cb.CatBoostClassifier(loss_function="Logloss" if num_class == 2 else "MultiClass",
                                   iterations=300, depth=6, learning_rate=0.08,
                                   random_state=RANDOM_STATE, verbose=False)
    sample_weights = compute_sample_weights(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def train_heavy_random_forest(X_train, y_train, X_val=None, y_val=None, num_class=2):
    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


HEAVY_MODEL_BUILDERS = {
    "xgboost": train_heavy_xgboost,
    "lightgbm": train_heavy_lightgbm,
    "catboost": train_heavy_catboost,
    "random_forest": train_heavy_random_forest,
}

LITE_PARAMS = dict(criterion="gini", splitter="best", max_depth=15, min_samples_split=10,
                    min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE)


def train_lite_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(**LITE_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_lite_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=5,
                                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_lite_logreg(X_train, y_train):
    model = LogisticRegression(solver="lbfgs", max_iter=500, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_lite_extra_trees(X_train, y_train):
    model = ExtraTreesClassifier(n_estimators=50, max_depth=10, min_samples_leaf=5,
                                  class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


LITE_MODEL_BUILDERS = {
    "decision_tree": train_lite_decision_tree,
    "random_forest_lite": train_lite_random_forest,
    "logistic_regression": train_lite_logreg,
    "extra_trees": train_lite_extra_trees,
}



# Evaluation & plotting 
def evaluate_model(model, X_test, y_test, model_name="model"):
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0
    latency_us = (elapsed / max(len(X_test), 1)) * 1e6
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        pass
    metrics = {
        "model": model_name, "n_test": len(X_test),
        "accuracy": acc, "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro,
        "weighted_precision": p_w, "weighted_recall": r_w, "weighted_f1": f1_w,
        "roc_auc_macro_ovr": roc_auc, "inference_latency_us_per_sample": latency_us,
    }
    return metrics, y_pred


def plot_confusion(y_test, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), max(6, len(class_names) * 0.8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    save_plot(fig, filename)


def plot_feature_importance(model, feature_names, title, filename):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.3)))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    ax.set_xlabel("importance")
    ax.set_title(title)
    plt.tight_layout()
    save_plot(fig, filename)


def plot_bar_comparison(labels, values_dict, ylabel, title, filename):
    x = np.arange(len(labels))
    width = 0.8 / max(len(values_dict), 1)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.3), 5))
    for i, (name, vals) in enumerate(values_dict.items()):
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    plt.tight_layout()
    save_plot(fig, filename)


def plot_line(x_vals, series_dict, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, y_vals in series_dict.items():
        ax.plot(x_vals, y_vals, marker="o", label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    plt.tight_layout()
    save_plot(fig, filename)



# Reusable "train the whole zoo on one feature set" helper
def run_model_zoo_flat(prepared, feature_list, tag, experiment, tracks=("heavy", "lite")):
    # Trains the requested track of the model zoo on the flat 8-class
    # target with the given feature list. Every model's metrics are appended
    # to the master CSV and its confusion-matrix / feature-importance plots
    # are saved. Returns the list of result rows and the fitted models.
    train, val, test = prepared["train"], prepared["val"], prepared["test"]
    label_encoder = prepared["label_encoder"]
    features = resolve_feature_set(train, feature_list)
    if len(features) < 2:
        print(f"S {tag}: fewer than 2 usable features")
        return [], {}

    scaler = fit_scaler(train, features)
    train_s = apply_scaler(train, features, scaler)
    val_s = apply_scaler(val, features, scaler)
    test_s = apply_scaler(test, features, scaler)
    num_class = len(label_encoder.classes_)
    class_names = list(label_encoder.classes_)

    rows, models = [], {}

    if "heavy" in tracks:
        for name, builder in HEAVY_MODEL_BUILDERS.items():
            model = builder(train_s[features], train_s["y"], val_s[features], val_s["y"], num_class)
            metrics, y_pred = evaluate_model(model, test_s[features], test_s["y"], model_name=f"{tag}_{name}")
            metrics.update({"experiment": experiment, "tag": tag, "track": "heavy", "model_kind": name,
                             "n_features": len(features)})
            save_result_row(metrics)
            rows.append(metrics)
            models[name] = model
            plot_confusion(test_s["y"], y_pred, class_names,
                            f"{experiment} | {tag} | Heavy-{name} confusion matrix",
                            f"{experiment}__{tag}__heavy_{name}__confusion")
            plot_feature_importance(model, features,
                                     f"{experiment} | {tag} | Heavy-{name} feature importance",
                                     f"{experiment}__{tag}__heavy_{name}__feature_importance")

    if "lite" in tracks:
        for name, builder in LITE_MODEL_BUILDERS.items():
            model = builder(train_s[features], train_s["y"])
            metrics, y_pred = evaluate_model(model, test_s[features], test_s["y"], model_name=f"{tag}_{name}")
            metrics.update({"experiment": experiment, "tag": tag, "track": "lite", "model_kind": name,
                             "n_features": len(features)})
            save_result_row(metrics)
            rows.append(metrics)
            models[name] = model
            plot_confusion(test_s["y"], y_pred, class_names,
                            f"{experiment} | {tag} | Lite-{name} confusion matrix",
                            f"{experiment}__{tag}__lite_{name}__confusion")
            plot_feature_importance(model, features,
                                     f"{experiment} | {tag} | Lite-{name} feature importance",
                                     f"{experiment}__{tag}__lite_{name}__feature_importance")

    return rows, models


 
# Experiment 1: dataset comparison, all available features

def experiment_1_dataset_comparison(watai_raw, merged_raw, min_per_class=1000, max_eval_rows=300_000):
    print( "\nEXPERIMENT 1: Dataset comparison (watai-46 vs merged-39)\n")
    results = {}
    if len(watai_raw):
        prepared = prepare_dataset(watai_raw, min_per_class=min_per_class, max_eval_rows=max_eval_rows)
        rows, _ = run_model_zoo_flat(prepared, WATAI_46_FEATURES, tag="watai_all46",
                                      experiment="exp1_dataset_comparison")
        results["watai_all46"] = rows
    if len(merged_raw):
        prepared = prepare_dataset(merged_raw, min_per_class=min_per_class, max_eval_rows=max_eval_rows)
        rows, _ = run_model_zoo_flat(prepared, MERGED_39_FEATURES, tag="merged_all39",
                                      experiment="exp1_dataset_comparison")
        results["merged_all39"] = rows

    all_models = sorted({r["model_kind"] for rows in results.values() for r in rows})
    values = {}
    for tag, rows in results.items():
        by_model = {r["model_kind"]: r["macro_f1"] for r in rows}
        values[tag] = [by_model.get(m, np.nan) for m in all_models]
    if values:
        plot_bar_comparison(all_models, values, "macro F1 (test)",
                             "Experiment 1: watai (46 features) vs merged (39 features) — all models",
                             "exp1__dataset_comparison__macro_f1_by_model")
    return results



# Experiment 2: data-driven Heavy/Lite feature selection
# computed independently on each dataset's own candidate
# feature list, then benchmarked against the "all features" baseline
# from Experiment 1 and swept across k to justify the 25/15 cutoff.

FEATURE_SELECTION_HEAVY_TOP_K = 25
FEATURE_SELECTION_LITE_TOP_K = 15
FEATURE_COUNT_SWEEP = (5, 10, 15, 20, 25, 30, 35)  # capped per-dataset to its candidate count


def experiment_2_feature_reduction(watai_raw, merged_raw, min_per_class=1000, max_eval_rows=300_000):
    print(
          f"\nEXPERIMENT 2: Data-driven feature selection "
          f"(Pearson filter to Gini rank to top-{FEATURE_SELECTION_HEAVY_TOP_K} heavy / "
          f"top-{FEATURE_SELECTION_LITE_TOP_K} lite)\n")
    results = {}
    selections = {}  # dataset_name -> {"heavy": [...], "lite": [...], "diagnostics": {...}}
    selection_summary_rows = []

    for name, raw, candidates in (("watai", watai_raw, WATAI_46_FEATURES),
                                   ("merged", merged_raw, MERGED_39_FEATURES)):
        if not len(raw):
            continue
        prepared = prepare_dataset(raw, min_per_class=min_per_class, max_eval_rows=max_eval_rows)
        train = prepared["train"]
        candidate_features = resolve_feature_set(train, candidates)

        heavy_features, lite_features, diag = select_heavy_lite_features(
            train, candidate_features, target_col="y",
            heavy_top_k=FEATURE_SELECTION_HEAVY_TOP_K, lite_top_k=FEATURE_SELECTION_LITE_TOP_K)
        selections[name] = {"heavy": heavy_features, "lite": lite_features, "diagnostics": diag}

        #  record every candidate feature's rank/importance for the writeup 
        lite_set, heavy_set = set(lite_features), set(heavy_features)
        pearson_dropped_set = set(diag["pearson_dropped"])
        for rank, feat in enumerate(diag["gini_ranked"], start=1):
            selection_summary_rows.append({
                "dataset": name, "feature": feat, "gini_importance": float(diag["gini_importances"][feat]),
                "gini_rank": rank, "pearson_survivor": feat not in pearson_dropped_set,
                "selected_heavy_top25": feat in heavy_set, "selected_lite_top15": feat in lite_set,
            })
        for feat in diag["pearson_dropped"]:
            selection_summary_rows.append({
                "dataset": name, "feature": feat, "gini_importance": np.nan,
                "gini_rank": np.nan, "pearson_survivor": False,
                "selected_heavy_top25": False, "selected_lite_top15": False,
            })

        #  Gini importance bar chart for the selected candidates 
        top_n_to_plot = min(len(diag["gini_ranked"]), 35)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n_to_plot * 0.28)))
        plot_feats = diag["gini_ranked"][:top_n_to_plot][::-1]
        plot_vals = [diag["gini_importances"][f] for f in plot_feats]
        colors = ["#1f77b4" if f in heavy_set else "#c7c7c7" for f in plot_feats]
        ax.barh(plot_feats, plot_vals, color=colors)
        ax.set_xlabel("Gini importance")
        ax.set_title(f"Experiment 2: {name} — Gini-ranked features (blue = selected in top-"
                     f"{FEATURE_SELECTION_HEAVY_TOP_K})")
        plt.tight_layout()
        save_plot(fig, f"exp2__{name}__gini_feature_ranking")

        #  benchmark heavy-25 (heavy zoo) and lite-15 (lite zoo) 
        heavy_rows, _ = run_model_zoo_flat(prepared, heavy_features, tag=f"{name}_heavy_top25",
                                            experiment="exp2_feature_reduction", tracks=("heavy",))
        lite_rows, _ = run_model_zoo_flat(prepared, lite_features, tag=f"{name}_lite_top15",
                                           experiment="exp2_feature_reduction", tracks=("lite",))
        results[f"{name}_heavy_top25"] = heavy_rows
        results[f"{name}_lite_top15"] = lite_rows

        #  top-k sweep: does going past top-25/15 actually buy anything? 
        sweep_rows = []
        ranked = diag["gini_ranked"]
        ks = sorted({k for k in FEATURE_COUNT_SWEEP if k < len(ranked)} | {len(ranked)})
        val, test = prepared["val"], prepared["test"]
        num_class = len(prepared["label_encoder"].classes_)
        for k in ks:
            feats_k = ranked[:k]
            scaler_k = fit_scaler(train, feats_k)
            train_k = apply_scaler(train, feats_k, scaler_k)
            val_k = apply_scaler(val, feats_k, scaler_k)
            test_k = apply_scaler(test, feats_k, scaler_k)
            model_k = train_heavy_xgboost(train_k[feats_k], train_k["y"], val_k[feats_k], val_k["y"],
                                           num_class, n_trials=5)
            metrics_k, _ = evaluate_model(model_k, test_k[feats_k], test_k["y"], model_name=f"{name}_xgboost_k{k}")
            metrics_k.update({"experiment": "exp2_feature_count_sweep", "tag": f"{name}_k{k}",
                               "track": "heavy", "model_kind": "xgboost", "n_features": k})
            save_result_row(metrics_k)
            sweep_rows.append(metrics_k)
            print(f"  [{name}] top-{k} features -> macro_f1={metrics_k['macro_f1']:.4f}")

        plot_line([r["n_features"] for r in sweep_rows],
                  {"macro_f1": [r["macro_f1"] for r in sweep_rows]},
                  "number of top Gini-ranked features used", "macro F1 (test)",
                  f"Experiment 2: {name} — feature-count sweep (justifies top-"
                  f"{FEATURE_SELECTION_HEAVY_TOP_K}/{FEATURE_SELECTION_LITE_TOP_K} cutoff)",
                  f"exp2__{name}__feature_count_sweep")

    #  selection diagnostics CSV, for the "why 25/15" writeup 
    if selection_summary_rows:
        sel_df = pd.DataFrame(selection_summary_rows)
        sel_path = RESULTS_DIR / "exp2__feature_selection_diagnostics.csv"
        sel_df.to_csv(sel_path, index=False)
        print(f"[csv saved] {sel_path}")

    #  reduced-feature-set comparison chart, all models 
    for name in ("watai", "merged"):
        heavy_key, lite_key = f"{name}_heavy_top25", f"{name}_lite_top15"
        if heavy_key in results and lite_key in results:
            models = sorted({r["model_kind"] for r in results[heavy_key] + results[lite_key]})
            by_model_heavy = {r["model_kind"]: r["macro_f1"] for r in results[heavy_key]}
            by_model_lite = {r["model_kind"]: r["macro_f1"] for r in results[lite_key]}
            vals = {f"heavy_top{FEATURE_SELECTION_HEAVY_TOP_K}_features": [by_model_heavy.get(m, np.nan) for m in models],
                    f"lite_top{FEATURE_SELECTION_LITE_TOP_K}_features": [by_model_lite.get(m, np.nan) for m in models]}
            plot_bar_comparison(models, vals, "macro F1 (test)",
                                 f"Experiment 2: {name} — data-driven reduced feature sets by model",
                                 f"exp2__{name}__reduced_features__macro_f1_by_model")

    return {"results": results, "selections": selections}



# Experiment 3; Srate/Drate ablation (watai-only, controlled)
#   Two nested ablations, both using ONLY watai rows so the comparison is
#   never confounded by which dataset/rows are used:
#     (a) all-46 vs the 37 features watai shares with the merged schema
#         (drops all 9 watai-only engineered columns at once)
#     (b) all-46 vs all-46 minus ONLY Srate/Drate
#         (isolates the two features the hypothesis is actually about)
# 
def experiment_3_srate_drate_ablation(watai_raw, min_per_class=1000, max_eval_rows=300_000):
    print(
          "\nEXPERIMENT 3: Do Srate/Drate (and watai's other engineered columns)\n"
          "explain the merged(39)-feature dataset's weaker DDoS-vs-DoS separation?\n")
    if not len(watai_raw):
        print("S wataiData not available")
        return {}

    prepared = prepare_dataset(watai_raw, min_per_class=min_per_class, max_eval_rows=max_eval_rows)
    train, val, test = prepared["train"], prepared["val"], prepared["test"]
    num_class_full = len(prepared["label_encoder"].classes_)

    without_srate_drate_only = [f for f in WATAI_46_FEATURES if f not in SRATE_DRATE_FEATURES]
    variants = {
        "watai_all46": WATAI_46_FEATURES,
        "watai_common37_no_watai_only_cols": COMMON_FEATURES,
        "watai_46_minus_srate_drate_only": without_srate_drate_only,
    }
    print(f"Ablation variants: all46={len(WATAI_46_FEATURES)} features | "
          f"common37={len(COMMON_FEATURES)} features (drops {WATAI_ONLY_FEATURES}) | "
          f"46-minus-srate/drate={len(without_srate_drate_only)} features (drops {SRATE_DRATE_FEATURES})")

    overall_results, flood_results = {}, {}
    for tag, feature_list in variants.items():
        features = resolve_feature_set(train, feature_list)
        scaler = fit_scaler(train, features)
        train_s = apply_scaler(train, features, scaler)
        val_s = apply_scaler(val, features, scaler)
        test_s = apply_scaler(test, features, scaler)

        # overall flat 8-class, for context
        model = train_heavy_xgboost(train_s[features], train_s["y"], val_s[features], val_s["y"], num_class_full)
        metrics, _ = evaluate_model(model, test_s[features], test_s["y"], model_name=f"{tag}_xgboost_flat")
        metrics.update({"experiment": "exp3_srate_drate_ablation", "tag": tag, "track": "heavy",
                         "model_kind": "xgboost_flat8", "n_features": len(features)})
        save_result_row(metrics)
        overall_results[tag] = metrics

        # isolate DDoS vs DoS (Flood rows only)
        flood_train = train_s[train_s["label"].isin(FLOOD_CLASSES)]
        flood_val = val_s[val_s["label"].isin(FLOOD_CLASSES)]
        flood_test = test_s[test_s["label"].isin(FLOOD_CLASSES)]
        flood_encoder = LabelEncoder().fit(flood_train["label"])
        y_tr = pd.Series(flood_encoder.transform(flood_train["label"]))
        y_va = pd.Series(flood_encoder.transform(flood_val["label"]))
        y_te = pd.Series(flood_encoder.transform(flood_test["label"]))

        flood_model = train_heavy_xgboost(flood_train[features], y_tr, flood_val[features], y_va, 2)
        f_metrics, f_pred = evaluate_model(flood_model, flood_test[features], y_te,
                                            model_name=f"{tag}_xgboost_ddos_vs_dos")
        f_metrics.update({"experiment": "exp3_srate_drate_ablation", "tag": tag, "track": "heavy",
                           "model_kind": "xgboost_ddos_vs_dos", "n_features": len(features)})
        save_result_row(f_metrics)
        flood_results[tag] = f_metrics
        plot_confusion(y_te, f_pred, list(flood_encoder.classes_),
                        f"Experiment 3: {tag} — DDoS vs DoS confusion (Flood rows only)",
                        f"exp3__{tag}__ddos_vs_dos__confusion")

    labels = ["overall (8-class)", "DDoS vs DoS (Flood only)"]
    vals = {tag: [overall_results[tag]["macro_f1"], flood_results[tag]["macro_f1"]] for tag in variants}
    plot_bar_comparison(labels, vals, "macro F1 (test)",
                         "Experiment 3: effect of removing watai's engineered columns / Srate+Drate only",
                         "exp3__srate_drate_ablation__macro_f1_comparison")
    return {"overall": overall_results, "flood_only": flood_results}


 
# Experiment 4: data-volume scaling
def experiment_4_data_volume_scaling(raw_df, dataset_name, feature_list,
                                      fractions=(0.1, 0.25, 0.5, 0.75, 1.0), min_per_class=1000, max_eval_rows=300_000):
    print( f"\nEXPERIMENT 4: Data-volume scaling on {dataset_name}\n" )
    if not len(raw_df):
        print("S dataset not available")
        return []

    prepared = prepare_dataset(raw_df, min_per_class=min_per_class, max_eval_rows=max_eval_rows)
    train_full, val, test = prepared["train"], prepared["val"], prepared["test"]
    features = resolve_feature_set(train_full, feature_list)
    num_class = len(prepared["label_encoder"].classes_)

    rows = []
    for frac in fractions:
        if frac < 1.0:
            train_sub, _ = train_test_split(train_full, train_size=frac,
                                             stratify=train_full["y"], random_state=RANDOM_STATE)
        else:
            train_sub = train_full

        scaler = fit_scaler(train_sub, features)
        train_s = apply_scaler(train_sub, features, scaler)
        val_s = apply_scaler(val, features, scaler)
        test_s = apply_scaler(test, features, scaler)

        model = train_heavy_xgboost(train_s[features], train_s["y"], val_s[features], val_s["y"], num_class)
        metrics, _ = evaluate_model(model, test_s[features], test_s["y"],
                                     model_name=f"{dataset_name}_xgboost_frac{frac}")
        metrics.update({"experiment": "exp4_data_volume_scaling", "tag": f"{dataset_name}_frac_{frac}",
                         "track": "heavy", "model_kind": "xgboost", "n_features": len(features),
                         "train_fraction": frac, "n_train_rows": len(train_sub)})
        save_result_row(metrics)
        rows.append(metrics)
        print(f"  frac={frac:.2f} n_train={len(train_sub):,} macro_f1={metrics['macro_f1']:.4f}")

    plot_line([r["n_train_rows"] for r in rows],
              {"macro_f1": [r["macro_f1"] for r in rows], "accuracy": [r["accuracy"] for r in rows]},
              "training rows", "score",
              f"Experiment 4: {dataset_name} — does more training data help?",
              f"exp4__{dataset_name}__volume_vs_performance")
    return rows



# Experiment 5: flat baseline vs 3-stage cascade
class ThreeStageHeavyModel:
    # Soft mixture of Stage1 (benign vs attack) x Stage2 (6-way family,
    # Flood-merged) x Stage3 (DDoS vs DoS, Flood rows only). Stage 2 and
    # Stage 3 are both run on EVERY row, not only rows a hard label would
    # route there, so an earlier stage's mistake doesn't silently discard a
    # later stage's opinion — this is a mixture, not a hard cascade.

    def __init__(self, stage1_model, stage2_model, stage3_model,
                 stage2_encoder, stage3_encoder, full_label_encoder):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.stage3_model = stage3_model
        self.stage2_encoder = stage2_encoder
        self.stage3_encoder = stage3_encoder
        self.full_label_encoder = full_label_encoder

        stage2_classes = list(stage2_encoder.classes_)
        self.flood_idx_stage2 = stage2_classes.index("Flood")
        stage3_class_to_idx = {name: i for i, name in enumerate(stage3_encoder.classes_)}

        self._plan = []
        for name in full_label_encoder.classes_:
            if name == "Benign":
                self._plan.append(("benign", None))
            elif name in stage3_class_to_idx:
                self._plan.append(("stage3", stage3_class_to_idx[name]))
            else:
                self._plan.append(("stage2", stage2_classes.index(name)))

    def predict_proba(self, X):
        p1 = self.stage1_model.predict_proba(X)   # [benign, attack]
        p2 = self.stage2_model.predict_proba(X)    # 6-way family incl. Flood
        p3 = self.stage3_model.predict_proba(X)    # DDoS vs DoS
        attack_mass = p1[:, 1]
        flood_mass = p2[:, self.flood_idx_stage2]
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        out = np.zeros((n, len(self._plan)), dtype=np.float64)
        for col, (source, idx) in enumerate(self._plan):
            if source == "benign":
                out[:, col] = p1[:, 0]
            elif source == "stage2":
                out[:, col] = attack_mass * p2[:, idx]
            else:
                out[:, col] = attack_mass * flood_mass * p3[:, idx]
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def experiment_5_three_stage_justification(raw_df, dataset_name, feature_list, min_per_class=1000, max_eval_rows=300_000):
    print(
          f"\nEXPERIMENT 5: Flat 8-class baseline vs 3-stage cascade on {dataset_name}\n")
    if not len(raw_df):
        print("S dataset not available")
        return {}

    prepared = prepare_dataset(raw_df, min_per_class=min_per_class, max_eval_rows=max_eval_rows)
    train, val, test = prepared["train"], prepared["val"], prepared["test"]
    full_label_encoder = prepared["label_encoder"]
    features = resolve_feature_set(train, feature_list)
    num_class_full = len(full_label_encoder.classes_)
    class_names = list(full_label_encoder.classes_)

    scaler = fit_scaler(train, features)
    train_s = apply_scaler(train, features, scaler)
    val_s = apply_scaler(val, features, scaler)
    test_s = apply_scaler(test, features, scaler)

    #  Flat baseline 
    flat_model = train_heavy_xgboost(train_s[features], train_s["y"], val_s[features], val_s["y"], num_class_full)
    flat_metrics, flat_pred = evaluate_model(flat_model, test_s[features], test_s["y"], model_name="flat_xgboost")
    flat_metrics.update({"experiment": "exp5_three_stage_justification", "tag": dataset_name,
                          "track": "heavy", "model_kind": "flat_baseline", "n_features": len(features)})
    save_result_row(flat_metrics)
    plot_confusion(test_s["y"], flat_pred, class_names,
                    f"Experiment 5: {dataset_name} — FLAT 8-class baseline",
                    f"exp5__{dataset_name}__flat_baseline__confusion")

    #  Stage 1: benign vs attack 
    y1_tr = (train_s["label"] != "Benign").astype(int)
    y1_va = (val_s["label"] != "Benign").astype(int)
    stage1_model = train_heavy_xgboost(train_s[features], y1_tr, val_s[features], y1_va, 2)

    #  Stage 2: 6-way family (attack rows only, Flood merged) 
    attack_train = train_s[train_s["label"] != "Benign"]
    attack_val = val_s[val_s["label"] != "Benign"]
    stage2_encoder = LabelEncoder().fit(attack_train["label"].replace(FLOOD_MERGE))
    y2_tr = pd.Series(stage2_encoder.transform(attack_train["label"].replace(FLOOD_MERGE)))
    y2_va = pd.Series(stage2_encoder.transform(attack_val["label"].replace(FLOOD_MERGE)))
    stage2_model = train_heavy_xgboost(attack_train[features], y2_tr, attack_val[features], y2_va,
                                        len(stage2_encoder.classes_))

    #  Stage 3: DDoS vs DoS (Flood rows only) 
    flood_train = attack_train[attack_train["label"].isin(FLOOD_CLASSES)]
    flood_val = attack_val[attack_val["label"].isin(FLOOD_CLASSES)]
    stage3_encoder = LabelEncoder().fit(flood_train["label"])
    y3_tr = pd.Series(stage3_encoder.transform(flood_train["label"]))
    y3_va = pd.Series(stage3_encoder.transform(flood_val["label"]))
    stage3_model = train_heavy_xgboost(flood_train[features], y3_tr, flood_val[features], y3_va,
                                        len(stage3_encoder.classes_))

    #  Combine into cascade & evaluate on the FULL test set 
    cascade = ThreeStageHeavyModel(stage1_model, stage2_model, stage3_model,
                                    stage2_encoder, stage3_encoder, full_label_encoder)
    cascade_metrics, cascade_pred = evaluate_model(cascade, test_s[features], test_s["y"],
                                                     model_name="three_stage_cascade")
    cascade_metrics.update({"experiment": "exp5_three_stage_justification", "tag": dataset_name,
                             "track": "heavy", "model_kind": "three_stage_cascade", "n_features": len(features)})
    save_result_row(cascade_metrics)
    plot_confusion(test_s["y"], cascade_pred, class_names,
                    f"Experiment 5: {dataset_name} — 3-STAGE cascade",
                    f"exp5__{dataset_name}__three_stage_cascade__confusion")

    plot_bar_comparison(
        ["accuracy", "macro_f1", "weighted_f1"],
        {"flat_baseline": [flat_metrics[m] for m in ("accuracy", "macro_f1", "weighted_f1")],
         "three_stage_cascade": [cascade_metrics[m] for m in ("accuracy", "macro_f1", "weighted_f1")]},
        "score", f"Experiment 5: {dataset_name} — flat vs 3-stage cascade",
        f"exp5__{dataset_name}__flat_vs_cascade__summary")

    #  Per-class breakdown: WHERE does the cascade help/hurt? 
    per_class_rows = []
    y_true = test_s["y"].values
    for i, cls in enumerate(class_names):
        mask = (y_true == i)
        if mask.sum() == 0:
            continue
        flat_acc = (flat_pred[mask] == i).mean()
        cascade_acc = (cascade_pred[mask] == i).mean()
        per_class_rows.append({"experiment": "exp5_three_stage_justification", "tag": dataset_name,
                                "class": cls, "n": int(mask.sum()),
                                "flat_accuracy": flat_acc, "cascade_accuracy": cascade_acc,
                                "delta": cascade_acc - flat_acc})
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = RESULTS_DIR / f"exp5__{dataset_name}__per_class_flat_vs_cascade.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"[csv saved] {per_class_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(per_class_df))
    ax.bar(x - 0.2, per_class_df["flat_accuracy"], 0.4, label="flat baseline")
    ax.bar(x + 0.2, per_class_df["cascade_accuracy"], 0.4, label="3-stage cascade")
    ax.set_xticks(x); ax.set_xticklabels(per_class_df["class"], rotation=30, ha="right")
    ax.set_ylabel("per-class accuracy")
    ax.set_title(f"Experiment 5: {dataset_name} — per-class accuracy, flat vs cascade")
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f"exp5__{dataset_name}__per_class_flat_vs_cascade")

    return {"flat": flat_metrics, "cascade": cascade_metrics, "per_class": per_class_rows}


 
# Main
 
def main(sample_frac_watai=0.8, sample_frac_merged=0.8, min_per_class=1000, max_eval_rows=300_000):

    print(f"Run ID: {RUN_ID}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    watai_raw = load_watai_dataset(sample_frac=sample_frac_watai)
    merged_raw = load_merged_dataset(sample_frac=sample_frac_merged)
    if not len(watai_raw) and not len(merged_raw):
        raise RuntimeError("Neither dataset could be loaded — check WATAI_ROOT / MERGED_ROOT at the top of this file.")

    experiment_1_dataset_comparison(watai_raw, merged_raw, min_per_class=min_per_class, max_eval_rows=max_eval_rows)
    exp2_output = experiment_2_feature_reduction(watai_raw, merged_raw, min_per_class=min_per_class,
                                                  max_eval_rows=max_eval_rows)
    experiment_3_srate_drate_ablation(watai_raw, min_per_class=min_per_class, max_eval_rows=max_eval_rows)

    # Experiments 4/5 build on Experiment 2's own data-driven Heavy-top-K
    # selection (not the raw all-features list), so the volume/architecture
    # study runs on the same reduced feature set the pipeline would actually
    # ship with. Prefer watai (46 candidates, richer signal) when available.
    selections = exp2_output["selections"]
    if "watai" in selections:
        volume_raw, volume_name = watai_raw, "watai"
        volume_features = selections["watai"]["heavy"]
    elif "merged" in selections:
        volume_raw, volume_name = merged_raw, "merged"
        volume_features = selections["merged"]["heavy"]
    else:
        volume_raw, volume_name, volume_features = watai_raw, "watai", WATAI_46_FEATURES

    experiment_4_data_volume_scaling(volume_raw, volume_name, volume_features, min_per_class=min_per_class,
                                      max_eval_rows=max_eval_rows)
    experiment_5_three_stage_justification(volume_raw, volume_name, volume_features, min_per_class=min_per_class,
                                            max_eval_rows=max_eval_rows)

    print("\nAll experiments complete.")
    print(f"Master results CSV: {MASTER_RESULTS_CSV.resolve()}")
    print(f"Plots directory:    {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main(sample_frac_watai=0.8, sample_frac_merged=0.8, min_per_class=1000)