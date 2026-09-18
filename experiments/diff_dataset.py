import glob
import re
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, skew
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

WATAI_ROOT = Path("data/wataiData/csv/CICIoT2023")
MERGED_ROOT = Path("data/MERGED_CSV/MERGED_CSV")

OUTPUT_DIR = Path(f"outputs/dataset_diff_{RUN_ID}")
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
for _d in (OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def save_csv(df, name):
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[csv] {path}  ({len(df)} rows)")
    return path


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
_m, _w = set(MERGED_39_FEATURES), set(WATAI_46_FEATURES)
COMMON_FEATURES = [f for f in MERGED_39_FEATURES if f in _w]           # 37
MERGED_ONLY_FEATURES = [f for f in MERGED_39_FEATURES if f not in _w]  # Time_To_Live, IGMP
WATAI_ONLY_FEATURES = [f for f in WATAI_46_FEATURES if f not in _m]    # 9 missing-in-39 features
NUMERIC_COMMON_FEATURES = [f for f in COMMON_FEATURES if f != "Protocol Type"]
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
FLOOD_CLASSES = ("DDoS", "DoS")


#  loading 
COLUMN_RENAME = {"Magnitue": "Magnitude"}  # documented CICIoT2023 raw-column typo


def norm_key(name):
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def normalize_columns(df, expected_features):
    df = df.rename(columns=COLUMN_RENAME, copy=False)
    lookup = {norm_key(c): c for c in df.columns}
    rename_map = {lookup[norm_key(f)]: f for f in expected_features
                  if f not in df.columns and norm_key(f) in lookup and lookup[norm_key(f)] != f}
    if rename_map:
        print(f"f renaming columns: {rename_map}")
        df = df.rename(columns=rename_map, copy=False)
    return df


def validate_columns(df, required, name):
    missing = [f for f in required if f not in df.columns]
    if missing:
        print(f"E {name}: missing {missing}")
        print(f"        actual columns: {sorted(df.columns)}")
        raise KeyError(f"{name} missing expected columns: {missing}")


def find_label_col(df):
    for c in ("label", "Label", "LABEL"):
        if c in df.columns:
            return c
    raise KeyError("no label column found")


def downcast(df):
    fcols = df.select_dtypes(include=["float64"]).columns
    if len(fcols):
        df[fcols] = df[fcols].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def load_watai(root=WATAI_ROOT, sample_frac=None, chunksize=500_000):
    files = sorted(glob.glob(str(root / "part-*.csv")))
    if not files:
        print(f"W no part-*.csv under {root}")
        return pd.DataFrame()
    frames = []
    for i, fpath in enumerate(files):
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk.rename(columns=COLUMN_RENAME, copy=False)
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            frames.append(downcast(chunk))
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  watai: {i + 1}/{len(files)} files")
    full = pd.concat(frames, ignore_index=True)
    full = normalize_columns(full, WATAI_46_FEATURES)
    validate_columns(full, WATAI_46_FEATURES, "watai")
    lc = find_label_col(full)
    if lc != "label":
        full = full.rename(columns={lc: "label"}, copy=False)
    print(f"watai: {len(full):,} rows, ~{full.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return full


def load_merged(root=MERGED_ROOT, start=1, end=63, sample_frac=None, chunksize=500_000):
    files = [f for f in (root / f"Merged{i:02d}.csv" for i in range(start, end + 1)) if f.exists()]
    if not files:
        print(f"W no Merged*.csv under {root}")
        return pd.DataFrame()
    frames = []
    for i, fpath in enumerate(files):
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk.rename(columns=COLUMN_RENAME, copy=False)
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            frames.append(downcast(chunk))
        print(f"  merged: {i + 1}/{len(files)} files ({fpath.name})")
    full = pd.concat(frames, ignore_index=True)
    full = normalize_columns(full, MERGED_39_FEATURES)
    validate_columns(full, MERGED_39_FEATURES, "merged")
    lc = find_label_col(full)
    if lc != "label":
        full = full.rename(columns={lc: "label"}, copy=False)
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


def add_family_column(df, label_col="label"):
    nm = {k.upper(): v for k, v in ATTACK_FAMILY_MAP.items()}
    df["raw_label"] = df[label_col]
    mapped = df[label_col].astype(str).str.upper().map(nm)
    df["family"] = mapped.fillna(df[label_col])
    return df


def stratified_cap(df, label_col, max_rows, rs=RANDOM_STATE):
    if max_rows is None or len(df) <= max_rows:
        return df
    sampled, _ = train_test_split(df, train_size=max_rows, stratify=df[label_col], random_state=rs)
    return sampled.reset_index(drop=True)


#  section 1/2: overview & subtype counts 
def dataset_overview(watai, merged):
    rows = []
    for name, df in (("watai", watai), ("merged", merged)):
        counts = df["family"].value_counts()
        total = len(df)
        for fam, n in counts.items():
            rows.append({"dataset": name, "family": fam, "n_rows": int(n), "proportion": n / total})
        rows.append({"dataset": name, "family": "TOTAL", "n_rows": total, "proportion": 1.0})
    overview = pd.DataFrame(rows)
    save_csv(overview, "01_dataset_overview")

    subtype_rows = []
    for name, df in (("watai", watai), ("merged", merged)):
        flood = df[df["family"].isin(FLOOD_CLASSES)]
        counts = flood["raw_label"].value_counts()
        for subtype, n in counts.items():
            subtype_rows.append({"dataset": name, "raw_subtype": subtype,
                                  "mapped_family": ATTACK_FAMILY_MAP.get(subtype, "UNKNOWN"), "n_rows": int(n)})
    subtype_df = pd.DataFrame(subtype_rows)
    save_csv(subtype_df, "02_ddos_dos_subtype_counts")
    return overview, subtype_df


#  section 3: duplicates & zero fractions 
def duplicate_and_zero_diagnostics(watai_raw, merged_raw):
    rows = []
    for name, df in (("watai", watai_raw), ("merged", merged_raw)):
        dup_frac = 1 - (len(df.drop_duplicates()) / len(df))
        rows.append({"dataset": name, "metric": "duplicate_row_fraction", "value": dup_frac})
        rows.append({"dataset": name, "metric": "total_rows", "value": len(df)})
    for name, df, feats in (("watai", watai_raw, WATAI_46_FEATURES), ("merged", merged_raw, MERGED_39_FEATURES)):
        num_feats = [f for f in feats if f != "Protocol Type"]
        for f in num_feats:
            zero_frac = float((df[f] == 0).mean())
            rows.append({"dataset": name, "metric": f"zero_fraction__{f}", "value": zero_frac})
    save_csv(pd.DataFrame(rows), "03_duplicate_and_zero_diagnostics")


#  section 4: feature distribution comparison 
def feature_distribution_comparison(watai_df, merged_df, features, scope_name):
    rows = []
    for f in features:
        w, m = watai_df[f].astype(float).values, merged_df[f].astype(float).values
        w_std, m_std = w.std(), m.std()
        pooled_std = np.sqrt((w_std ** 2 + m_std ** 2) / 2) or np.nan
        smd = (w.mean() - m.mean()) / pooled_std if pooled_std else np.nan
        try:
            ks_stat, ks_p = ks_2samp(w, m)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        rows.append({
            "feature": f, "scope": scope_name,
            "watai_mean": w.mean(), "merged_mean": m.mean(),
            "watai_std": w_std, "merged_std": m_std,
            "watai_median": np.median(w), "merged_median": np.median(m),
            "watai_min": w.min(), "merged_min": m.min(),
            "watai_max": w.max(), "merged_max": m.max(),
            "watai_skew": skew(w), "merged_skew": skew(m),
            "standardized_mean_diff": smd, "ks_statistic": ks_stat, "ks_pvalue": ks_p,
        })
    df = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)
    save_csv(df, f"04a_feature_distribution_{scope_name}" if scope_name == "all_rows"
              else f"04b_feature_distribution_{scope_name}")

    top5 = df.head(5)["feature"].tolist()
    for f in top5:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(watai_df[f].astype(float), bins=60, alpha=0.5, density=True, label="watai")
        ax.hist(merged_df[f].astype(float), bins=60, alpha=0.5, density=True, label="merged")
        ax.set_title(f"{f} distribution ({scope_name}) — largest KS gap")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, f"{scope_name}__{f}__histogram")
    return df


def protocol_type_comparison(watai_df, merged_df):
    w_counts = watai_df["Protocol Type"].astype(str).value_counts(normalize=True)
    m_counts = merged_df["Protocol Type"].astype(str).value_counts(normalize=True)
    all_vals = sorted(set(w_counts.index) | set(m_counts.index))
    rows = [{"protocol_value": v, "watai_proportion": w_counts.get(v, 0.0),
             "merged_proportion": m_counts.get(v, 0.0)} for v in all_vals]
    try:
        w_raw = watai_df["Protocol Type"].astype(str).value_counts()
        m_raw = merged_df["Protocol Type"].astype(str).value_counts()
        table = pd.DataFrame({"watai": w_raw, "merged": m_raw}).fillna(0)
        chi2, p, _, _ = chi2_contingency(table.values)
    except Exception:
        chi2, p = np.nan, np.nan
    df = pd.DataFrame(rows)
    df["chi2_statistic"] = chi2
    df["chi2_pvalue"] = p
    save_csv(df, "protocol_type_comparison")
    return df


#  section 5/6: univariate AUC 
def _auc(values, y_binary):
    try:
        auc = roc_auc_score(y_binary, values)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan


def univariate_auc_ddos_vs_dos(watai_flood, merged_flood, features):
    rows = []
    for name, df in (("watai", watai_flood), ("merged", merged_flood)):
        pass
    wy = (watai_flood["family"] == "DDoS").astype(int).values
    my = (merged_flood["family"] == "DDoS").astype(int).values
    for f in features:
        w_auc = _auc(watai_flood[f].astype(float).values, wy)
        m_auc = _auc(merged_flood[f].astype(float).values, my)
        rows.append({"feature": f, "watai_auc": w_auc, "merged_auc": m_auc, "delta": w_auc - m_auc})
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    save_csv(df, "univariate_auc_ddos_vs_dos")

    top = pd.concat([df.head(8), df.tail(4)]).drop_duplicates("feature")
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    ax.barh(top["feature"][::-1], top["watai_auc"][::-1], height=0.4, label="watai", align="edge")
    ax.barh(top["feature"][::-1], top["merged_auc"][::-1], height=-0.4, label="merged", align="edge")
    ax.set_xlabel("single-feature AUC for DDoS vs DoS"); ax.set_title("Univariate separability, per dataset")
    ax.legend()
    plt.tight_layout()
    save_plot(fig, "univariate_auc_ddos_vs_dos__top_deltas")
    return df


def univariate_auc_overall(watai_df, merged_df, features):
    classes = sorted(set(watai_df["family"].unique()) & set(merged_df["family"].unique()))
    rows = []
    for f in features:
        w_aucs, m_aucs = [], []
        for cls in classes:
            wy = (watai_df["family"] == cls).astype(int).values
            my = (merged_df["family"] == cls).astype(int).values
            w_aucs.append(_auc(watai_df[f].astype(float).values, wy))
            m_aucs.append(_auc(merged_df[f].astype(float).values, my))
        rows.append({"feature": f, "watai_avg_ovr_auc": np.nanmean(w_aucs),
                     "merged_avg_ovr_auc": np.nanmean(m_aucs),
                     "delta": np.nanmean(w_aucs) - np.nanmean(m_aucs)})
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    save_csv(df, "univariate_auc_overall")
    return df


#  section 7: class separability 
def class_separability(df, features, dataset_name):
    scaler = StandardScaler().fit(df[features])
    scaled = scaler.transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns=features)
    scaled_df["family"] = df["family"].values

    classes = sorted(scaled_df["family"].unique())
    centroids = {c: scaled_df.loc[scaled_df["family"] == c, features].mean().values for c in classes}
    spreads = {c: scaled_df.loc[scaled_df["family"] == c, features].std().mean() for c in classes}

    rows = []
    for a, b in combinations(classes, 2):
        dist = float(np.linalg.norm(centroids[a] - centroids[b]))
        avg_spread = (spreads[a] + spreads[b]) / 2
        rows.append({"dataset": dataset_name, "class_a": a, "class_b": b,
                     "centroid_distance": dist, "avg_intra_class_spread": avg_spread,
                     "separation_ratio": dist / avg_spread if avg_spread else np.nan})
    return pd.DataFrame(rows)


def class_separability_report(watai_df, merged_df, features):
    w = class_separability(watai_df, features, "watai")
    m = class_separability(merged_df, features, "merged")
    combined = pd.concat([w, m], ignore_index=True)
    save_csv(combined, "07_class_separability")

    ddos_dos = combined[((combined["class_a"] == "DDoS") & (combined["class_b"] == "DoS")) |
                         ((combined["class_a"] == "DoS") & (combined["class_b"] == "DDoS"))]
    print("DDoS-DoS separation ratio (higher = easier to tell apart):")
    print(ddos_dos[["dataset", "separation_ratio"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(ddos_dos["dataset"], ddos_dos["separation_ratio"])
    ax.set_ylabel("centroid distance / avg intra-class spread")
    ax.set_title("DDoS vs DoS structural separability (shared 37 features)")
    plt.tight_layout()
    save_plot(fig, "ddos_dos_separation_ratio")
    return combined, ddos_dos


#  section 8: correlation structure 
def correlation_structure_diff(watai_df, merged_df, features):
    cw = watai_df[features].corr()
    cm = merged_df[features].corr()
    diff = (cw - cm).abs()
    frob = float(np.sqrt((diff.values ** 2).sum()))

    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            pairs.append({"feature_a": a, "feature_b": b, "watai_corr": cw.loc[a, b],
                          "merged_corr": cm.loc[a, b], "abs_diff": diff.loc[a, b]})
    pairs_df = pd.DataFrame(pairs).sort_values("abs_diff", ascending=False)
    pairs_df["frobenius_norm_full_matrix"] = frob
    save_csv(pairs_df.head(40), "08_correlation_structure_diff_top_pairs")
    print(f"Correlation matrix Frobenius-norm difference (whole 37x37): {frob:.3f}")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(diff.values, cmap="Reds", vmin=0, vmax=diff.values.max())
    ax.set_xticks(range(len(features))); ax.set_xticklabels(features, rotation=90, fontsize=6)
    ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=6)
    ax.set_title("|corr(watai) - corr(merged)| — shared 37 features")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    save_plot(fig, "correlation_structure_diff_heatmap")
    return pairs_df, frob


#  section 9: merged-only columns diagnostics 
def merged_only_columns_diagnostics(merged_df, features=MERGED_ONLY_FEATURES):
    classes = sorted(merged_df["family"].unique())
    rows = []
    for f in features:
        vals = merged_df[f].astype(float)
        ovr_aucs = [_auc(vals.values, (merged_df["family"] == c).astype(int).values) for c in classes]
        rows.append({
            "feature": f, "mean": vals.mean(), "std": vals.std(), "n_unique": vals.nunique(),
            "zero_fraction": float((vals == 0).mean()), "avg_ovr_auc_vs_family": float(np.nanmean(ovr_aucs)),
            "is_near_constant": bool(vals.std() < 1e-6),
        })
    df = pd.DataFrame(rows)
    save_csv(df, "merged_only_columns_diagnostics")
    return df


#  section 10: summary 
def build_summary(dup_zero_path, dist_ddos_df, auc_ddos_df, sep_ratio_df, corr_frob, merged_only_df):
    rows = []
    max_auc_delta = auc_ddos_df["delta"].abs().max()
    n_big_auc_drop = int((auc_ddos_df["delta"] > 0.10).sum())
    rows.append({"metric": "max univariate AUC delta| (DDoS vs DoS, shared features)",
                  "value": max_auc_delta,
                  "flag": "LIKELY CONTRIBUTOR" if max_auc_delta > 0.15 else "unlikely on its own"})
    rows.append({"metric": "# shared features with AUC(watai)-AUC(merged) > 0.10 for DDoS/DoS",
                  "value": n_big_auc_drop,
                  "flag": "LIKELY CONTRIBUTOR" if n_big_auc_drop > 0 else "none found"})

    ddos_dos = sep_ratio_df[((sep_ratio_df["class_a"] == "DDoS") & (sep_ratio_df["class_b"] == "DoS")) |
                             ((sep_ratio_df["class_a"] == "DoS") & (sep_ratio_df["class_b"] == "DDoS"))]
    w_ratio = ddos_dos.loc[ddos_dos["dataset"] == "watai", "separation_ratio"].values
    m_ratio = ddos_dos.loc[ddos_dos["dataset"] == "merged", "separation_ratio"].values
    if len(w_ratio) and len(m_ratio):
        ratio_drop = float(w_ratio[0] - m_ratio[0])
        rows.append({"metric": "DDoS-DoS separation ratio drop (watai - merged)", "value": ratio_drop,
                      "flag": "LIKELY CONTRIBUTOR" if ratio_drop > 0.3 else "unlikely on its own"})

    rows.append({"metric": "correlation matrix Frobenius-norm difference (37x37 shared features)",
                  "value": corr_frob, "flag": "LIKELY CONTRIBUTOR" if corr_frob > 3.0 else "unlikely on its own"})

    for _, r in merged_only_df.iterrows():
        rows.append({"metric": f"merged-only column '{r['feature']}' near-constant?", "value": r["is_near_constant"],
                      "flag": "noise, not informative" if r["is_near_constant"] else "carries some signal"})

    ks_top = dist_ddos_df.sort_values("ks_statistic", ascending=False).head(3)
    for _, r in ks_top.iterrows():
        rows.append({"metric": f"largest distribution shift (KS stat) in shared features, DDoS/DoS rows: '{r['feature']}'",
                      "value": r["ks_statistic"], "flag": "LIKELY CONTRIBUTOR" if r["ks_statistic"] > 0.3 else "moderate"})

    df = pd.DataFrame(rows)
    save_csv(df, "investigation_summary")
    print("\n SUMMARY ")
    print(df.to_string(index=False))
    return df


#  main 
def main(sample_frac_watai=0.5, sample_frac_merged=0.5, max_rows_for_stats=200_000):
    print(f"Run ID: {RUN_ID} | output: {OUTPUT_DIR.resolve()}")
    watai_raw = load_watai(sample_frac=sample_frac_watai)
    merged_raw = load_merged(sample_frac=sample_frac_merged)
    if not len(watai_raw) or not len(merged_raw):
        raise RuntimeError("both datasets are required for this investigation")

    duplicate_and_zero_diagnostics(watai_raw, merged_raw)

    watai = add_family_column(clean(watai_raw))
    merged = add_family_column(clean(merged_raw))
    dataset_overview(watai, merged)

    watai_s = stratified_cap(watai, "family", max_rows_for_stats)
    merged_s = stratified_cap(merged, "family", max_rows_for_stats)

    dist_all = feature_distribution_comparison(watai_s, merged_s, NUMERIC_COMMON_FEATURES, "all_rows")
    protocol_type_comparison(watai_s, merged_s)

    watai_flood = watai_s[watai_s["family"].isin(FLOOD_CLASSES)]
    merged_flood = merged_s[merged_s["family"].isin(FLOOD_CLASSES)]
    dist_ddos = feature_distribution_comparison(watai_flood, merged_flood, NUMERIC_COMMON_FEATURES, "ddos_dos_rows")

    auc_ddos = univariate_auc_ddos_vs_dos(watai_flood, merged_flood, NUMERIC_COMMON_FEATURES)
    univariate_auc_overall(watai_s, merged_s, NUMERIC_COMMON_FEATURES)

    _, sep_ratio_df = class_separability_report(watai_s, merged_s, NUMERIC_COMMON_FEATURES)
    _, corr_frob = correlation_structure_diff(watai_s, merged_s, NUMERIC_COMMON_FEATURES)
    merged_only_df = merged_only_columns_diagnostics(merged_s)

    build_summary(None, dist_ddos, auc_ddos, sep_ratio_df, corr_frob, merged_only_df)
    print(f"\nDone. Results: {RESULTS_DIR.resolve()}")
    print(f"Plots: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main(sample_frac_watai=0.5, sample_frac_merged=0.5, max_rows_for_stats=200_000)