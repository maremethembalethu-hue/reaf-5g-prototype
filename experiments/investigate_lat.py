
import glob
import re
import warnings
from datetime import datetime
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
 
WATAI_ROOT = Path("data/wataiData/csv/CICIoT2023")
MERGED_ROOT = Path("data/MERGED_CSV/MERGED_CSV")
OUTPUT_DIR = Path(f"outputs/iat_forensics_{RUN_ID}")
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
for _d in (OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
 
 
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
 
 
def norm_key(name):
    return re.sub(r"\s+", " ", str(name).strip()).lower()
 
 
def normalize_columns(df, expected):
    df = df.rename(columns=COLUMN_RENAME, copy=False)
    lookup = {norm_key(c): c for c in df.columns}
    rename_map = {lookup[norm_key(f)]: f for f in expected
                  if f not in df.columns and norm_key(f) in lookup and lookup[norm_key(f)] != f}
    return df.rename(columns=rename_map, copy=False) if rename_map else df
 
 
def find_label_col(df):
    for c in ("label", "Label", "LABEL"):
        if c in df.columns:
            return c
    raise KeyError("no label column")
 
 
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
        return pd.DataFrame()
    frames = []
    for i, fpath in enumerate(files):
        fname = Path(fpath).name
        for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk.rename(columns=COLUMN_RENAME, copy=False)
            if sample_frac is not None:
                chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)
            chunk = downcast(chunk)
            chunk["source_file"] = fname
            frames.append(chunk)
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  watai: {i + 1}/{len(files)} files")
    full = normalize_columns(pd.concat(frames, ignore_index=True), WATAI_46_FEATURES)
    lc = find_label_col(full)
    if lc != "label":
        full = full.rename(columns={lc: "label"}, copy=False)
    print(f"watai: {len(full):,} rows")
    return full
 
 
def load_merged(root=MERGED_ROOT, start=1, end=63, sample_frac=None, chunksize=500_000):
    files = [f for f in (root / f"Merged{i:02d}.csv" for i in range(start, end + 1)) if f.exists()]
    if not files:
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
    full = normalize_columns(pd.concat(frames, ignore_index=True), MERGED_39_FEATURES)
    lc = find_label_col(full)
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
    df["family"] = df["label"].astype(str).str.upper().map(nm).fillna(df["label"])
    return df
 
 
# 1 physical consistency 
def physical_consistency(watai, merged):
    rows = []
    derived = watai["IAT"] * (watai["Number"] - 1)
    for target in ("Duration", "flow_duration"):
        c = np.corrcoef(derived, watai[target])[0, 1]
        ratio = (derived / watai[target].replace(0, np.nan)).median()
        rows.append({"dataset": "watai", "check": f"IAT*(Number-1) vs {target}",
                     "correlation": c, "median_ratio": ratio})
 
    for name, df in (("watai", watai), ("merged", merged)):
        v = df[["IAT", "Rate"]].replace([np.inf, -np.inf], np.nan).dropna()
        v = v[v["Rate"] > 0]
        inv_rate = 1 / v["Rate"]
        c = np.corrcoef(v["IAT"], inv_rate)[0, 1]
        c_log = np.corrcoef(np.log1p(v["IAT"]), np.log1p(inv_rate))[0, 1]
        rows.append({"dataset": name, "check": "IAT vs 1/Rate", "correlation": c, "log_correlation": c_log,
                     "n": len(v)})
 
    df = pd.DataFrame(rows)
    save_csv(df, "physical_consistency")
    print(df.to_string(index=False))
 
    fig, ax = plt.subplots(figsize=(6, 6))
    s = watai.sample(n=min(20000, len(watai)), random_state=RANDOM_STATE)
    ax.scatter(s["Duration"], s["IAT"] * (s["Number"] - 1), s=2, alpha=0.3)
    ax.set_xlabel("Duration"); ax.set_ylabel("IAT * (Number-1)")
    ax.set_title("watai: does IAT*(Number-1) track Duration?")
    plt.tight_layout()
    save_plot(fig, "watai__derived_duration_vs_duration_scatter")
    return df
 
 
#  2 quantization 
def quantization(watai, merged):
    rows = []
    for name, df in (("watai", watai), ("merged", merged)):
        vals = df["IAT"].dropna()
        u = np.sort(vals.unique())
        diffs = np.diff(u)
        top_share = vals.value_counts(normalize=True).iloc[0]
        rows.append({"dataset": name, "n_rows": len(vals), "n_unique": len(u),
                     "frac_unique": len(u) / len(vals), "median_gap_between_unique_values": np.median(diffs),
                     "top_value_share": top_share, "top_value": vals.mode().iloc[0]})
    df = pd.DataFrame(rows)
    save_csv(df, "quantization")
    print(df.to_string(index=False))
    return df
 
 
#  3 class gap 
def class_gap(watai, merged):
    rows = []
    for name, df in (("watai", watai), ("merged", merged)):
        a = df.loc[df["family"] == "DDoS", "IAT"]
        b = df.loc[df["family"] == "DoS", "IAT"]
        lo, hi = (a, b) if a.mean() < b.mean() else (b, a)
        gap = hi.min() - lo.max()
        overlap_frac = ((lo > hi.min()).mean() + (hi < lo.max()).mean())
        rows.append({"dataset": name, "ddos_mean": a.mean(), "ddos_std": a.std(),
                     "dos_mean": b.mean(), "dos_std": b.std(), "gap_between_ranges": gap,
                     "ranges_overlap": bool(gap < 0), "cross_over_row_fraction": overlap_frac})
    df = pd.DataFrame(rows)
    save_csv(df, "03_class_gap")
    print(df.to_string(index=False))
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, d) in zip(axes, (("watai", watai), ("merged", merged))):
        ax.hist(d.loc[d["family"] == "DDoS", "IAT"], bins=80, alpha=0.5, density=True, label="DDoS")
        ax.hist(d.loc[d["family"] == "DoS", "IAT"], bins=80, alpha=0.5, density=True, label="DoS")
        ax.set_title(name); ax.legend()
    plt.tight_layout()
    save_plot(fig, "iat_by_class_histograms")
    return df
 
 
#  4 row order 
def row_order(watai):
    df = watai.copy()
    df["_pos"] = df.groupby("source_file").cumcount()
    c = df[["IAT", "_pos"]].corr().iloc[0, 1]
    per_file_trend = df.groupby("source_file").apply(
        lambda g: np.corrcoef(g["_pos"], g["IAT"])[0, 1] if g["_pos"].nunique() > 1 else np.nan)
    out = pd.DataFrame([{"dataset": "watai", "corr_IAT_vs_within_file_row_order": c,
                          "mean_per_file_corr": per_file_trend.mean(),
                          "std_per_file_corr": per_file_trend.std()}])
    save_csv(out, "row_order")
    print(out.to_string(index=False))
    return out
 
 
def main(sample_frac_watai=0.5, sample_frac_merged=0.5):
    print(f"Run ID: {RUN_ID} | output: {OUTPUT_DIR.resolve()}")
    watai_raw = load_watai(sample_frac=sample_frac_watai)
    merged_raw = load_merged(sample_frac=sample_frac_merged)
    if not len(watai_raw) or not len(merged_raw):
        raise RuntimeError("both datasets required")
 
    watai = add_family(clean(watai_raw))
    merged = add_family(clean(merged_raw))
    watai_flood = watai[watai["family"].isin(FLOOD_CLASSES)].reset_index(drop=True)
    merged_flood = merged[merged["family"].isin(FLOOD_CLASSES)].reset_index(drop=True)
 
    physical_consistency(watai_flood, merged_flood)
    quantization(watai_flood, merged_flood)
    class_gap(watai_flood, merged_flood)
    row_order(watai_flood)
 
    print(f"\nDone. Results: {RESULTS_DIR.resolve()}")
    print(f"Plots: {PLOTS_DIR.resolve()}")
 
 
if __name__ == "__main__":
    main(sample_frac_watai=0.5, sample_frac_merged=0.5)