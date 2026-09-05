from collections import Counter

import numpy as np
import pandas as pd

# Copied verbatim from feature_lists.json, the actual output of the retrain
# that added Rate + syn_count/fin_count/rst_count. Exact order preserved —
# never sort or alphabetize these.
HEAVY_FEATURES = [
   "Header_Length",
    "Protocol Type",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "cwr_flag_number",
    "ack_count",
    "HTTP",
    "HTTPS",
    "IAT",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "ICMP",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Number",
    "DNS",
    "Variance"
]

LITE_FEATURES = [
    "Tot size",
    "Protocol Type",
    "fin_flag_number",
    "syn_flag_number",
    "Header_Length",
    "UDP",
    "Min",
    "Max",
    "AVG",
    "Number",
    "Std",
    "TCP"
]

PROTOCOL_TYPE_AGGREGATION = "mode"  # "mode" or "mean"


def aggregate_window(rows):
    # row is a list of 1..WINDOW_SIZE raw per-packet dicts from
    # flow_builder.build_packet_row().

    n = len(rows)
    sizes = [r["Tot size"] for r in rows]
    tss = [r["ts"] for r in rows]

    agg = {k: sum(r[k] for r in rows) / n for k in rows[0] if k not in ("ts", "Protocol Type")}

    protocol_values = [r["Protocol Type"] for r in rows]
    agg["Protocol Type"] = Counter(protocol_values).most_common(1)[0][0]

    agg["ack_count"] = sum(r["ack_count"] for r in rows)
    agg["syn_count"] = sum(r["syn_count"] for r in rows)   # ADDED
    agg["fin_count"] = sum(r["fin_count"] for r in rows)   # ADDED
    agg["rst_count"] = sum(r["rst_count"] for r in rows)   # ADDED

    agg["Tot sum"] = sum(sizes)
    agg["Min"] = min(sizes)
    agg["Max"] = max(sizes)
    agg["AVG"] = sum(sizes) / n
    agg["Std"] = float(np.std(sizes, ddof=1)) if n > 1 else 0.0
    agg["Variance"] = float(np.var(sizes, ddof=1)) if n > 1 else 0.0
    agg["Number"] = n

    duration = max(tss) - min(tss)
    agg["Rate"] = (n / duration) if duration > 0 else 0.0

    return agg


def compute_base_features(window):
    return window["features"]


def vectorize(feats, feature_list):
    # Orders a feature dict into the exact column order a model expects.
    return pd.DataFrame([[feats[f] for f in feature_list]], columns=feature_list)


def extract_heavy(window):
    return vectorize(compute_base_features(window), HEAVY_FEATURES)


def extract_lite(window):
    return vectorize(compute_base_features(window), LITE_FEATURES)