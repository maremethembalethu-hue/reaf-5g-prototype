
from collections import Counter
 
import numpy as np
import pandas as pd
 
# Copied from validate_5g_replay_pipeline.py's verified MODEL_SPECS --
# CRITICAL: exact training order, never sorted or alphabetized.
HEAVY_FEATURES = [
    "Rate", "Tot sum", "Number", "Tot size", "IAT", "Header_Length",
    "Min", "Max", "AVG", "Std", "Variance",
    "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "rst_count", "Protocol Type",
]
 
LITE_FEATURES = [
    "IAT", "Protocol Type", "Header_Length", "Min", "fin_count", "rst_count",
]
 
# Feature_extraction.py assigns 'Protocol Type' as the MODE of the raw
# per-packet protocol numbers over the window. Switch to "mean" only if you
# have verified your training data was built that way instead.
PROTOCOL_TYPE_AGGREGATION = "mode"  # "mode" or "mean"
 
 
def aggregate_window(rows):
    
    #row is a list of 1..WINDOW_SIZE raw per-packet dicts from
    # low_builder.build_packet_row().
    
    n = len(rows)
    sizes = [r["Tot size"] for r in rows]
    tss = [r["ts"] for r in rows]
 
    agg = {}
    for key in rows[0].keys():
        if key == "ts":
            continue
        agg[key] = sum(r[key] for r in rows) / n
 
    if PROTOCOL_TYPE_AGGREGATION == "mode":
        proto_counts = Counter(r["Protocol Type"] for r in rows)
        agg["Protocol Type"] = proto_counts.most_common(1)[0][0]
    # else: leave as the mean already computed above
 
    agg["ack_count"] = sum(r["ack_count"] for r in rows)
    agg["syn_count"] = sum(r["syn_count"] for r in rows)
    agg["fin_count"] = sum(r["fin_count"] for r in rows)
    agg["rst_count"] = sum(r["rst_count"] for r in rows)
 
    agg["Tot sum"] = sum(sizes)
    agg["Min"] = min(sizes)
    agg["Max"] = max(sizes)
    agg["AVG"] = sum(sizes) / n
    agg["Std"] = float(np.std(sizes, ddof=1)) if n > 1 else 0.0
    agg["Variance"] = float(np.var(sizes, ddof=1)) if n > 1 else 0.0
    # "Tot size" intentionally left as mean(sizes) (== AVG) -- matches the
    # original extractor, which never re-assigns this column.
 
    agg["Number"] = n
    duration = max(tss) - min(tss)
    agg["Rate"] = (n / duration) if duration > 0 else 0.0
 
    return agg
 
 
def compute_base_features(window):
   
    return window["features"]
 
 
def vectorize(feats, feature_list) :
    # Orders a feature dict into the exact column order a model expects.
    return pd.DataFrame(
        [[feats.get(f, 0.0) for f in feature_list]],
        columns=feature_list
    )
 
 
def extract_heavy(window):
    return vectorize(compute_base_features(window), HEAVY_FEATURES)
 
 
def extract_lite(window) :
    return vectorize(compute_base_features(window), LITE_FEATURES)
 