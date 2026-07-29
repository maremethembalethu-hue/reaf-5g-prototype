
# Extracts the 23 feature subset (Heavy Model) and 10 feature subset (Lite Model) from a live Scapy packet.

# Feature order must exactly match what was used during training


import numpy as np
import pandas as pd

EPS = 1e-6

# Must match REAF-5G_AI_Pipeline.ipynb, Section 7, exactly.
HEAVY_FEATURES = [
    "Rate",
    "Tot sum",
    "Number",
    "Tot size",
    "IAT",
    "Header_Length",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Variance",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "rst_count",
    "Protocol Type",
]

LITE_FEATURES = [ "IAT" , "Protocol Type", "Header_Length", "Min", "fin_count", "rst_count", ]


def compute_base_features(flow):
    # Turns a finished flow record into a dict covering every feature name used by either HEAVY_FEATURES or LITE_FEATURES.
    lengths = np.array(flow["lengths"], dtype=np.float64)
    n = len(lengths)
    duration = max(flow["last_time"] - flow["start_time"], 0.0)
    iat = np.array(flow["interarrival"], dtype=np.float64) if flow["interarrival"] else np.array([0.0])
    headers = np.array(flow["header_lengths"], dtype=np.float64) if flow["header_lengths"] else np.array([0.0])

    mean_len = float(lengths.mean()) if n else 0.0
    std_len = float(lengths.std()) if n else 0.0
    var_len = float(lengths.var()) if n else 0.0

    # Lag-3 autocovariance of packet size as a stand-in for "Covariance" between consecutive packets in the flow.
    if n >= 3:
        try:
            covariance = float(np.cov(lengths[:-1], lengths[1:])[0, 1])
            if np.isnan(covariance):
                covariance = 0.0
        except Exception:
            covariance = 0.0
    else:
        covariance = 0.0
    feats = {
        "flow_duration": duration,
        "Number": float(n),
        "Tot sum": float(lengths.sum()) if n else 0.0,
        "Tot size": float(lengths.sum()) if n else 0.0,  # see docstring: same total-bytes value as Tot sum here
        "Min": float(lengths.min()) if n else 0.0,
        "Max": float(lengths.max()) if n else 0.0,
        "AVG": mean_len,
        "Std": std_len,
        "IAT": float(iat.mean()),
        "Header_Length": float(headers.mean()),
        "Rate": n / duration if duration > EPS else float(n) / EPS,
        "Srate": flow["fwd_count"] / duration if duration > EPS else float(flow["fwd_count"]) / EPS,
        "Drate": flow["bwd_count"] / duration if duration > EPS else float(flow["bwd_count"]) / EPS,
        "Magnitude": float(np.sqrt(mean_len)) if mean_len > 0 else 0.0,
        "Radius": std_len,
        "Covariance": covariance,
        "Variance": var_len,
        "Weight": float(n),
        "Protocol Type": float(flow["proto"]),
        "syn_flag_number": float(flow["flags"]["syn"]),
        "rst_flag_number": float(flow["flags"]["rst"]),
        "psh_flag_number": float(flow["flags"]["psh"]),
        "ack_flag_number": float(flow["flags"]["ack"]),
        "fin_count": float(flow["flags"]["fin"]),
        "rst_count": float(flow["flags"]["rst"]),
        "urg_count": float(flow["flags"]["urg"]),
    }
    
    for key, value in feats.items():
        if not np.isfinite(value):
            feats[key] = 0.0
    return feats


def vectorize(feats, feature_list):
    # Orders a feature dict into the exact column order a model expects.
   return pd.DataFrame(
        [[feats.get(f, 0.0) for f in feature_list]],
        columns=feature_list,
        dtype=np.float32
    )


def extract_heavy(flow) :
    return vectorize(compute_base_features(flow), HEAVY_FEATURES)


def extract_lite(flow):
    return vectorize(compute_base_features(flow), LITE_FEATURES)
