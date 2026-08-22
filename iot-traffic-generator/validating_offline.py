

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
import time

from scapy.all import IP, TCP, UDP, ARP, fragment, RawPcapReader, Ether
from scapy.layers.inet import defragment
from scapy.layers.inet6 import IPv6




SAMPLE_PCAP = Path(
    "pre-selected/replay_pcaps_contiguous/Benign_Final/"
    "BenignTraffic_contiguous_sample.pcap"
)

EXPECTED_LABEL = "Benign_Final"

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = CODE_DIR.parent / "5g-edge-node" / "trained_models/variation"

HEAVY_MODEL_PATH = MODEL_DIR / "heavy_robust_xgboost_1.onnx"
LITE_MODEL_PATH = MODEL_DIR / "lite_robust_decision_tree_1.onnx"

HEAVY_SCALER_PATH = MODEL_DIR / "heavy_robust_scaler.pkl"
LITE_SCALER_PATH = MODEL_DIR / "lite_robust_scaler.pkl"

LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder_robust.pkl"

# Must match replay_pcap.py exactly
ORIGINAL_UE = "192.168.137.175"
UE_IP = "192.168.100.3"          # EDIT: real UE tunnel IP used at replay time
TARGET_IP = "192.168.100.1"

FRAGMENT_SIZE = 1300
ETHERNET_HEADER_LEN = 14

# How long an incomplete fragment train is kept around before being
# dropped.
REASSEMBLY_TIMEOUT = 5.0

# Must match the original training extractor's n_rows = 10.
WINDOW_SIZE = 10

# Linktype -> decoder, since RawPcapReader hands back raw bytes and we
# have to know how to parse the link layer ourselves.
_LINKTYPE_DECODERS = {1: Ether, 101: IP, 228: IP, 229: IPv6}



# FEATURES USED BY YOUR DEPLOYED MODELS
# Confirmed against the actual trained model/scaler artifacts.

HEAVY_FEATURES = [
    "Rate",
    "Tot sum",
    "Number",
    "Tot size",
    "IAT",
    "Header_Length",
    "Min",
    "Max",
    "Variance",
    "Protocol Type",
]

LITE_FEATURES = [
    "IAT",
    "Protocol Type",
    "Header_Length",
    "Min",
]


 
# FRAGMENT REASSEMBLY


class FragmentReassembler:
   

    def __init__(self, timeout=5.0):
        self._buffers = {}   # key -> {"frags": [...], "first_seen": ts}
        self.timeout = timeout

    def _key(self, ip):
        return (ip.src, ip.dst, ip.proto, ip.id)

    def feed(self, pkt, ts=None):
        if IP not in pkt:
            return pkt

        ip = pkt[IP]
        more_fragments = bool(ip.flags & 0x1)   # MF bit
        frag_offset = ip.frag

        if not more_fragments and frag_offset == 0:
            return pkt  # ordinary packet, nothing to reassemble

        ts = ts if ts is not None else time.time()
        key = self._key(ip)
        entry = self._buffers.setdefault(key, {"frags": [], "first_seen": ts})
        entry["frags"].append(pkt)

        if more_fragments:
            return None  # still waiting on the rest of the fragment train

        frags = self._buffers.pop(key)["frags"]
        try:
            reassembled = defragment(frags)
        except Exception:
            return None

        return reassembled[0] if len(reassembled) == 1 else None

    def expire_stale(self, now=None):
        # Drops fragment trains whose final fragment never arrived so the
        # buffer can't grow unbounded (mirrors a real receiver giving up).
        now = now if now is not None else time.time()
        for key in list(self._buffers.keys()):
            if now - self._buffers[key]["first_seen"] > self.timeout:
                del self._buffers[key]


# 
# REWRITE
# 

def rewrite_packet(pkt, ue_ip=UE_IP, target_ip=TARGET_IP, original_ue=ORIGINAL_UE):
    if IP not in pkt:
        return None

    ip_pkt = pkt[IP].copy()

    if ip_pkt.src == original_ue:
        ip_pkt.src, ip_pkt.dst = ue_ip, target_ip
    elif ip_pkt.dst == original_ue:
        ip_pkt.src, ip_pkt.dst = target_ip, ue_ip
    else:
        ip_pkt.src, ip_pkt.dst = ue_ip, target_ip

    if hasattr(ip_pkt, "len"):
        del ip_pkt.len
    if hasattr(ip_pkt, "chksum"):
        del ip_pkt.chksum
    if TCP in ip_pkt and hasattr(ip_pkt[TCP], "chksum"):
        del ip_pkt[TCP].chksum
    if UDP in ip_pkt and hasattr(ip_pkt[UDP], "chksum"):
        del ip_pkt[UDP].chksum

    ip_pkt.time = pkt.time
    return ip_pkt


def rewrite_and_fragment(pkt, fragment_size=FRAGMENT_SIZE):
    rewritten = rewrite_packet(pkt)
    if rewritten is None:
        return

    t = pkt.time
    if len(bytes(rewritten)) > fragment_size:
        for frag in fragment(rewritten, fragsize=fragment_size):
            frag.time = t
            yield frag
    else:
        yield rewritten


# 
# PACKET SIZE
#
# `wire_size()` needs to match whatever the training extractor's
# `ethernet_frame_size = len(buf)` actually captured. For a raw packet
# read with an Ethernet decoder, len(pkt) already includes the 14-byte
# Ethernet header. rewrite_packet() builds an L3-only IP packet, so the
# rewritten condition needs the 14 bytes added back for a fair,
# apples-to-apples "Tot size" comparison against the raw condition.
#
# If SAMPLE_PCAP's linktype isn't Ethernet (see _LINKTYPE_DECODERS),
# or if your training buf was never really a full Ethernet frame,
# re-verify this against the actual training extractor before trusting
# Tot size numbers.
# 

def wire_size(pkt, apply_rewrite):
    size = len(pkt)
    if apply_rewrite:
        size += ETHERNET_HEADER_LEN
    return size


# 
# TCP FLAGS
# 

def get_flag_values(tcp):
    flags = int(tcp.flags)
    return [
        int((flags & 0x01) != 0),  # FIN
        int((flags & 0x02) != 0),  # SYN
        int((flags & 0x04) != 0),  # RST
        int((flags & 0x08) != 0),  # PSH
        int((flags & 0x10) != 0),  # ACK
        int((flags & 0x20) != 0),  # URG
        int((flags & 0x40) != 0),  # ECE
        int((flags & 0x80) != 0),  # CWR
    ]



# PACKET ROW


def build_packet_row(pkt, ts, last_pac_time, apply_rewrite):
    has_ip = IP in pkt
    has_arp = (not has_ip) and (ARP in pkt)

    if not has_ip and not has_arp:
        return None, last_pac_time

    iat = 0.0 if last_pac_time is None else max(0.0, float(ts) - float(last_pac_time))

    row = {
        "ts": ts,
        "Header_Length": 0.0,
        # Protocol Type stays 0 for ARP-only packets, same as the
        # original (proto_type is never set outside the IP branch).
        "Protocol Type": int(pkt[IP].proto) if has_ip else 0,
        "Rate": 0.0,
        "fin_flag_number": 0,
        "syn_flag_number": 0,
        "rst_flag_number": 0,
        "psh_flag_number": 0,
        "ack_flag_number": 0,
        "ece_flag_number": 0,
        "cwr_flag_number": 0,
        "ack_count": 0,
        "syn_count": 0,
        "fin_count": 0,
        "rst_count": 0,
        "Tot size": wire_size(pkt, apply_rewrite),
        "IAT": iat,
        "Number": 1,
    }

    if not has_ip:
        # ARP-only packet: everything else stays at the defaults above.
        return row, ts

    ip = pkt[IP]

    if ip.proto == 17 and UDP in pkt:
        row["Header_Length"] = 8.0

    elif ip.proto == 6 and TCP in pkt:
        tcp = pkt[TCP]
        dataofs = tcp.dataofs if tcp.dataofs else 5
        row["Header_Length"] = int(dataofs) * 4

        flag_values = get_flag_values(tcp)
        row["fin_flag_number"] = flag_values[0]
        row["syn_flag_number"] = flag_values[1]
        row["rst_flag_number"] = flag_values[2]
        row["psh_flag_number"] = flag_values[3]
        row["ack_flag_number"] = flag_values[4]
        row["ece_flag_number"] = flag_values[6]
        row["cwr_flag_number"] = flag_values[7]

        # Per-packet, NOT cumulative across the pcap -- see the note
        # above build_packet_row(). Summed across the window later in
        # aggregate_window().
        row["ack_count"] = row["ack_flag_number"]
        row["syn_count"] = row["syn_flag_number"]
        row["fin_count"] = row["fin_flag_number"]
        row["rst_count"] = row["rst_flag_number"]

    return row, ts


# 
# AGGREGATION
# 

def aggregate_window(rows):
    n = len(rows)
    if n == 0:
        return None

    sizes = [r["Tot size"] for r in rows]
    tss = [r["ts"] for r in rows]

    # Original extractor
    agg = {k: sum(r[k] for r in rows) / n for k in rows[0] if k != "ts"}

    # Protocol Type = mode 
    protocol_values = [r["Protocol Type"] for r in rows]
    agg["Protocol Type"] = Counter(protocol_values).most_common(1)[0][0]

    # Flag counts are sums of ea
    for c in ("ack_count", "syn_count", "fin_count", "rst_count"):
        agg[c] = sum(r[c] for r in rows)

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


 
# BUILD WINDOWS


def build_windows(pcap_path, apply_rewrite, window_size=WINDOW_SIZE):
    buffer = []
    last_ts = None

    reassembler = FragmentReassembler(timeout=REASSEMBLY_TIMEOUT) if apply_rewrite else None

    with RawPcapReader(str(pcap_path)) as reader:
        decoder = _LINKTYPE_DECODERS.get(reader.linktype, Ether)

        for raw_bytes, pkt_meta in reader:
            raw_pkt = decoder(raw_bytes)

            
            if IP not in raw_pkt and ARP not in raw_pkt:
                continue
            raw_pkt.time = pkt_meta.sec + pkt_meta.usec / 1e6

            
            pkts = list(rewrite_and_fragment(raw_pkt)) if apply_rewrite else [raw_pkt]

            for pkt in pkts:
                t = float(pkt.time)

                if apply_rewrite:
                    whole = reassembler.feed(pkt, ts=t)
                    if whole is None:
                        continue  # still waiting on the rest of this fragment train
                    pkt = whole
                    t = float(pkt.time)
                    reassembler.expire_stale(now=t)

                row, last_ts = build_packet_row(pkt, t, last_ts, apply_rewrite)
                if row is None:
                    continue

                buffer.append(row)
                if len(buffer) >= window_size:
                    feats = aggregate_window(buffer)
                    if feats is not None:
                        yield feats
                    buffer = []

    # Original extractor also processes the final partial window.
    if buffer:
        feats = aggregate_window(buffer)
        if feats is not None:
            yield feats


 
# FEATURE VALIDATION / VECTORIZATION


def vectorize(feats, feature_list):
    missing = [f for f in feature_list if f not in feats]
    if missing:
        raise ValueError(
            "\nFEATURE EXTRACTION ERROR\n"
            "The following model features were not produced:\n"
            f"{missing}\n\n"
            "missing feature can produce a completely meaningless, "
            "over-confident prediction."
        )

    return pd.DataFrame([[feats[f] for f in feature_list]], columns=feature_list)


 
# MODEL INFERENCE


def classify(feats, session, scaler, feature_list, label_encoder):
    raw = vectorize(feats, feature_list)

    if scaler is not None:
        if hasattr(scaler, "feature_names_in_"):
            scaler_features = list(scaler.feature_names_in_)
            if scaler_features != list(feature_list):
                raise ValueError(
                    "\nSCALER FEATURE ORDER MISMATCH\n"
                    f"Expected by script: {feature_list}\n"
                    f"Stored in scaler:   {scaler_features}"
                )
        scaled = scaler.transform(raw)
    else:
        scaled = raw.to_numpy()

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: scaled.astype(np.float32)})
    pred_class = int(np.asarray(outputs[0]).reshape(-1)[0])

    if label_encoder is not None:
        return str(label_encoder.inverse_transform([pred_class])[0])
    return f"class_{pred_class}"


def run_condition(pcap_path, apply_rewrite, session, scaler, feature_list, label_encoder, expected):
    preds = []
    for feats in build_windows(pcap_path, apply_rewrite):
        preds.append(classify(feats, session, scaler, feature_list, label_encoder))

    n = len(preds)
    correct = sum(1 for p in preds if p == expected)
    dist = Counter(preds)

    return {
        "n_windows": n,
        "correct": correct,
        "recall": correct / n if n else 0.0,
        "prediction_distribution": dict(dist),
    }



# MODEL RUNNER

def run_model(name, model_path, scaler_path, feature_list):
    print()
    print("=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)
    print(f"Model:  {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Features ({len(feature_list)}):")
    for i, f in enumerate(feature_list):
        print(f"  {i:02d}: {f}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(str(model_path))
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    label_encoder = joblib.load(LABEL_ENCODER_PATH) if LABEL_ENCODER_PATH.exists() else None

    model_shape = session.get_inputs()[0].shape
    print(f"ONNX input shape: {model_shape}")
    if (
        len(model_shape) >= 2
        and isinstance(model_shape[1], int)
        and model_shape[1] != len(feature_list)
    ):
        raise ValueError(
            "\nMODEL FEATURE COUNT MISMATCH\n"
            f"Model expects: {model_shape[1]}\n"
            f"Script supplies: {len(feature_list)}"
        )

    if scaler is not None and hasattr(scaler, "n_features_in_"):
        print(f"Scaler expects: {scaler.n_features_in_}")
        if scaler.n_features_in_ != len(feature_list):
            raise ValueError(
                "\nSCALER FEATURE COUNT MISMATCH\n"
                f"Scaler expects: {scaler.n_features_in_}\n"
                f"Script supplies: {len(feature_list)}"
            )

    if label_encoder is not None:
        print("Labels:", list(label_encoder.classes_))
        if EXPECTED_LABEL not in label_encoder.classes_:
            print(
                f"WARNING: EXPECTED_LABEL '{EXPECTED_LABEL}' is not in "
                f"label_encoder.classes_ -- recall will always read as 0."
            )

    for condition, apply_rewrite in [("raw", False), ("rewritten", True)]:
        result = run_condition(
            SAMPLE_PCAP,
            apply_rewrite,
            session,
            scaler,
            feature_list,
            label_encoder,
            EXPECTED_LABEL,
        )
        print()
        print(f"=== {name} / {condition} ===")
        print(json.dumps(result, indent=2))


# MAIN

if __name__ == "__main__":
    # Heavy and lite are run through separate calls, each with its own
   
    run_model("HEAVY", HEAVY_MODEL_PATH, HEAVY_SCALER_PATH, HEAVY_FEATURES)
    run_model("LITE", LITE_MODEL_PATH, LITE_SCALER_PATH, LITE_FEATURES)