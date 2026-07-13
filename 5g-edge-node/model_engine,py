
#Loads ONNX models and runs inference on extracted features. Implements dynamic switching between Heavy and Lite model based on resource_monitor output.

#Label mapping must match the LabelEncoder used during training


import os
import logging
import numpy as np
import onnxruntime as ort

from feature_extractor import extract_heavy, extract_lite
from resource_monitor import get_model_tier, get_metrics

log = logging.getLogger(__name__)

#  Model paths 
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "detection_models")
HEAVY_PATH = os.path.join(MODEL_DIR, "model_full.onnx")
LITE_PATH  = os.path.join(MODEL_DIR, "model_lite.onnx")

#  Label map, match LabelEncoder.classes_ from training

LABEL_MAP = {
    0: "Benign",
    1: "DDoS",
    2: "DoS",
    3: "Mirai",
    4: "Reconnaissance",
    5: "Spoofing",
    6: "Web Attack",
    7: "Brute Force"
}

# Load models once at import time 
_heavy_session = None
_lite_session  = None

def _load_models():
    global _heavy_session, _lite_session
    if os.path.exists(HEAVY_PATH):
        _heavy_session = ort.InferenceSession(HEAVY_PATH)
        log.info(f"Heavy model loaded: {HEAVY_PATH}")
    else:
        log.warning(f"Heavy model not found: {HEAVY_PATH}")

    if os.path.exists(LITE_PATH):
        _lite_session = ort.InferenceSession(LITE_PATH)
        log.info(f"Lite model loaded: {LITE_PATH}")
    else:
        log.warning(f"Lite model not found: {LITE_PATH}")

_load_models()


def classify_packet(packet):
    
    # Run inference on a packet.
    # Returns dict with label, confidence, attack_type, model_used.

    tier    = get_model_tier()
    metrics = get_metrics()

    try:
        if tier == "heavy" and _heavy_session:
            features = extract_heavy(packet)
            session  = _heavy_session
            model_used = "heavy_xgboost"
        else:
            features = extract_lite(packet)
            session  = _lite_session
            model_used = "lite_decision_tree"

        if session is None:
            return _benign_result()

        # Run ONNX inference
        input_name = session.get_inputs()[0].name
        outputs    = session.run(None, {input_name: features})

        # outputs[0] = predicted class index
        # outputs[1] = probability array 
        pred_class = int(outputs[0][0])

        if len(outputs) > 1 and outputs[1] is not None:
            proba      = outputs[1][0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0   # Decision Tree may not output probabilities

        attack_type = LABEL_MAP.get(pred_class, "Unknown")
        is_attack   = attack_type != "Benign"

        label = "[!! ATTACK ]" if is_attack else "[UE-TUNNEL ]"

        return {
            "label":       label,
            "attack_type": attack_type,
            "confidence":  confidence,
            "model_used":  model_used,
            "cpu_percent": metrics["cpu_percent"],
            "ram_percent": metrics["ram_percent"],
            "pred_class":  pred_class
        }

    except Exception as e:
        log.error(f"Inference error: {e}")
        return _benign_result()


def _benign_result():
    return {
        "label":       "[UE-TUNNEL ]",
        "attack_type": "Benign",
        "confidence":  0.0,
        "model_used":  "none",
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "pred_class":  0
    }