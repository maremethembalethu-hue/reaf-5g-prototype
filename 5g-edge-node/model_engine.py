
#Loads ONNX models and runs inference on extracted features. Implements dynamic switching between Heavy and Lite model based on resource_monitor output.

#Label mapping must match the LabelEncoder used during training


import os
import logging
import numpy as np
import onnxruntime as ort
import joblib

from feature_extraction import compute_base_features, vectorize, HEAVY_FEATURES, LITE_FEATURES
from resource_monitor import get_model_tier, get_metrics

log = logging.getLogger(__name__)

MODEL_DIR         = os.path.join(os.path.dirname(__file__), "trained_models/outputs")
HEAVY_MODEL_PATH  = os.path.join(MODEL_DIR, "heavy_xgboost.onnx")
LITE_MODEL_PATH   = os.path.join(MODEL_DIR, "lite_decision_tree.onnx")
HEAVY_SCALER_PATH = os.path.join(MODEL_DIR, "heavy_scaler.pkl")
LITE_SCALER_PATH  = os.path.join(MODEL_DIR, "lite_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

_heavy_session = None
ite_session = None
heavy_scaler = None
lite_scaler = None
_label_encoder = None


def _load_artifacts():
    global _heavy_session, ite_session, heavy_scaler, lite_scaler, _label_encoder

    if os.path.exists(HEAVY_MODEL_PATH):
        _heavy_session = ort.InferenceSession(HEAVY_MODEL_PATH)
        log.info(f"Heavy model loaded: {HEAVY_MODEL_PATH}")
    else:
        log.warning(f"Heavy model not found: {HEAVY_MODEL_PATH}")

    if os.path.exists(LITE_MODEL_PATH):
        ite_session = ort.InferenceSession(LITE_MODEL_PATH)
        log.info(f"Lite model loaded: {LITE_MODEL_PATH}")
    else:
        log.warning(f"Lite model not found: {LITE_MODEL_PATH}")

    if os.path.exists(HEAVY_SCALER_PATH):
        heavy_scaler = joblib.load(HEAVY_SCALER_PATH)
        log.info(f"Heavy scaler loaded: {HEAVY_SCALER_PATH}")
    else:
        log.warning(f"Heavy scaler not found: {HEAVY_SCALER_PATH} — "
                    f"inference will run on unscaled features (likely wrong).")

    if os.path.exists(LITE_SCALER_PATH):
        lite_scaler = joblib.load(LITE_SCALER_PATH)
        log.info(f"Lite scaler loaded: {LITE_SCALER_PATH}")
    else:
        log.warning(f"Lite scaler not found: {LITE_SCALER_PATH} — "
                    f"inference will run on unscaled features (likely wrong).")

    if os.path.exists(LABEL_ENCODER_PATH):
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        log.info(f"Label encoder loaded: {LABEL_ENCODER_PATH} "
                 f"({len(_label_encoder.classes_)} classes)")
    else:
        log.warning(f"Label encoder not found: {LABEL_ENCODER_PATH} — "
                    f"predictions will be reported as raw class indices.")


_load_artifacts()


def _decode_label(pred_class) :
    if _label_encoder is not None:
        return str(_label_encoder.inverse_transform([pred_class])[0])
    return f"class_{pred_class}"


def classify_flow(flow):
    # Run inference on one finished flow record. Returns label, confidence, attack_type, and which model tier was used.
    tier = get_model_tier()
    metrics = get_metrics()
    base_features = compute_base_features(flow)

    try:
        if tier == "heavy" and _heavy_session is not None:
            raw = vectorize(base_features, HEAVY_FEATURES)
            scaled = heavy_scaler.transform(raw) if heavy_scaler is not None else raw
            session = _heavy_session
            model_used = "heavy_xgboost"
        else:
            raw = vectorize(base_features, LITE_FEATURES)
            scaled = lite_scaler.transform(raw) if lite_scaler is not None else raw
            session = ite_session
            model_used = "lite_decision_tree"

        if session is None:
            return _benign_result(metrics)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: scaled.astype(np.float32)})

        pred_class = int(np.asarray(outputs[0]).reshape(-1)[0])
        confidence = 1.0
        if len(outputs) > 1 and outputs[1] is not None:
            try:
                proba = np.asarray(outputs[1])
                # ONNX classifiers may return probabilities as a list-of-dicts or an array; handle both without assuming a specific runtime's output shape.
                if proba.dtype == object:
                    confidence = float(max(proba[0].values()))
                else:
                    confidence = float(proba.reshape(len(proba), -1)[0].max())
            except Exception:
                confidence = 1.0

        attack_type = _decode_label(pred_class)
        is_attack = attack_type.lower() != "benign"
        label = "[!! ATTACK ]" if is_attack else "[UE-TUNNEL ]"

        return {
            "label": label,
            "attack_type": attack_type,
            "confidence": confidence,
            "model_used": model_used,
            "cpu_percent": metrics["cpu_percent"],
            "ram_percent": metrics["ram_percent"],
            "pred_class": pred_class,
        }

    except Exception as e:
        log.error(f"Inference error: {e}")
        return _benign_result(metrics)


def _benign_result(metrics=None):
    metrics = metrics or {"cpu_percent": 0.0, "ram_percent": 0.0}
    return {
        "label": "[UE-TUNNEL ]",
        "attack_type": "Benign",
        "confidence": 0.0,
        "model_used": "none",
        "cpu_percent": metrics["cpu_percent"],
        "ram_percent": metrics["ram_percent"],
        "pred_class": 0,
    }