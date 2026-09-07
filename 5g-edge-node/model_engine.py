
import os
import json
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
 
from feature_extraction import compute_base_features, vectorize, HEAVY_FEATURES, LITE_FEATURES
from resource_monitor import get_model_tier, get_metrics
 
log = logging.getLogger(__name__)
 

DEBUG_LOG_PATH = os.getenv("DEBUG_LOG_PATH", os.path.join(
    os.getenv("EVAL_DIR", "/evaluation"), "debug_predictions.jsonl"))
MAX_DEBUG_RECORDS = int(os.getenv("MAX_DEBUG_RECORDS", "300"))
_debug_record_count = 0
_debug_cap_notice_shown = False
 
 
def _write_debug_record(record):
    global _debug_record_count, _debug_cap_notice_shown
    if _debug_record_count >= MAX_DEBUG_RECORDS:
        if not _debug_cap_notice_shown:
            log.info(f"DEBUG: reached MAX_DEBUG_RECORDS={MAX_DEBUG_RECORDS}, "
                     f"no further records written to {DEBUG_LOG_PATH}")
            _debug_cap_notice_shown = True
        return
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
        _debug_record_count += 1
    except Exception as e:
        log.error(f"DEBUG: failed to write debug record: {e}")
 

MODEL_DIR = os.getenv("MODEL_ARTIFACTS_DIR",
                       os.path.join(os.path.dirname(__file__), "trained_models/three_stage_all"))
METADATA_PATH = os.path.join(MODEL_DIR, "feature_lists.json")
 
HEAVY_SCALER_PATH = os.path.join(MODEL_DIR, "heavy_scaler.pkl")
LITE_SCALER_PATH = os.path.join(MODEL_DIR, "lite_scaler.pkl")
STAGE2_ENCODER_PATH = os.path.join(MODEL_DIR, "stage2_encoder.pkl")
STAGE3_ENCODER_PATH = os.path.join(MODEL_DIR, "stage3_encoder.pkl")
PROTO_ENCODER_PATH = os.path.join(MODEL_DIR, "protocol_encoder.joblib")
 
STAGE1_THRESHOLD = {"heavy": 0.5, "lite": 0.5}
MIN_MARGIN = 0.20

_stage2_override = {"heavy": None, "lite": None}  # {"family","confidence","stage3_label",
                                                    #  "stage3_confidence","valid_upto_position"}
 
_sessions = {"heavy": {}, "lite": {}}
_scalers = {"heavy": None, "lite": None}
_features = {"heavy": None, "lite": None}
_stage2_encoder = None
_stage3_encoder = None
_proto_encoder = None
_metadata = None
 
 
def _load_session(path, label):
    if path and os.path.exists(path):
        sess = ort.InferenceSession(path)
        log.info(f"{label} loaded: {path}")
        return sess
    log.warning(f"{label} not found: {path}")
    return None
 
 
def _load_artifacts():
    global _stage2_encoder, _stage3_encoder, _proto_encoder, _metadata
 
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    else:
        parent = os.path.dirname(MODEL_DIR)
        try:
            siblings = os.listdir(parent) if os.path.isdir(parent) else []
        except Exception:
            siblings = []
        log.critical(
            f"feature_lists.json not found at: {METADATA_PATH}\n"
            f"  MODEL_DIR does not contain training artifacts: {MODEL_DIR}\n"
            f"  Contents of {parent}: {siblings}\n"
            f"  Set MODEL_ARTIFACTS_DIR to the correct path, or copy the "
            f"training run's output files directly into {MODEL_DIR}."
        )
        _metadata = {}
 
    onnx_files = _metadata.get("onnx_files", {})
    for tier in ("heavy", "lite"):
        for stage in ("stage1", "stage2", "stage3"):
            fname = onnx_files.get(f"{tier}_{stage}")
            path = os.path.join(MODEL_DIR, fname) if fname else None
            _sessions[tier][stage] = _load_session(path, f"{tier} {stage}")
 
    _features["heavy"] = _metadata.get("heavy_features")
    _features["lite"] = _metadata.get("lite_features")
    if _features["heavy"] and _features["heavy"] != HEAVY_FEATURES:
        log.warning("feature_lists.json heavy_features differs from feature_extraction.HEAVY_FEATURES")
    if _features["lite"] and _features["lite"] != LITE_FEATURES:
        log.warning("feature_lists.json lite_features differs from feature_extraction.LITE_FEATURES")
    _features["heavy"] = _features["heavy"] or HEAVY_FEATURES
    _features["lite"] = _features["lite"] or LITE_FEATURES
 
    if os.path.exists(HEAVY_SCALER_PATH):
        _scalers["heavy"] = joblib.load(HEAVY_SCALER_PATH)
    else:
        log.warning(f"Heavy scaler not found: {HEAVY_SCALER_PATH}")
    if os.path.exists(LITE_SCALER_PATH):
        _scalers["lite"] = joblib.load(LITE_SCALER_PATH)
    else:
        log.warning(f"Lite scaler not found: {LITE_SCALER_PATH}")
 
    if os.path.exists(STAGE2_ENCODER_PATH):
        _stage2_encoder = joblib.load(STAGE2_ENCODER_PATH)
        log.info(f"Stage-2 classes: {list(_stage2_encoder.classes_)}")
    else:
        log.warning(f"Stage-2 encoder not found: {STAGE2_ENCODER_PATH}")
 
    if os.path.exists(STAGE3_ENCODER_PATH):
        _stage3_encoder = joblib.load(STAGE3_ENCODER_PATH)
        log.info(f"Stage-3 classes: {list(_stage3_encoder.classes_)}")
    else:
        log.warning(f"Stage-3 encoder not found: {STAGE3_ENCODER_PATH}")
 
    if os.path.exists(PROTO_ENCODER_PATH):
        _proto_encoder = joblib.load(PROTO_ENCODER_PATH)
        log.info(f"Protocol encoder loaded ({len(_proto_encoder.classes_)} classes: "
                 f"{list(_proto_encoder.classes_)})")
    else:
        log.warning(f"Protocol encoder not found: {PROTO_ENCODER_PATH}.")
 
 
_load_artifacts()
 
BENIGN_LABELS = {"benign"}
REQUIRE_MODELS_ON_START = os.getenv("REQUIRE_MODELS_ON_START", "0") == "1"
DEBUG_FEATURES = os.getenv("DEBUG_FEATURES", "0") == "1"
 
 
def _check_artifacts_or_fail():
    problems = []
    for tier in ("heavy", "lite"):
        for stage in ("stage1", "stage2", "stage3"):
            if _sessions[tier].get(stage) is None:
                problems.append(f"{tier}/{stage} ONNX session missing")
        if _scalers[tier] is None:
            problems.append(f"{tier} scaler missing")
    if _stage2_encoder is None:
        problems.append("stage2 encoder missing")
    if _stage3_encoder is None:
        problems.append("stage3 encoder missing")
    if _proto_encoder is None:
        problems.append("protocol encoder missing")
 
    heavy_ready = all(_sessions["heavy"].values()) and _scalers["heavy"] is not None
    lite_ready = all(_sessions["lite"].values()) and _scalers["lite"] is not None
    if not heavy_ready and not lite_ready:
        problems.append("NEITHER tier fully loaded — every window will be "
                         "reported as ModelUnavailable.")
 
    if problems:
        msg = "Model artifacts incomplete:\n  - " + "\n  - ".join(problems)
        if REQUIRE_MODELS_ON_START:
            raise RuntimeError(msg)
        log.critical(msg)
 
 
_check_artifacts_or_fail()
 
 
def _onnx_predict_proba(session, X_row):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_row.astype(np.float32)})
    proba_raw = outputs[1] if len(outputs) > 1 else None
    if proba_raw is None:
        label = int(np.asarray(outputs[0]).reshape(-1)[0])
        return np.array([1.0]), np.array([label])
    proba_arr = np.asarray(proba_raw)
    if proba_arr.dtype == object:
        row0 = proba_arr[0]
        classes = sorted(row0.keys())
        proba = np.array([row0[c] for c in classes], dtype=np.float64)
        return proba, np.array(classes)
    proba = proba_arr.reshape(-1).astype(np.float64)
    return proba, np.arange(len(proba))
 
 
def encode_protocol_type(raw_value, proto_encoder, log=None):
    key = str(int(raw_value))
    classes = set(proto_encoder.classes_)
    if key in classes:
        return int(proto_encoder.transform([key])[0])
    if log is not None:
        log.warning(f"Protocol number {raw_value!r} (as {key!r}) not in "
                     f"proto_encoder.classes_ ({sorted(classes)}); falling back to class 0.")
    return 0
 
 
def _scale_window(window, tier):
    # Shared by both the w=10 and w=100 paths: encode protocol, vectorize,
    # scale. Both window sizes use the SAME feature list and SAME scaler —
    # the scaler was fit on training data that already mixes w=10 and w=100
    # rows (that's just how the classes were originally captured), so it's
    # already correctly calibrated for either scale without any change.
    base_features = dict(compute_base_features(window))
    if _proto_encoder is not None:
        base_features["Protocol Type"] = encode_protocol_type(
            base_features["Protocol Type"], _proto_encoder, log=log)
    else:
        log.warning("No protocol encoder loaded; Protocol Type left raw.")
    features = _features.get(tier)
    scaler = _scalers.get(tier)
    raw = vectorize(base_features, features)
    scaled = scaler.transform(raw).astype(np.float32)
    return base_features, features, scaled
 
 
def _run_stage2_stage3(sessions, scaled):
    # Returns (family, confidence, stage3_label_or_None, stage3_confidence_or_None, debug_dict)
    p2, p2_classes = _onnx_predict_proba(sessions["stage2"], scaled)
    stage2_idx = int(np.argmax(p2))
    family = str(_stage2_encoder.inverse_transform([p2_classes[stage2_idx]])[0])
    family_conf = float(p2[stage2_idx])
    debug = {"stage2": {"predicted_family": family, "confidence": family_conf}}
 
    stage3_label, stage3_conf = None, None
    if family == "Flood":
        p3, p3_classes = _onnx_predict_proba(sessions["stage3"], scaled)
        class_names = _stage3_encoder.inverse_transform(p3_classes)
        ddos_pos = list(class_names).index("DDoS")
        dos_pos = list(class_names).index("DoS")
        ddos_proba, dos_proba = float(p3[ddos_pos]), float(p3[dos_pos])
        margin = abs(ddos_proba - dos_proba)
        stage3_idx = int(np.argmax(p3))
        stage3_label = str(class_names[stage3_idx]) if margin >= MIN_MARGIN else "Flood_uncertain"
        stage3_conf = float(p3[stage3_idx])
        debug["stage3"] = {
            "margin": margin, "confident": bool(margin >= MIN_MARGIN),
            "predicted_label": stage3_label,  # was missing — couldn't tell DDoS from DoS
                                                # in past debug dumps, only whether the call
                                                # was "confident" or not
            "P(DDoS)": ddos_proba, "P(DoS)": dos_proba,
        }
 
    return family, family_conf, stage3_label, stage3_conf, debug
 
 
def register_w100_result(window_100, position):
    # w=100 windows never gate on their own — they only ever EXIST to
    # override the w=10 stage2 answer for Flood/Mirai (see model_engine.
    # register_w100_result). Still logged/acted on like a normal detection
    # event so it's auditable, tagged distinctly via model_used.
    tier = get_model_tier()
    metrics = get_metrics()
    sessions = _sessions.get(tier)
    if sessions is None or any(v is None for v in sessions.values()) or _scalers.get(tier) is None:
        return _unavailable_result(metrics)
 
    try:
        base_features, features, scaled = _scale_window(window_100, tier)
        family, family_conf, stage3_label, stage3_conf, dbg = _run_stage2_stage3(sessions, scaled)
 
        debug_record = None
        if DEBUG_FEATURES:
            debug_record = {
                "captured_ts": datetime.now(timezone.utc).isoformat(),
                "tier": tier, "path": "w100",
                "window": {"packet_count": window_100.get("packet_count"), "position": position,
                           "flow_id": window_100.get("flow_id"), "mixed_flow": window_100.get("mixed_flow")},
                "raw_features": {k: base_features.get(k) for k in features},
                "scaled_features": {k: float(v) for k, v in zip(features, scaled.reshape(-1).tolist())},
            }
            debug_record.update(dbg)
 
        if family in ("Flood", "Mirai"):
            _stage2_override[tier] = {
                "family": family, "confidence": family_conf,
                "stage3_label": stage3_label, "stage3_confidence": stage3_conf,
                "valid_upto_position": position,
                "flow_id": window_100.get("flow_id"),  # scopes the override to THIS replay
                                                         # job only — position alone isn't
                                                         # enough, since it never resets
                                                         # between different jobs and a stale
                                                         # override could otherwise bleed into
                                                         # the start of the next, unrelated job.
            }
            attack_type = stage3_label if family == "Flood" else family
            confidence = stage3_conf if family == "Flood" else family_conf
        else:
            attack_type, confidence = family, family_conf
 
        result = {
            "label": "[!! ATTACK ]", "attack_type": attack_type, "confidence": confidence,
            "model_used": f"{tier}_stage2_w100", "cpu_percent": metrics["cpu_percent"],
            "ram_percent": metrics["ram_percent"], "pred_class": attack_type,
        }
        if debug_record is not None:
            debug_record["final"] = result
            _write_debug_record(debug_record)
        return result
    except Exception as e:
        log.error(f"w100 inference error: {e}")
        return _unavailable_result(metrics)
 
 
def classify_flow(window, position=None):
    tier = get_model_tier()
    metrics = get_metrics()
 
    try:
        sessions = _sessions.get(tier)
        threshold = STAGE1_THRESHOLD.get(tier, 0.5)
        if sessions is None or _scalers.get(tier) is None or _features.get(tier) is None \
                or any(v is None for v in sessions.values()):
            return _unavailable_result(metrics)
 
        base_features, features, scaled = _scale_window(window, tier)
 
        debug_record = None
        if DEBUG_FEATURES:
            debug_record = {
                "captured_ts": datetime.now(timezone.utc).isoformat(),
                "tier": tier,
                "window": {
                    "packet_count": window.get("packet_count"),
                    "start_time": window.get("start_time"),
                    "last_time": window.get("last_time"),
                    "duration": (window.get("last_time", 0) - window.get("start_time", 0)),
                    "true_chronological_span": window.get("true_chronological_span"),
                    "out_of_order_packets": window.get("out_of_order_packets"),
                    "close_reason": window.get("close_reason"),
                    "flow_id": window.get("flow_id"),
                    "mixed_flow": window.get("mixed_flow"),
                    "position": position,
                },
                "raw_features": {k: base_features.get(k) for k in features},
                "scaled_features": {k: float(v) for k, v in zip(features, scaled.reshape(-1).tolist())},
            }
 
        p1, p1_classes = _onnx_predict_proba(sessions["stage1"], scaled)
        p1_attack = float(p1[list(p1_classes).index(1)]) if 1 in p1_classes else float(p1[-1])
 
        if debug_record is not None:
            debug_record["stage1"] = {
                "proba_attack": p1_attack, "threshold": threshold,
                "passed_gate": bool(p1_attack >= threshold),
            }
 
        if p1_attack < threshold:
            result = {
                "label": "[UE-TUNNEL ]", "attack_type": "Benign", "confidence": 1.0 - p1_attack,
                "model_used": f"{tier}_stage1", "cpu_percent": metrics["cpu_percent"],
                "ram_percent": metrics["ram_percent"], "pred_class": "Benign",
            }
            if debug_record is not None:
                debug_record["final"] = result
                _write_debug_record(debug_record)
            return result
 
        family, family_conf, stage3_label, stage3_conf, dbg = _run_stage2_stage3(sessions, scaled)
        attack_type, confidence = family, family_conf
        model_used = f"{tier}_stage2"
        if family == "Flood":
            attack_type, confidence = stage3_label, stage3_conf
            model_used = f"{tier}_stage3"
        if debug_record is not None:
            debug_record.update(dbg)
 
       
        override = _stage2_override.get(tier)
        if override is not None and position is not None:
            valid_upto = override["valid_upto_position"]
            same_flow = (override.get("flow_id") is not None
                         and override["flow_id"] == window.get("flow_id"))
            if same_flow and valid_upto < position <= valid_upto + 100:
                family = override["family"]
                if family == "Flood":
                    attack_type, confidence = override["stage3_label"], override["stage3_confidence"]
                else:
                    attack_type, confidence = family, override["confidence"]
                model_used = f"{tier}_stage2_w100_override"
                if debug_record is not None:
                    debug_record["w100_override_applied"] = override
 
        result = {
            "label": "[!! ATTACK ]", "attack_type": attack_type, "confidence": confidence,
            "model_used": model_used, "cpu_percent": metrics["cpu_percent"],
            "ram_percent": metrics["ram_percent"], "pred_class": attack_type,
        }
        if debug_record is not None:
            debug_record["final"] = result
            _write_debug_record(debug_record)
        return result
 
    except Exception as e:
        log.error(f"Inference error: {e}")
        return _unavailable_result(metrics)
 
 
def _unavailable_result(metrics=None):
    metrics = metrics or {"cpu_percent": 0.0, "ram_percent": 0.0}
    return {
        "label": "[?? NO MODEL ]", "attack_type": "ModelUnavailable", "confidence": 0.0,
        "model_used": "none", "cpu_percent": metrics["cpu_percent"],
        "ram_percent": metrics["ram_percent"], "pred_class": "ModelUnavailable",
    }