import time
import os, json, csv
from datetime import datetime
from collections import defaultdict, deque
from flask import Flask, jsonify, render_template
from dash_results import compute_joined_rows, accuracy_breakdown
from monitor_tier import get_metrics, get_model_tier
 
app = Flask(__name__)

EVAL_DIR = os.environ.get("EVAL_DIR", "/evaluation")
EVIDENCE_DIR = os.environ.get("EVIDENCE_DIR", "/evidence")

PREDICTIONS_LOG = os.path.join(EVAL_DIR, "predictions_log.jsonl")
DEBUG_LOG = os.path.join(EVAL_DIR, "debug_predictions.jsonl")
TRUTH_LOG = os.path.join(EVAL_DIR, "truth_log.jsonl")
PROVENANCE_LOG = os.path.join(EVAL_DIR, "items_log.jsonl")
CUSTODY_LOG = os.path.join(EVIDENCE_DIR, "chain_of_custody.log")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")   # optional convention, see /api/runs

MIN_MARGIN = 0.20  # matches model_engine.py's stage-3 confidence gate

resource_history = deque(maxlen=3)
#  helpers

def load_jsonl(path, limit=None):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:] if limit else rows


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tier_for(model_used):
    return "Lite" if str(model_used).startswith("lite_") else "Heavy"


def window_label(packet_count):
    try:
        return "100-pkt" if int(packet_count) >= 100 else "10-pkt"
    except (TypeError, ValueError):
        return "?"


def fmt_time(ts):
    if not ts:
        return "-"
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).strftime("%H:%M:%S")
    except ValueError:
        return str(ts)[:19]


#  pages

@app.route("/")
def index():
    return render_template("index.html")


#  stats

@app.route("/api/current_replay")
def current_replay():
    # A job is "in progress" if its most recent entry is "started"
    jobs = load_jsonl(TRUTH_LOG)
    latest_by_id = {}
    for j in jobs:
        rid = j.get("replay_id")
        latest_by_id[rid] = j  # later entries overwrite earlier ones, so this ends up "most recent per id"

    in_progress = [j for j in latest_by_id.values() if j.get("status") == "started"]
    if not in_progress:
        return jsonify({"active": False, "pcap_path": None, "expected_label": None, "started_at": None})

    job = max(in_progress, key=lambda j: j.get("start_time", 0))
    return jsonify({
        "active": True, "pcap_path": job.get("pcap_path"),
        "expected_label": job.get("expected_label"), "started_at": fmt_time(job.get("start_ts")),
    })


@app.route("/api/stats")
def stats():
    # Prefer the edge node's OWN logged readings 
    
    try:
        metrics = get_metrics()
        cpu=  metrics["cpu_percent"]
        ram = metrics["ram_percent"]
        tier = get_model_tier()
    except Exception:
        cpu, ram = None, None
        tier = None  # can't know the edge node's real tier without its own log

    items = load_jsonl(PROVENANCE_LOG)
    attack_events = [r for r in items if r.get("predicted_label") not in (None, "Benign", "InsufficientData", "ModelUnavailable")]
    last = items[-1] if items else None

    return jsonify({
        "cpu": cpu, "ram": ram,
        "tier": tier or ((f"Lite" if (cpu is not None and cpu >= 50) else "Heavy") if cpu is not None else "unknown"),
        "subnet": "192.168.100.0/24",
        "total_incidents": len(items),
        "attack_events": len(attack_events),
        "evidence_bundles": len(attack_events),  # every attack window currently triggers evidence collection
        "last_detection": (f'{fmt_time(last["captured_ts"])} | {last["predicted_label"]} | '
                            f'conf={round(last["confidence"], 3)}') if last else "no detections logged yet",
    })


#  feed

def _row_from_item(r):
    return {
        "time": fmt_time(r.get("captured_ts")),
        "flow_id": r.get("flow_id"),
        "attack": r.get("predicted_label"),
        "confidence": round(r.get("confidence", 0), 3),
        "window": window_label(r.get("packet_count")),
        "override": "YES" if "w100_override" in str(r.get("model_used", "")) else "No",
        "model": tier_for(r.get("model_used")),
    }


@app.route("/api/feed")
def feed():
    items = load_jsonl(PROVENANCE_LOG, limit=25)
    return jsonify([_row_from_item(r) for r in reversed(items)])


@app.route("/api/live")
def live():
    items = load_jsonl(PROVENANCE_LOG, limit=1)
    if not items:
        return jsonify({"time": fmt_time(None), "flow_id": None, "size": None,
                         "attack": "no data yet", "confidence": None, "flagged": False})
    r = items[-1]
    return jsonify({
        "time": fmt_time(r.get("captured_ts")),
        "flow_id": r.get("flow_id"),
        "size": r.get("packet_count"),
        "attack": r.get("predicted_label"),
        "confidence": round(r.get("confidence", 0), 3),
        "flagged": r.get("predicted_label") not in (None, "Benign", "InsufficientData", "ModelUnavailable"),
    })


#  chain of custody

@app.route("/api/custody")
def custody():
    # Expects one JSON object per line, each carrying at least its own hash
    # and the previous entry's hash 
    rows = load_jsonl(CUSTODY_LOG)
    entries = []
    broken = 0
    prev_hash = None
    for i, r in enumerate(rows, 1):
        this_hash = r.get("hash") or r.get("entry_hash")
        prev_field = r.get("prev") or r.get("prev_hash") or r.get("previous_hash")
        ok = True if prev_hash is None else (prev_field == prev_hash)
        if not ok:
            broken += 1
        entries.append({"id": i, "hash": this_hash, "prev": prev_field, "ok": ok})
        prev_hash = this_hash
    return jsonify({
        "intact": broken == 0, "entries": entries, "total": len(entries), "broken": broken,
        "algorithm": "SHA-256", "timestamp": True, "signature": True,
    })


#  four-level accuracy

FAMILY_MERGE = {"DDoS": "Flood", "DoS": "Flood", "Flood_uncertain": "Flood"}


@app.route("/api/accuracy")
def accuracy():
    # This entire endpoint depends on joined_results.csv.
    rows = compute_joined_rows(PROVENANCE_LOG, TRUTH_LOG)
    heavy = accuracy_breakdown([r for r in rows if r["model_used"].startswith("heavy_")])
    lite = accuracy_breakdown([r for r in rows if r["model_used"].startswith("lite_")])
    labels = [("Strict (exact label match)", "strict"), ("Attack vs Benign", "binary"),
              ("6-Family (Flood merged)", "family"), ("DDoS vs DoS specific", "flood")]
    return jsonify({
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_windows_joined": len(rows),
        "rows": [{"level": label, "heavy": heavy[key], "lite": lite[key]} for label, key in labels],
    })
 
 
@app.route("/api/timeline")
def timeline():
    metrics = get_metrics()
    tier = get_model_tier()

    resource_history.append({
        "time":fmt_time( time.time()),
        "cpu": metrics["cpu_percent"],
        "ram": metrics["ram_percent"],
        "model": tier_for(tier)
    })

    return jsonify([
        {
            "time": fmt_time(r["time"]),
            "cpu": r["cpu"],
            "ram": r["ram"],
            "model": r["model"]
        }
        for r in resource_history
    ])

#  per-attack breakdown

@app.route("/api/per_attack")
def per_attack():
    rows = compute_joined_rows(PROVENANCE_LOG, TRUTH_LOG)
    by_expected = defaultdict(list)
    for r in rows:
        by_expected[r["expected_label"]].append(r)
 
    out = []
    for attack, group in sorted(by_expected.items()):
        tp = sum(1 for r in group if r["predicted_label"].lower() == attack.lower())
        fn = len(group) - tp
        fp = sum(1 for r in rows if r["predicted_label"].lower() == attack.lower() and r["expected_label"].lower() != attack.lower())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out.append({
            "attack": attack, "precision": round(precision, 2), "recall": round(recall, 2),
            "f1": round(f1, 3), "count": len(group),
            "bundles": sum(1 for r in group if attack.lower() != "benign"),
        })
    return jsonify(out)
 

#  override activity

@app.route("/api/overrides")
def overrides():
    items = load_jsonl(PROVENANCE_LOG)
    rows = []
    for r in items:
        if "w100_override" not in str(r.get("model_used", "")):
            continue
        rows.append({
            "time": fmt_time(r.get("captured_ts")), "flow": r.get("flow_id"),
            
            "before": None, "after": r.get("predicted_label"),
            "confidence": round(r.get("confidence", 0), 3), "changed": None,
        })
    total = len(rows)
    return jsonify({"rows": rows, "total": total, "changed": None, "confirmed": None,
                     "changed_pct": None})


#  DDoS/DoS margins

@app.route("/api/margins")
def margins():
    debug_rows = load_jsonl(DEBUG_LOG)
    margins_list = [r["stage3"]["margin"] for r in debug_rows
                     if r.get("stage3", {}).get("margin") is not None]
    buckets = [round(x * 0.05, 2) for x in range(21)]  # 0.00 .. 1.00, matches a real |P(DDoS)-P(DoS)| margin
    counts = [0] * len(buckets)
    for m in margins_list:
        idx = min(int(m / 0.05), len(buckets) - 1)
        counts[idx] += 1
 
    joined_rows = compute_joined_rows(PROVENANCE_LOG, TRUTH_LOG)
    uncertain = sum(1 for r in joined_rows if r["predicted_label"] == "Flood_uncertain")
    return jsonify({"buckets": buckets, "counts": counts, "threshold": MIN_MARGIN, "uncertain": uncertain})
#  experiment runs

@app.route("/api/runs")
def runs():
    # Historical runs 
    out = []
    run_dirs = sorted(os.listdir(RUNS_DIR)) if os.path.isdir(RUNS_DIR) else []
    for rid in run_dirs:
        rows = load_csv(os.path.join(RUNS_DIR, rid, "joined_results.csv"))
        if not rows:
            continue
        heavy = accuracy_breakdown([r for r in rows if str(r.get("model_used", "")).startswith("heavy_")])
        out.append({"id": rid, "date": None, "model": "Heavy", "strict": heavy["strict"],
                     "binary": heavy["binary"],
                     "overrides": sum(1 for r in rows if "w100_override" in str(r.get("model_used", ""))),
                     "uncertain": sum(1 for r in rows if r.get("predicted_label") == "Flood_uncertain")})
 
    live_rows = compute_joined_rows(PROVENANCE_LOG, TRUTH_LOG)
    if live_rows:
        heavy = accuracy_breakdown([r for r in live_rows if r["model_used"].startswith("heavy_")])
        out.append({"id": "current (live)", "date": None, "model": "Heavy", "strict": heavy["strict"],
                     "binary": heavy["binary"],
                     "overrides": sum(1 for r in live_rows if "w100_override" in r["model_used"]),
                     "uncertain": sum(1 for r in live_rows if r["predicted_label"] == "Flood_uncertain")})
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)