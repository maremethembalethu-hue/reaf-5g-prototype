
import json
import statistics
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))
from results_engine import compute_joined_rows, accuracy_breakdown  # already in evaluation/
 
EVAL_DIR = Path(__file__).parent
EVIDENCE_DIR = EVAL_DIR.parent / "evidence"
 
TARGET_MS = 500  # this project's stated end-to-end target
COOLDOWN_SECONDS = float(os.environ.get("TRIGGER_COOLDOWN_SECONDS", "30"))

def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out
 
 
def experiment_1_latency():
    # Detection time, evidence acquisition time, end-to-end time the
    # 500ms-target table Chapter 4 asks for directly.
    timing = load_jsonl(EVAL_DIR / "timing_log.jsonl")
    if not timing:
        return {"note": "timing_log.jsonl is empty or missing apply the "
                         "capture.py patch and re-run a replay first.", "runs": []}
 
    runs = []
    for t in timing:
        acq_ms = t["detection_to_evidence_ms"]
        total_ms = (t["evidence_complete_ts"] - t["detection_ts"]) * 1000
        runs.append({"flow_id": t["flow_id"], "attack_type": t["attack_type"],
                      "evidence_acquisition_ms": round(acq_ms, 2), "total_ms": round(total_ms, 2)})
 
    totals = [r["total_ms"] for r in runs]
    return {
        "runs": runs,
        "mean_ms": round(statistics.mean(totals), 2),
        "median_ms": round(statistics.median(totals), 2),
        "min_ms": round(min(totals), 2),
        "max_ms": round(max(totals), 2),
        "stdev_ms": round(statistics.stdev(totals), 2) if len(totals) > 1 else 0.0,
        "pct_under_target": round(100 * sum(1 for x in totals if x <= TARGET_MS) / len(totals), 1),
        "target_ms": TARGET_MS,
    }
 
 
def experiment_2_scenarios():
    # Per-attack-family detection outcome one row per scenario Chapter 4
    # names (Benign, DDoS/Flood, Spoofing/MITM, Recon/scan), built from
    # whatever replay jobs truth_log.jsonl actually recorded.
    rows = compute_joined_rows(str(EVAL_DIR / "items_log.jsonl"), str(EVAL_DIR / "truth_log.jsonl"))
    by_expected = {}
    for r in rows:
        by_expected.setdefault(r["expected_label"], []).append(r)
 
    out = []
    for label, group in sorted(by_expected.items()):
        detected = sum(1 for r in group if (r["expected_label"].lower() == "benign")
                        == (r["predicted_label"].lower() == "benign") and r["predicted_label"].lower() != "benign")
        if label.lower() == "benign":
            # for Benign, "success" means NOT raising an incident
            correct = sum(1 for r in group if r["predicted_label"].lower() == "benign")
            out.append({"scenario": label, "windows": len(group),
                        "correctly_left_benign": correct,
                        "false_positive_rate_pct": round(100 * (len(group) - correct) / len(group), 1)})
        else:
            correct = sum(1 for r in group if r["family_correct"])
            out.append({"scenario": label, "windows": len(group),
                        "correctly_flagged": correct,
                        "detection_rate_pct": round(100 * correct / len(group), 1)})
    return out
 

def experiment_3_completeness():
   
    expected_files = {
        "pcap": "network_capture.pcap", "processes": "processes.json",
        "memory": "memory.json", "syslog": "syslog.txt", "metadata": "metadata.json",
    }
    incident_dirs = sorted(EVIDENCE_DIR.glob("incident_*"))
    if not incident_dirs:
        return {"note": "No evidence/incident_* directories found yet — run a replay with at least one attack first.", "incidents": []}
 
    custody_by_id = {e.get("incident_id"): e for e in load_jsonl(EVIDENCE_DIR / "chain_of_custody.log")}
 
    rows = []
    for d in incident_dirs:
        incident_id = d.name.removeprefix("incident_")
        present = {key: (d / fname).exists() for key, fname in expected_files.items()}
        present["chain_of_custody"] = incident_id in custody_by_id
        rows.append({"incident": incident_id, **present,
                      "completeness_pct": round(100 * sum(present.values()) / len(present), 1)})
 
    overall = round(sum(r["completeness_pct"] for r in rows) / len(rows), 1) if rows else 0
    return {"incidents": rows, "overall_completeness_pct": overall}
 

def _parse_ts(ts_str):
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
 
 
 
def _group_into_bursts(rows):
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("predicted_label")].append(r)   # CHANGED: was r.get("flow_id")
 
    bursts = []
    for attack_type, type_rows in by_type.items():
        type_rows_sorted = sorted(type_rows, key=lambda r: r["captured_ts"])
        current_burst, prev_ts = [], None
        for r in type_rows_sorted:
            ts = _parse_ts(r["captured_ts"])
            if prev_ts is not None and (ts - prev_ts).total_seconds() > COOLDOWN_SECONDS:
                bursts.append(current_burst)
                current_burst = []
            current_burst.append(r)
            prev_ts = ts
        if current_burst:
            bursts.append(current_burst)
    return bursts
 
 
def experiment_6_burst_level():
    rows = compute_joined_rows(str(EVAL_DIR / "items_log.jsonl"), str(EVAL_DIR / "truth_log.jsonl"))
    should_have_evidence = [r for r in rows if r["expected_label"].lower() != "benign" and r["family_correct"]]
    custody_by_id = {e["incident_id"] for e in load_jsonl(EVIDENCE_DIR / "chain_of_custody.log")}
 
    bursts = _group_into_bursts(should_have_evidence)
    bursts_with_evidence = sum(
        1 for burst in bursts if any(r.get("incident_id") in custody_by_id for r in burst)
    )
    n_bursts = len(bursts) or 1
 
    return {
        "cooldown_seconds_used": COOLDOWN_SECONDS,
        "total_bursts": len(bursts),
        "bursts_with_at_least_one_incident": bursts_with_evidence,
        "burst_level_evidence_pct": round(100 * bursts_with_evidence / n_bursts, 1),
        "note": ("cooldown_seconds_used must match trigger.py's real cooldown value -- "
                 "set TRIGGER_COOLDOWN_SECONDS if the default (5.0s) is wrong for this deployment."),
    }
 
def experiment_7_reliability():
    truth = load_jsonl(EVAL_DIR / "truth_log.jsonl")
    attack_jobs = [j for j in truth if j.get("status") == "completed" and j.get("expected_label", "").lower() != "benign"]
    rows = compute_joined_rows(str(EVAL_DIR / "items_log.jsonl"), str(EVAL_DIR / "truth_log.jsonl"))
    timing = load_jsonl(EVAL_DIR / "timing_log.jsonl")
 
    detected_replay_ids = {r["replay_id"] for r in rows if r["family_correct"] and r["expected_label"].lower() != "benign"}
    total = len(attack_jobs)
 
    # NEW: real per-window evidence success rate, via incident_id
    custody_by_id = {e["incident_id"] for e in load_jsonl(EVIDENCE_DIR / "chain_of_custody.log")}
    windows_with_evidence = sum(1 for t in timing if t.get("incident_id") in custody_by_id)
    windows_needing_evidence = len(timing)  # every timing_log.jsonl row already means acquisition was triggered
 
    return {
        "attack_runs": total,
        "detected": len(detected_replay_ids),
        "detection_rate_pct": round(100 * len(detected_replay_ids) / total, 1) if total else None,
        "evidence_success_rate_pct": round(100 * windows_with_evidence / windows_needing_evidence, 1) if windows_needing_evidence else None,  # RESTORED, now real
        "avg_acquisition_ms": round(statistics.mean(t["detection_to_evidence_ms"] for t in timing), 2) if timing else None,
    }
 
def run_all():
    return {
        "generated_at": datetime.now().isoformat(),
        "experiment_1_latency": experiment_1_latency(),
        "experiment_2_scenarios": experiment_2_scenarios(),
        "experiment_3_completeness": experiment_3_completeness(),
        "experiment_6_burst_level": experiment_6_burst_level(),
        "experiment_7_reliability": experiment_7_reliability(),
    }
 
 
if __name__ == "__main__":
    result = run_all()
    out_path = EVAL_DIR / "experiment_suite_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")