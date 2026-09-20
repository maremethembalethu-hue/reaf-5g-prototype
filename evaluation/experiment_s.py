
import json
import statistics
import sys
from pathlib import Path
from datetime import datetime
 
sys.path.insert(0, str(Path(__file__).parent))
from results_engine import compute_joined_rows, accuracy_breakdown  # already in evaluation/
 
EVAL_DIR = Path(__file__).parent
EVIDENCE_DIR = EVAL_DIR.parent / "evidence"
 
TARGET_MS = 500  # this project's stated end-to-end target
 

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
    expected_types = ["pcap", "memory", "processes", "syslogs", "chain_of_custody"]
    incident_ids = set()
    for sub in ("packets", "memory", "processes", "syslogs"):
        d = EVIDENCE_DIR / sub
        if d.exists():
            incident_ids.update(p.stem for p in d.iterdir())
 
    if not incident_ids:
        return {"note": "No evidence/ subfolders found yet — run a replay with at least one attack first.", "incidents": []}
 
    custody_entries = {e.get("flow_id") or e.get("incident_id") for e in _load_jsonl(EVIDENCE_DIR / "chain_of_custody.log")}
 
    rows = []
    for iid in sorted(incident_ids):
        present = {
            "pcap": (EVIDENCE_DIR / "packets" / f"{iid}.pcap").exists(),
            "memory": (EVIDENCE_DIR / "memory" / f"{iid}.json").exists(),
            "processes": (EVIDENCE_DIR / "processes" / f"{iid}.json").exists(),
            "syslogs": (EVIDENCE_DIR / "syslogs" / f"{iid}.json").exists(),
            "chain_of_custody": iid in custody_entries,
        }
        rows.append({"incident": iid, **present,
                      "completeness_pct": round(100 * sum(present.values()) / len(present), 1)})
 
    overall = round(sum(r["completeness_pct"] for r in rows) / len(rows), 1) if rows else 0
    return {"incidents": rows, "overall_completeness_pct": overall}
 
 
 
def experiment_6_forensic_bridge():
    # Bridges ML ground truth with forensic evidence: for every window that
    # SHOULD have triggered evidence, confirm evidence actually exists and is chain-of-custody valid.
    rows = compute_joined_rows(str(EVAL_DIR / "items_log.jsonl"), str(EVAL_DIR / "truth_log.jsonl"))
    custody_entries = {e.get("flow_id") or e.get("incident_id") for e in load_jsonl(EVIDENCE_DIR / "chain_of_custody.log")}
 
    should_have_evidence = [r for r in rows if r["expected_label"].lower() != "benign" and r["family_correct"]]
    with_evidence = sum(1 for r in should_have_evidence
                         if (EVIDENCE_DIR / "packets" / f'{r["replay_id"]}.pcap').exists())
    with_valid_custody = sum(1 for r in should_have_evidence if r["replay_id"] in custody_entries)
 
    n = len(should_have_evidence) or 1
    return {
        "correctly_detected_attacks": len(should_have_evidence),
        "evidence_generated_pct": round(100 * with_evidence / n, 1),
        "custody_valid_pct": round(100 * with_valid_custody / n, 1),
    }
 
 
def experiment_7_reliability(n_runs_expected=None):
    # Detection rate and evidence success rate across every attack replay
    # job recorded in truth_log.jsonl the "don't run it only once" test.
    truth = load_jsonl(EVAL_DIR / "truth_log.jsonl")
    attack_jobs = [j for j in truth if j.get("status") == "completed" and j.get("expected_label", "").lower() != "benign"]
    rows = compute_joined_rows(str(EVAL_DIR / "items_log.jsonl"), str(EVAL_DIR / "truth_log.jsonl"))
    timing = load_jsonl(EVAL_DIR / "timing_log.jsonl")
 
    detected_flow_ids = {r["replay_id"] for r in rows if r["family_correct"] and r["expected_label"].lower() != "benign"}
    evidence_flow_ids = {t["flow_id"] for t in timing}
 
    total = len(attack_jobs)
    detected = len(detected_flow_ids)
    evidence_created = len(evidence_flow_ids & detected_flow_ids)
 
    return {
        "attack_runs": total,
        "detected": detected,
        "evidence_bundles_created": evidence_created,
        "detection_rate_pct": round(100 * detected / total, 1) if total else None,
        "evidence_success_rate_pct": round(100 * evidence_created / detected, 1) if detected else None,
        "avg_acquisition_ms": round(statistics.mean(t["detection_to_evidence_ms"] for t in timing), 2) if timing else None,
    }
 
 
def run_all():
    return {
        "generated_at": datetime.now().isoformat(),
        "experiment_1_latency": experiment_1_latency(),
        "experiment_2_scenarios": experiment_2_scenarios(),
        "experiment_3_completeness": experiment_3_completeness(),
        "experiment_6_forensic_bridge": experiment_6_forensic_bridge(),
        "experiment_7_reliability": experiment_7_reliability(),
    }
 
 
if __name__ == "__main__":
    result = run_all()
    out_path = EVAL_DIR / "experiment_suite_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")