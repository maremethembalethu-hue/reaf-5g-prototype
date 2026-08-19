
import json
import csv
from pathlib import Path
from collections import defaultdict

DEFAULT_PROVENANCE = "items_log.jsonl"
DEFAULT_GROUND_TRUTH = "truth_log.jsonl"
DEFAULT_OUT = "joined_results.csv"
DEFAULT_RECALL_OUT = "flow_recall_summary.csv"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_job_index(jobs):
   
    by_flow = {}
    for job in jobs:
        flow_id = job.get("flow_id")
        if flow_id is None:
            continue
        if flow_id not in by_flow or job.get("status") == "completed":
            by_flow[flow_id] = job
    return by_flow


def join(predictions_path=DEFAULT_PROVENANCE, ground_truth_path=DEFAULT_GROUND_TRUTH,
         out_csv=DEFAULT_OUT, recall_out=DEFAULT_RECALL_OUT):
    predictions = load_jsonl(predictions_path)
    jobs = load_jsonl(ground_truth_path)
    job_by_flow = build_job_index(jobs)

    fieldnames = ["replay_id", "pcap_path", "expected_label", "predicted_label",
                  "correct", "confidence", "model_used", "packet_count",
                  "packet_ids", "mixed_flow", "captured_ts"]
    rows = []
    unmatched = 0
    mixed = 0
    packets_seen_by_flow = defaultdict(set)

    for pred in predictions:
        flow_id = pred.get("flow_id")
        if pred.get("mixed_flow"):
            mixed += 1

        job = job_by_flow.get(flow_id) if flow_id is not None else None
        if job is None:
         
            unmatched += 1
            continue

        for pid in pred.get("packet_ids", []):
            packets_seen_by_flow[flow_id].add(pid)

        expected = job["expected_label"]
        predicted = pred["predicted_label"]
        rows.append({
            "replay_id": job["replay_id"],
            "pcap_path": job["pcap_path"],
            "expected_label": expected,
            "predicted_label": predicted,
            "correct": (expected.lower() == predicted.lower()),
            "confidence": pred["confidence"],
            "model_used": pred["model_used"],
            "packet_count": pred["packet_count"],
            "packet_ids": ";".join(str(p) for p in pred.get("packet_ids", [])),
            "mixed_flow": pred.get("mixed_flow", False),
            "captured_ts": pred.get("captured_ts"),
        })

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


    recall_fieldnames = ["replay_id", "pcap_path", "packets_read", "packets_received", "recall_pct"]
    recall_rows = []
    for flow_id, job in job_by_flow.items():
        total = job.get("packets_read")
        received = len(packets_seen_by_flow.get(flow_id, ()))
        recall_rows.append({
            "replay_id": job["replay_id"],
            "pcap_path": job["pcap_path"],
            "packets_read": total,
            "packets_received": received,
            "recall_pct": round(100.0 * received / total, 2) if total else None,
        })

    recall_out = Path(recall_out)
    with open(recall_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=recall_fieldnames)
        writer.writeheader()
        writer.writerows(recall_rows)

    correct_n = sum(1 for r in rows if r["correct"])
    print(f"Joined {len(rows)} windows to a replay job by flow_id "
          f"({unmatched} unmatched) -> {out_csv}")
    if rows:
        print(f"  Window-level accuracy: {correct_n}/{len(rows)} "
              f"({100.0 * correct_n / len(rows):.1f}%)")
    print(f"  {mixed} window(s) straddled more than one flow_id -- their "
          f"expected_label is genuinely ambiguous, not a join error.")
    print(f"Per-flow packet recall -> {recall_out}")
    if unmatched > 0.1 * (len(rows) + unmatched):
        print("  [WARN] >10% unmatched — check that truth_log.jsonl entries carry "
              "a flow_id field (requires the updated traffic_generator.py) and "
              "that provenance_log.jsonl exists at the expected path (requires "
              "the updated capture.py).")
    return rows, recall_rows


if __name__ == "__main__":
    join(DEFAULT_PROVENANCE, DEFAULT_GROUND_TRUTH, DEFAULT_OUT, DEFAULT_RECALL_OUT)
