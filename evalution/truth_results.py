


import json
import csv
import argparse
from pathlib import Path

DEFAULT_PREDICTIONS = "evaluation/predictions_log.jsonl"
DEFAULT_GROUND_TRUTH = "evaluation/ground_truth_log.jsonl"
DEFAULT_OUT = "evaluation/joined_results.csv"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_job(ts, jobs):
    for job in jobs:
        if job["start_time"] <= ts <= job["end_time"]:
            return job
    return None


def join(predictions_path=DEFAULT_PREDICTIONS, ground_truth_path=DEFAULT_GROUND_TRUTH, out_csv=DEFAULT_OUT):
    predictions = load_jsonl(predictions_path)
    jobs = load_jsonl(ground_truth_path)

    fieldnames = ["replay_id", "pcap_path", "expected_label", "predicted_label",
                  "confidence", "model_used", "packet_count", "flow_start_time"]
    rows = []
    unmatched = 0
    for pred in predictions:
        job = find_job(pred["flow_start_time"], jobs)
        if job is None:
            unmatched += 1
            continue
        rows.append({
            "replay_id": job["replay_id"],
            "pcap_path": job["pcap_path"],
            "expected_label": job["expected_label"],
            "predicted_label": pred["attack_type"],
            "confidence": pred["confidence"],
            "model_used": pred["model_used"],
            "packet_count": pred["packet_count"],
            "flow_start_time": pred["flow_start_time"],
        })

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Joined {len(rows)} flows to a replay job ({unmatched} unmatched, dropped) -> {out_csv}")
    if unmatched > 0.1 * (len(rows) + unmatched):
        print("  [WARN] >10% unmatched — check clock sync between containers "
              "and whether SCHEDULER_DRAIN_SECONDS is long enough.")
    return rows

if __name__ == "__main__":
    join(DEFAULT_PREDICTIONS, DEFAULT_GROUND_TRUTH, DEFAULT_OUT)