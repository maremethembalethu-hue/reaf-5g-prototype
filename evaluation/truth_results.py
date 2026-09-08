import json
import csv
from pathlib import Path
from collections import defaultdict

DEFAULT_PROVENANCE = "items_log.jsonl"
DEFAULT_GROUND_TRUTH = "truth_log.jsonl"
DEFAULT_OUT = "joined_results.csv"
DEFAULT_RECALL_OUT = "flow_recall_summary.csv"

# Flood-merge for FAMILY-level scoring
FAMILY_MERGE = {"DDoS": "Flood", "DoS": "Flood", "Flood_uncertain": "Flood"}


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


def _to_family(label):
    return FAMILY_MERGE.get(label, label)


def join(predictions_path=DEFAULT_PROVENANCE, ground_truth_path=DEFAULT_GROUND_TRUTH,
         out_csv=DEFAULT_OUT, recall_out=DEFAULT_RECALL_OUT):
    predictions = load_jsonl(predictions_path)
    jobs = load_jsonl(ground_truth_path)
    job_by_flow = build_job_index(jobs)

    fieldnames = ["replay_id", "pcap_path", "expected_label", "predicted_label",
                  "correct", "family_expected", "family_predicted", "family_correct",
                  "confidence", "model_used", "packet_count",
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
        family_expected = _to_family(expected)
        family_predicted = _to_family(predicted)

        rows.append({
            "replay_id": job["replay_id"],
            "pcap_path": job["pcap_path"],
            "expected_label": expected,
            "predicted_label": predicted,
            "correct": (expected.lower() == predicted.lower()),
            "family_expected": family_expected,
            "family_predicted": family_predicted,
            "family_correct": (family_expected.lower() == family_predicted.lower()),
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

    print(f"Joined {len(rows)} windows to a replay job by flow_id "
          f"({unmatched} unmatched) -> {out_csv}")
    print(f"  {mixed} window(s) straddled more than one flow_id -- their "
          f"expected_label is genuinely ambiguous, not a join error.")
    print(f"Per-flow packet recall -> {recall_out}")
    if unmatched > 0.1 * (len(rows) + unmatched):
        print("  [WARN] >10% unmatched — check that truth_log.jsonl entries carry "
              "a flow_id field and that provenance_log.jsonl exists at the expected path.")

    _print_accuracy_breakdown(rows)
    return rows, recall_rows


def _print_accuracy_breakdown(rows):
    if not rows:
        return
    n = len(rows)

    def pct(numer, denom):
        return f"{100.0 * numer / denom:.1f}% ({numer}/{denom})" if denom else "n/a (0 rows)"

    # 1. Strict, exact-label accuracy (no merging at all) — the number as it
    #    always was, kept for continuity with every prior run's reporting.
    strict_correct = sum(1 for r in rows if r["correct"])

    # 2. Attack vs Benign — the stage-1 question in isolation.
    binary_correct = sum(
        1 for r in rows
        if (r["expected_label"].lower() != "benign") == (r["predicted_label"].lower() != "benign")
    )

    # 3. Family-level accuracy, DDoS+DoS merged to Flood, Flood_uncertain
   
    family_correct = sum(1 for r in rows if r["family_correct"])

    # 3b. Same family-level accuracy, restricted to rows expected to be an
 
    attack_rows = [r for r in rows if r["expected_label"].lower() != "benign"]
    family_attack_correct = sum(1 for r in attack_rows if r["family_correct"])

    # 4. DDoS-vs-DoS specifically: stage 3's own question in isolation,
   
    flood_rows = [r for r in rows if r["expected_label"].lower() in ("ddos", "dos")]
    flood_correct = sum(1 for r in flood_rows if r["correct"])

    print("\n Accuracy breakdown ")
    print(f"  Strict (exact label match, no merging):        {pct(strict_correct, n)}")
    print(f"  Attack vs Benign (stage 1 in isolation):        {pct(binary_correct, n)}")
    print(f"  6-family, DDoS+DoS merged to Flood (all rows):  {pct(family_correct, n)}")
    print(f"  6-family, attack rows only:                     {pct(family_attack_correct, len(attack_rows))}")
    print(f"  DDoS vs DoS specifically (stage 3 in isolation): {pct(flood_correct, len(flood_rows))}")


if __name__ == "__main__":
    join(DEFAULT_PROVENANCE, DEFAULT_GROUND_TRUTH, DEFAULT_OUT, DEFAULT_RECALL_OUT)