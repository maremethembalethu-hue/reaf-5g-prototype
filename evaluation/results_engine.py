# equivalent of true_results.py's join(): reads predictions_log.jsonl
# and truth_log.jsonl DIRECTLY, both of which are written continuously as
# the pipeline runs, and returns the same row shape true_results.py used to
# write to joined_results.csv.

import json

FAMILY_MERGE = {"DDoS": "Flood", "DoS": "Flood", "Flood_uncertain": "Flood"}


def _load_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line caught mid-write by a concurrent append — skip, not fatal
    except FileNotFoundError:
        pass
    return rows


def _to_family(label):
    return FAMILY_MERGE.get(label, label)


def _build_job_index(jobs):
    by_flow = {}
    for job in jobs:
        flow_id = job.get("flow_id")
        if flow_id is None:
            continue
        if flow_id not in by_flow or job.get("status") == "completed":
            by_flow[flow_id] = job
    return by_flow


def compute_joined_rows(predictions_path, truth_path):
    predictions = _load_jsonl(predictions_path)
    jobs = _load_jsonl(truth_path)
    job_by_flow = _build_job_index(jobs)

    rows = []
    for pred in predictions:
        flow_id = pred.get("flow_id")
        job = job_by_flow.get(flow_id) if flow_id is not None else None
        if job is None:
            continue  # no matching truth_log entry yet (e.g. it hasn't been written this instant) — skip for now, not an error

        expected = job["expected_label"]
        predicted = pred["predicted_label"]
        family_expected = _to_family(expected)
        family_predicted = _to_family(predicted)

        rows.append({
            "replay_id": job["replay_id"], "pcap_path": job["pcap_path"],
            "expected_label": expected, "predicted_label": predicted,
            "correct": expected.lower() == predicted.lower(),
            "family_expected": family_expected, "family_predicted": family_predicted,
            "family_correct": family_expected.lower() == family_predicted.lower(),
            "confidence": pred.get("confidence"), "model_used": pred.get("model_used", ""),
            "packet_count": pred.get("packet_count"),
            "mixed_flow": pred.get("mixed_flow", False),
            "captured_ts": pred.get("captured_ts"),
        })
    return rows


def accuracy_breakdown(rows):

    if not rows:
        return {"strict": None, "binary": None, "family": None, "flood": None}
    n = len(rows)
    strict = sum(1 for r in rows if r["expected_label"].lower() == r["predicted_label"].lower())
    binary = sum(1 for r in rows
                 if (r["expected_label"].lower() != "benign") == (r["predicted_label"].lower() != "benign"))
    fam_correct = sum(1 for r in rows if r["family_correct"])
    flood_rows = [r for r in rows if r["expected_label"].lower() in ("ddos", "dos")]
    flood = (sum(1 for r in flood_rows if r["correct"]) / len(flood_rows) * 100) if flood_rows else None
    return {
        "strict": round(100 * strict / n, 1), "binary": round(100 * binary / n, 1),
        "family": round(100 * fam_correct / n, 1), "flood": flood,
    }