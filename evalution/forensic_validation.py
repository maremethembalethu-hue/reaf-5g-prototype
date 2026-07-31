
# Validates the forensic-readiness
# Independently re-verifies the chain-of-custody hash chain
# Checks which truth attack types actually got at least one evidence bundle, versus which were expected

import os
import csv
import json
import hashlib

from pathlib import Path

DEFAULT_EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "evidence")
DEFAULT_JOINED_RESULTS = "evaluation/joined_results.csv"
DEFAULT_OUT = "evaluation/forensic_validation_report.json"


def hash_directory(directory: Path) -> str:
    sha256 = hashlib.sha256()
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            with open(Path(root) / filename, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
    return sha256.hexdigest()


def verify_chain(coc_log: Path):
    # Recomputes every entry_hash and bundle_hash independently and checks
    # the previous_hash links form an unbroken chain.
    if not coc_log.exists():
        return False, ["chain_of_custody.log not found"], []

    with open(coc_log) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    problems = []
    expected_prev = "0" * 64
    for i, entry in enumerate(entries):
        if entry["previous_hash"] != expected_prev:
            problems.append(f"entry {i} ({entry['incident_id']}): previous_hash link broken")

        recomputed_entry_hash = hashlib.sha256(
            f"{entry['previous_hash']}{entry['bundle_hash']}{entry['timestamp']}{entry['incident_id']}".encode()
        ).hexdigest()
        if recomputed_entry_hash != entry["entry_hash"]:
            problems.append(f"entry {i} ({entry['incident_id']}): entry_hash does not match recomputed value")

        bundle_dir = Path(entry["bundle_dir"])
        if bundle_dir.exists():
            recomputed_bundle_hash = hash_directory(bundle_dir)
            if recomputed_bundle_hash != entry["bundle_hash"]:
                problems.append(f"entry {i} ({entry['incident_id']}): bundle_hash mismatch — "
                                 f"bundle contents changed since acquisition (possible tampering)")
        else:
            problems.append(f"entry {i} ({entry['incident_id']}): bundle_dir missing: {bundle_dir}")

        expected_prev = entry["entry_hash"]

    return (len(problems) == 0), problems, entries


def evaluate_forensic_readiness(evidence_dir=DEFAULT_EVIDENCE_DIR,
                                 joined_results_path=DEFAULT_JOINED_RESULTS,
                                 out_path=DEFAULT_OUT, benign_label="Benign"):
    coc_log = Path(evidence_dir) / "chain_of_custody.log"
    is_valid, problems, entries = verify_chain(coc_log)

    ground_truth_attack_types = set()
    joined_results_path = Path(joined_results_path)
    if joined_results_path.exists():
        with open(joined_results_path) as f:
            rows = list(csv.DictReader(f))
        ground_truth_attack_types = {r["expected_label"] for r in rows if r["expected_label"] != benign_label}

    attack_types_with_evidence = {e["attack_type"] for e in entries}
    attack_types_missed = sorted(ground_truth_attack_types - attack_types_with_evidence)

    report = {
        "chain_of_custody_valid": is_valid,
        "chain_problems": problems,
        "n_evidence_bundles": len(entries),
        "ground_truth_attack_types": sorted(ground_truth_attack_types),
        "attack_types_with_evidence": sorted(attack_types_with_evidence),
        "attack_types_never_triggered_evidence": attack_types_missed,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    evaluate_forensic_readiness(DEFAULT_EVIDENCE_DIR, DEFAULT_JOINED_RESULTS, DEFAULT_OUT)