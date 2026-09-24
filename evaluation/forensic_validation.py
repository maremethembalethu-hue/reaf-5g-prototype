import os
import csv
import json
import hashlib

from pathlib import Path

# PATHS
CODE_DIR = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_DIR = ( CODE_DIR.parent / Path(os.getenv("EVIDENCE_DIR", "evidence")) )
DEFAULT_TRUTH_LOG = ( CODE_DIR.parent / "evaluation" / "truth_log.jsonl" )
DEFAULT_JOINED_RESULTS = (CODE_DIR.parent / "evaluation" / "joined_results.csv" )
DEFAULT_OUT = (CODE_DIR.parent / "evaluation" /"validation" / "forensic_validation_report.json")



# HASH DIRECTORY

def hash_directory(directory):

    # Calculate a SHA-256 hash over all files in a directory.
    sha256 = hashlib.sha256()
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            file_path = Path(root) / filename
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
    return sha256.hexdigest()



# VERIFY CHAIN OF CUSTODY


def verify_chain(coc_log):

    # Independently verifies the chain_of_custody.log.
    if not coc_log.exists():
        return (False,["chain_of_custody.log not found"],[])
    entries = []

    try:
        with open(coc_log, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

    except Exception as exc:

        return (False,[f"Could not read chain_of_custody.log: {exc}"],[])

    problems = []

    # First entry must point to an all-zero previous hash.
    expected_prev = "0" * 64

    for i, entry in enumerate(entries):
        incident_id = entry.get("incident_id", f"entry_{i}")
        if entry.get("previous_hash") != expected_prev:
            problems.append(
                f"entry {i} ({incident_id}): "
                f"previous_hash link broken"
            )
        # Recompute entry_hash
        try:
            recomputed_entry_hash = hashlib.sha256(
                (
                    f"{entry['previous_hash']}"
                    f"{entry['bundle_hash']}" 
                    f"{entry['timestamp']}" 
                    f"{entry['incident_id']}").encode()).hexdigest()

            if recomputed_entry_hash != entry.get("entry_hash"):

                problems.append(f"entry {i} ({incident_id}): "
                    f"entry_hash does not match recomputed value")

        except KeyError as exc:

            problems.append(f"entry {i} ({incident_id}): "
                f"missing field required to recompute entry_hash: {exc}")

        
        # Verify evidence bundle
        bundle_dir = None
        try:
            logged_bundle_path = Path(entry["bundle_dir"])
            # Evidence path recorded by the edge container.
            if str(logged_bundle_path).startswith("/evidence/"):
                bundle_dir = ( CODE_DIR.parent / "evidence" / logged_bundle_path.relative_to("/evidence"))
            # Absolute path.
            elif logged_bundle_path.is_absolute():
                bundle_dir = logged_bundle_path
            # Relative path.
            else:
                bundle_dir = (CODE_DIR.parent / logged_bundle_path)

        except KeyError:
            problems.append(
                f"entry {i} ({incident_id}): "
                f"bundle_dir field missing")

        
        # Check bundle exists and hash matches
        if bundle_dir is not None:
            if bundle_dir.exists():
                try:
                    recomputed_bundle_hash = hash_directory(bundle_dir)
                    if recomputed_bundle_hash != entry.get("bundle_hash"):
                        problems.append(
                            f"entry {i} ({incident_id}): "
                            f"bundle_hash mismatch — "
                            f"bundle contents changed since acquisition "
                            f"(possible tampering)")

                except Exception as exc:
                    problems.append(
                        f"entry {i} ({incident_id}): "
                        f"could not hash bundle: {exc}")

            else:
                problems.append(
                    f"entry {i} ({incident_id}): "
                    f"bundle_dir missing: {bundle_dir}")

        
        # Move to the next chain entry
        

        expected_prev = entry.get("entry_hash",expected_prev)
    return (len(problems) == 0,problems, entries)



# READ TRUTH LOG
def read_truth_log(truth_log_path, benign_label="Benign"):
    # Reads truth_log.jsonl.
    truth_records = []
    ground_truth_attack_types = set()
    problems = []

    if not truth_log_path.exists():

        problems.append(f"truth_log.jsonl not found at {truth_log_path}")
        return ( truth_records, ground_truth_attack_types, problems )

    try:
        with open(truth_log_path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    job = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append(f"truth_log.jsonl line {line_number}: " f"invalid JSON: {exc}")
                    continue
                truth_records.append(job)
                expected_label = job.get( "expected_label", "")
                status = job.get("status","")
                if (status == "completed" and expected_label and expected_label != benign_label):
                    ground_truth_attack_types.add(expected_label)

    except Exception as exc:
        problems.append( f"Could not read truth_log.jsonl: {exc}")
    return ( truth_records, ground_truth_attack_types, problems)



# READ JOINED RESULTS

def read_joined_results(joined_results_path):
    
    # Reads joined_results.csv using csv.DictReader.
    joined_records = []
    problems = []
    if not joined_results_path.exists():
        problems.append(
            f"joined_results.csv not found at "
            f"{joined_results_path}")

        return (joined_records,problems)

    try:
        with open( joined_results_path, "r", newline="" ) as f:
            reader = csv.DictReader(f)
            for row in reader:
                joined_records.append(row)
    except Exception as exc:
        problems.append(f"Could not read joined_results.csv: {exc}")
    return (joined_records, problems)

# EXTRACT ATTACK TYPES FROM JOINED RESULTS
def get_attack_types_from_results(joined_records):
    # Extract attack types from joined_results.csv.
    attack_types = set()

    possible_columns = ["attack_type","Attack Type","attack","label","Label","expected_label","predicted_label","Predicted Label"]

    for row in joined_records:
        for column in possible_columns:
            if column in row:
                value = row.get(column, "")
                if value is not None:
                    value = str(value).strip()
                if value:
                    attack_types.add(value)
                    break

    return attack_types



# COMPARE GROUND TRUTH WITH RESULTS


def compare_ground_truth_with_results(ground_truth_attack_types,result_attack_types,benign_label="Benign"):

    # Compare attack types expected by the ground truth with attack
    # types present in joined_results.csv.
    

    # Do not treat benign traffic as an attack.
    result_attack_types = {attack_type for attack_type in result_attack_types if attack_type != benign_label }

    attack_types_missed = sorted(ground_truth_attack_types - result_attack_types)

    attack_types_present = sorted( ground_truth_attack_types & result_attack_types)

    return (attack_types_present,attack_types_missed)



# MAIN FORENSIC READINESS EVALUATION


def evaluate_forensic_readiness(evidence_dir=DEFAULT_EVIDENCE_DIR,truth_log_path=DEFAULT_TRUTH_LOG,joined_results_path=DEFAULT_JOINED_RESULTS,out_path=DEFAULT_OUT,benign_label="Benign"):
     # Complete forensic-readiness validation.
    # 1. VERIFY CHAIN OF CUSTODY
    coc_log = Path(evidence_dir) / "chain_of_custody.log"
    chain_valid, chain_problems, entries = verify_chain(coc_log)

    # 2. READ TRUTH LOG
    (truth_records,ground_truth_attack_types,truth_problems) = read_truth_log(Path(truth_log_path),benign_label)
    # 3. READ JOINED RESULTS
    (joined_records,joined_problems) = read_joined_results(Path(joined_results_path))
    # 4. GET ATTACK TYPES FROM JOINED RESULTS

    result_attack_types = get_attack_types_from_results(joined_records)
    # 5. COMPARE TRUTH WITH JOINED RESULTS
    (attack_types_present_in_results,attack_types_missed) = compare_ground_truth_with_results(ground_truth_attack_types,result_attack_types,benign_label)
    # 6. GET ATTACK TYPES THAT PRODUCED EVIDENCE
    

    attack_types_with_evidence = { e.get("attack_type")for e in entries if e.get("attack_type") }

    attack_types_with_evidence = { attack_type for attack_type in attack_types_with_evidence if attack_type != benign_label}

    attack_types_never_triggered_evidence = sorted(ground_truth_attack_types - attack_types_with_evidence)

    
    # 7. COMBINE PROBLEMS
    

    all_problems = []

    all_problems.extend(chain_problems)
    all_problems.extend(truth_problems)
    all_problems.extend(joined_problems)

    
    # 8. DETERMINE WHETHER GROUND-TRUTH COMPARISON RAN
    

    ground_truth_comparison_ran = ( truth_log_path.exists() and joined_results_path.exists() and len(truth_records) > 0 and len(joined_records) > 0 )

    
    # 9. BUILD REPORT
    

    report = {

        # Chain of custody
        "chain_of_custody_valid": chain_valid,
        "chain_problems": chain_problems,
        "n_evidence_bundles": len(entries),
        # Files
        "truth_log_path": str( Path(truth_log_path) ),
        "joined_results_path": str( Path(joined_results_path) ),
        # Ground truth
        "n_truth_records": len(truth_records),

        "ground_truth_comparison_ran": ground_truth_comparison_ran,

        "ground_truth_attack_types": sorted(ground_truth_attack_types),

        # Joined results
        "n_joined_result_records": len(joined_records),

        "attack_types_in_joined_results": sorted(result_attack_types),

        "ground_truth_attack_types_present_in_results": attack_types_present_in_results,

        "ground_truth_attack_types_missing_from_results": attack_types_missed,

        # Evidence
        "attack_types_with_evidence": sorted(attack_types_with_evidence),

        "attack_types_never_triggered_evidence": attack_types_never_triggered_evidence,

        # General problems
        "validation_problems": all_problems
    }

    
    # 10. WRITE REPORT
    

    out_path = Path(out_path)

    out_path.parent.mkdir(parents=True,exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(report,f,indent=2)
    # 11. PRINT REPORT
    print(json.dumps(report,indent=2))

    return report



# PROGRAM ENTRY POINT
if __name__ == "__main__":
    evaluate_forensic_readiness( evidence_dir=DEFAULT_EVIDENCE_DIR, truth_log_path=DEFAULT_TRUTH_LOG, joined_results_path=DEFAULT_JOINED_RESULTS, out_path=DEFAULT_OUT)

