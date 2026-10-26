
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # evaluation/: existing scripts
 
import chapter4_metrics as metrics
import chapter4_tables as tables
import run_layout
import validate_run
 
 
def load_existing_modules():
    try:
        import experiment_s, forensic_validation, results_engine, tamper_tests, truth_results
    except ImportError as error:
        sys.exit(f"cannot import an existing evaluation module ({error}); run from the repository's evaluation/ folder")
    return experiment_s, forensic_validation, results_engine, tamper_tests, truth_results
 
 
def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
 
 
def read_resource_samples(path):
    return [metrics.parse_resource_line(record) for record in read_jsonl(path)] if Path(path).exists() else []
 
 
def print_section(title, body=""):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if body:
        print(body)
 
 
def run_tamper_tests(tamper_tests, raw_dir, has_evidence):
    if not has_evidence:
        return None
    tamper_tests.EVAL_DIR = raw_dir
    tamper_tests.EVIDENCE_DIR = raw_dir / "evidence"
    return tamper_tests.run_all()
 
 
def model_provenance(config):
    models = sorted((Path(__file__).resolve().parents[3] / "5g-edge-node" / "detection_models").glob("*"))
    rows = [[f"model file {m.name}", hashlib.sha256(m.read_bytes()).hexdigest()[:16]] for m in models if m.is_file()]
    rows += [["packets per replay job (flow_samples.py)", config.get("packets_per_job")],
             ["window size (packets)", validate_run.WINDOW_SIZE],
             ["sampler blocks: size cap / seed", f"{config.get('block_packets', 1000)} / 42"],
             ["evidence trigger confidence", os.environ.get("TRIGGER_CONFIDENCE", "0.80")],
             ["trigger cooldown (s)", config.get("trigger_cooldown_seconds")]]
    return rows if models else rows + [["model files", "5g-edge-node/detection_models not found next to evaluation/"]]
 
 
def build_summary_rows(latency, detection, suite, tamper, linkage, bursts, resources):
    benign = next((d for d in detection if d["kind"] == "false_positive"), None)
    attacks = [d for d in detection if d["kind"] == "detection"]
    rows = [
        ["1 Latency", None if latency is None else
         f"median {latency['median_ms']:.2f} ms; {latency['pct_within_target']:.1f}% of {latency['n']} within {latency['target_ms']} ms"],
        ["2 Scenario detection", "; ".join(f"{d['scenario']} {d['rate_pct']:.1f}% ({d['windows']})" for d in attacks)],
        ["2 Benign false positives", None if benign is None else f"{benign['rate_pct']:.1f}% of {benign['windows']} windows"],
        ["3 Evidence completeness", f"{suite['experiment_3_completeness'].get('overall_completeness_pct')}%"
         if suite["experiment_3_completeness"].get("incidents") else None],
        ["4 Evidence integrity", None if tamper is None else
         ", ".join(f"{k}: {t['result']}" for k, t in sorted(tamper["tests"].items()))],
        ["5 Chain of custody", None if tamper is None else
         f"valid before tampering: {'yes' if tamper['valid_before_tampering'] else 'no'} ({tamper['chain_entries']} entries)"],
        ["6 Evidence for detection", f"L1 {len(linkage['l1_families_with_evidence'])} of {len(linkage['l1_families_replayed'])}; "
         f"L2 {bursts['burst_level_evidence_pct']}%; L3 {linkage['l3_pct']}%"],
        ["7 Acquisition after detection", f"{suite['experiment_7_reliability'].get('evidence_success_rate_pct')}%"],
        ["8 Resource overhead", None if resources is None else
         f"CPU mean {resources['cpu_mean']}%; RAM mean {resources['ram_mean']}%"],
    ]
    return rows
 
 
def evaluate(run_dir):
    run_dir = Path(run_dir).resolve()
    raw_dir = run_dir / "raw"
    config = run_layout.read_config(run_dir)
    config.setdefault("run_id", run_dir.name)
    analysis = run_layout.new_analysis_dir(run_dir)
    metrics_dir, tables_dir = analysis / "metrics", analysis / "tables"
 
    validation = validate_run.validate(raw_dir, config)
    (metrics_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    print_section("Data validation")
    for message in validation["errors"]:
        print(f"ERROR   {message}")
    for message in validation["warnings"]:
        print(f"WARNING {message}")
    if not validation["ok"]:
        print(f"\nEvaluation stopped: required data is missing or the experiment is incomplete. See {metrics_dir / 'validation.json'}")
        return 2
    print("raw data present" if not validation["warnings"] else "raw data present with the warnings above")
 
    experiment_s, forensic_validation, results_engine, tamper_tests, truth_results = load_existing_modules()
    truth_path = raw_dir / "truth_log.jsonl"
    items_path = metrics_dir / "items_log_w10.jsonl"
    items_path.write_text("".join(json.dumps(i) + "\n" for i in read_jsonl(raw_dir / "items_log.jsonl")
                                  if i.get("packet_count") == validate_run.WINDOW_SIZE))
    evidence_dir = raw_dir / "evidence"
    has_evidence = bool(list(evidence_dir.glob("incident_*")))
 
    print_section("Join of predictions with replay ground truth (truth_results.py)")
    _, recall_rows = truth_results.join(str(items_path), str(truth_path), str(metrics_dir / "joined_results.csv"),
                       str(metrics_dir / "flow_recall_summary.csv"))
    rows = results_engine.compute_joined_rows(str(items_path), str(truth_path))
 
    experiment_s.EVAL_DIR, experiment_s.EVIDENCE_DIR = raw_dir, evidence_dir
    suite = experiment_s.run_all()
    (metrics_dir / "experiment_suite_report.json").write_text(json.dumps(suite, indent=2))
 
    tamper_report = run_tamper_tests(tamper_tests, raw_dir, has_evidence and (evidence_dir / "chain_of_custody.log").exists())
    if tamper_report:
        (metrics_dir / "tamper_test_report.json").write_text(json.dumps(tamper_report, indent=2))
    forensic_report = None
    if has_evidence:
        forensic_report = forensic_validation.evaluate_forensic_readiness(
            evidence_dir=evidence_dir, truth_log_path=truth_path,
            joined_results_path=metrics_dir / "joined_results.csv", out_path=metrics_dir / "forensic_validation_report.json")
 
    custody = read_jsonl(evidence_dir / "chain_of_custody.log")
    timing = read_jsonl(raw_dir / "timing_log.jsonl")
    truth_families = {job["expected_label"] for job in read_jsonl(truth_path) if metrics.is_attack(job["expected_label"])}
    latency = metrics.latency_summary(timing, experiment_s.TARGET_MS)
    counts = metrics.scenario_counts(rows)
    detection = metrics.detection_by_scenario(rows)
    labels, matrix = metrics.confusion_matrix(rows)
    report = metrics.classification_report(labels, matrix)
    linkage = metrics.linkage_levels(rows, metrics.evidence_ids(custody), truth_families)
    resources = metrics.resource_summary(read_resource_samples(evidence_dir / "resources.log"))
    tamper = metrics.tamper_summary(tamper_report, forensic_report["chain_of_custody_valid"]) if tamper_report and forensic_report else None
 
    jobs = read_jsonl(truth_path)
    windows, unmatched = metrics.join_windows(read_jsonl(items_path), jobs, truth_results._to_family)
    custody_ids = metrics.evidence_ids(custody)
    trigger_threshold = float(os.environ.get("TRIGGER_CONFIDENCE", "0.80"))
    stages = metrics.stage_confusions(windows, truth_results._to_family)
    profile = metrics.false_positive_profile(windows, unmatched, trigger_threshold, config.get("block_packets", 1000), custody_ids)
    sweep = metrics.threshold_sweep(windows)
    consequences = metrics.fp_consequences(windows, custody_ids, evidence_dir, timing, experiment_s.COOLDOWN_SECONDS, experiment_s._parse_ts)
    coverage = metrics.evidence_vs_attack(windows, custody_ids, evidence_dir)
    def _load_json(path):
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return None
    authenticity = {"verify": _load_json(run_dir / "logs" / "custody_verify.json"),
                    "probes": _load_json(run_dir / "logs" / "threat_model_probes.json")}
    claims = metrics.custody_claims(custody, len(forensic_report["chain_problems"]) if forensic_report else 0,
                                    tamper["tests"] if tamper else {},
                                    authenticity if authenticity["verify"] else None)
    stage_latency = metrics.latency_stage_breakdown(timing)
    packets_processed = sum(r["packets_received"] for r in recall_rows if r["packets_received"] is not None)
 
    counters_path = raw_dir / "capture_counters.json"
    counters = json.loads(counters_path.read_text()) if counters_path.exists() else None
    stats_path = raw_dir / "tcpdump_stats.txt"
    drops = metrics.tcpdump_drops(stats_path.read_text(errors="ignore") if stats_path.exists() else "")
    funnel = metrics.packet_funnel(jobs, counters, windows)
    if counters:
        packets_processed = counters["totals"].get("processed", packets_processed)
 
    run_id = config["run_id"]
    bursts = suite["experiment_6_burst_level"]
    all_tables = [tables.run_summary_table(config, validation["packet_volume"]),
                  tables.seven_counts_table(config, validation["packet_volume"], counts, len(custody_ids),
                                            packets_processed, bursts["total_bursts"]),
                  tables.packet_funnel_table(funnel, drops, counters is not None),
                  tables.latency_table(run_id, latency), tables.detection_table(run_id, detection),
                  *tables.classification_tables(labels, matrix, report, name="Overall (window level).", key=""),
                  *tables.stage_tables(stages), *tables.false_positive_tables(profile, sweep, consequences),
                  tables.latency_stage_table(stage_latency, latency), tables.evidence_vs_attack_table(coverage),
                  tables.provenance_table(model_provenance(config)),
                  tables.acquisition_table(run_id, suite["experiment_7_reliability"]),
                  tables.completeness_table(run_id, suite["experiment_3_completeness"])]
    if tamper:
        all_tables += [tables.integrity_table(tamper), tables.chain_table(tamper),
                       tables.claims_table(claims, tamper["validator_chain_valid"])]
    all_tables += [tables.linkage_table(linkage, bursts), tables.resource_table(run_id, resources),
                   tables.summary_table(build_summary_rows(latency, detection, suite, tamper, linkage, bursts, resources))]
 
    (metrics_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "config": config, "packet_volume": validation["packet_volume"], "counts": counts,
        "latency": latency, "detection": detection, "classification": report, "confusion_labels": labels,
        "confusion_matrix": matrix, "false_positive_profile": profile, "threshold_sweep": sweep,
        "fp_consequences": consequences, "evidence_vs_attack": coverage, "latency_stages": stage_latency, "packet_funnel": funnel, "tcpdump": drops,
        "stages": {k: {"labels": v[0], "matrix": v[1], "report": v[2]} for k, v in stages.items()}, "linkage": linkage, "episodes": bursts, "resources": resources,
        "tamper": tamper}, indent=2, default=str))
    tables.write_all(all_tables, tables_dir)
 
    for table in all_tables:
        print(f"\n{table.to_text()}")
    print(f"\nResults written to {analysis}")
    return 0
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="experiment run directory containing raw/ and config/")
    sys.exit(evaluate(parser.parse_args().run_dir))