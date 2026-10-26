import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chapter4_metrics as metrics
import chapter4_tables as tables
import make_fixture_run as fixture
import validate_run


def fixture_rows():
    rows = []
    for replay_id, label in fixture.JOBS:
        for index in range(10):
            predicted = fixture.predicted_label(label, index)
            correct = (label == predicted)
            rows.append({"replay_id": replay_id, "expected_label": label, "predicted_label": predicted,
                         "family_expected": label, "family_correct": correct,
                         "incident_id": "inc" if index == 0 and predicted != "Benign" else None})
    return rows


class MetricTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(metrics.percentile([200, 300, 400, 600], 95), 570)

    def test_latency_summary(self):
        timing = [{"detection_ts": 0, "evidence_complete_ts": ms / 1000, "detection_to_evidence_ms": 50} for ms in fixture.LATENCY_MS]
        summary = metrics.latency_summary(timing, 500)
        self.assertEqual((summary["n"], summary["mean_ms"], summary["median_ms"], summary["p95_ms"], summary["max_ms"]),
                         (4, 375.0, 350.0, 570.0, 600.0))
        self.assertEqual((summary["within_target"], summary["pct_within_target"]), (3, 75.0))

    def test_counts_and_detection(self):
        counts = metrics.scenario_counts(fixture_rows())
        self.assertEqual(counts, {"windows": 50, "benign_windows": 10, "attack_windows": 40,
                                  "attack_windows_detected": 23, "false_positives": 2})
        rates = {d["scenario"]: d["rate_pct"] for d in metrics.detection_by_scenario(fixture_rows())}
        self.assertEqual(rates, {"Benign": 20.0, "DDoS": 90.0, "Mirai": 100.0, "Recon": 40.0, "Spoofing": 0.0})

    def test_confusion_matrix_and_report(self):
        labels, matrix = metrics.confusion_matrix(fixture_rows())
        report = metrics.classification_report(labels, matrix)
        self.assertEqual(sum(map(sum, matrix)), 50)
        self.assertAlmostEqual(report["accuracy"], 0.62)
        recon = next(c for c in report["per_class"] if c["label"] == "Recon")
        self.assertEqual((recon["precision"], recon["recall"], recon["support"]), (1.0, 0.4, 10))

    def test_linkage_levels(self):
        linkage = metrics.linkage_levels(fixture_rows(), {"inc"}, {"DDoS", "Mirai", "Recon", "Spoofing"})
        self.assertEqual(linkage["l1_families_with_evidence"], ["DDoS", "Mirai", "Recon"])
        self.assertEqual((linkage["l3_windows_with_own_incident"], linkage["l3_detected_windows"], linkage["l3_pct"]), (3, 23, 13.0))

    def test_resource_summary_handles_missing_columns(self):
        samples = [metrics.parse_resource_line({"cpu_percent": c, "ram_percent": 30.0}) for c in (10, 20, 30, 40, 50)]
        self.assertEqual(metrics.resource_summary(samples), {"n_samples": 5, "cpu_mean": 30.0, "cpu_max": 50.0, "ram_mean": 30.0, "ram_max": 30.0})
        self.assertIsNone(metrics.resource_summary([metrics.parse_resource_line({"timestamp": 1})]))

    def test_percent_of_zero_denominator_is_none(self):
        self.assertIsNone(metrics.percent(1, 0))


class TableTests(unittest.TestCase):
    def test_missing_values_are_labelled_not_invented(self):
        text = tables.latency_table("r", None).to_text()
        self.assertIn("timing_log.jsonl missing", text)

    def test_markdown_and_csv(self):
        table = tables.Table("t", "Title", ["a", "b"], [[1, None]])
        self.assertIn("| 1 | not available |", table.to_markdown())


class ValidationTests(unittest.TestCase):
    def test_complete_fixture_passes_and_reports_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture.build_run(Path(tmp) / "run")
            config = {"packets_per_job": 100, "expected_jobs": 5}
            result = validate_run.validate(run / "raw", config)
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["packet_volume"]["intended_total_packets"], 500)
            self.assertEqual(result["packet_volume"]["observed_vs_expected_pct"], 100.0)

    def test_incomplete_experiment_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture.build_run(Path(tmp) / "run")
            result = validate_run.validate(run / "raw", {"packets_per_job": 100, "expected_jobs": 10})
            self.assertFalse(result["ok"])
            self.assertIn("incomplete experiment", result["errors"][0])

    def test_missing_raw_data_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "raw").mkdir()
            result = validate_run.validate(Path(tmp) / "raw", {})
            self.assertFalse(result["ok"])
            self.assertEqual(len(result["errors"]), 2)

    def test_duplicate_windows_are_flagged(self):
        items = [{"flow_id": 1, "packet_ids": [1, 2]}, {"flow_id": 1, "packet_ids": [1, 2]}]
        self.assertEqual(validate_run.duplicate_windows(items), 1)



class NewMetricTests(unittest.TestCase):
    def test_latency_stage_breakdown(self):
        rows = metrics.latency_stage_breakdown([{"window_first_packet_ts": 100.0, "window_complete_ts": 101.0, "detection_ts": 101.0,
                                                 "classification_done_ts": 101.05, "evidence_start_ts": 102.0,
                                                 "evidence_complete_ts": 102.3}])
        by = {r["stage"]: r for r in rows}
        self.assertEqual(by["Packet-window accumulation"]["median_ms"], 1000.0)
        self.assertEqual(by["Evidence acquisition (four layers + metadata)"]["n"], 0)
        self.assertEqual(by["Hashing and custody-log writing (preserve_bundle)"]["n"], 0)

    def test_pcap_packet_count(self):
        import struct
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.pcap"
            body = struct.pack("<IHHiIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)
            for _ in range(3):
                body += struct.pack("<IIII", 0, 0, 4, 4) + b"abcd"
            f.write_bytes(body)
            self.assertEqual(metrics.count_pcap_packets(f), 3)
            self.assertIsNone(metrics.count_pcap_packets(Path(d) / "missing.pcap"))

    def test_packet_funnel_and_tcpdump(self):
        jobs = [{"replay_id": 1, "status": "completed", "flow_id": 7, "expected_label": "Spoofing",
                 "pcap_path": "x/a.pcap", "packets_read": 100, "packets_sent": 100}]
        counters = {"by_flow": {"7": {"envelopes": 100, "non_ip_dropped": 90, "processed": 10}}}
        windows = [{"flow_id": 7}]
        row = metrics.packet_funnel(jobs, counters, windows)[0]
        self.assertEqual((row["envelopes"], row["non_ip_dropped"], row["processed"]), (100, 90, 10))
        self.assertEqual((row["windows_expected"], row["windows_logged"], row["windows_logged_pct"]), (10, 1, 10.0))
        self.assertIsNone(metrics.packet_funnel(jobs, None, windows)[0]["envelopes"])
        text = "120 packets captured\n3 packets dropped by kernel\n"
        self.assertEqual(metrics.tcpdump_drops(text), {"captured": 120, "dropped": 3})
        self.assertEqual(metrics.tcpdump_drops(""), {"captured": None, "dropped": None})

    def test_custody_claims_no_provenance(self):
        c = metrics.custody_claims([{"entry_hash": "x"}], 0, {"B": {"result": "INVALID"}})
        self.assertEqual(c["provenance_fields_present"], [])
        self.assertEqual(c["tamper_results"], {"B": "INVALID"})

    def test_threshold_sweep_monotonic(self):
        w = [{"expected": "Benign", "predicted": "DDoS", "confidence": 0.85, "family_correct": False},
             {"expected": "Benign", "predicted": "DDoS", "confidence": 0.6, "family_correct": False},
             {"expected": "DDoS", "predicted": "DDoS", "confidence": 0.9, "family_correct": True}]
        rows = {r["threshold"]: r for r in metrics.threshold_sweep(w)}
        self.assertEqual(rows[0.5]["fp_windows"], 2)
        self.assertEqual(rows[0.8]["fp_windows"], 1)
        self.assertEqual(rows[0.95]["fp_windows"], 0)

if __name__ == "__main__":
    unittest.main()
