# Samples a fixed number of complete flows from a source PCAP, writing them
# to a new, much smaller PCAP that replay_engine.py can replay.


import random
import json
import time
import statistics
from pathlib import Path
from collections import defaultdict

from scapy.utils import PcapReader, wrpcap
from scapy.all import IP, TCP, UDP


RANDOM_STATE = 42
SOURCE_ROOT = "./pcaps"
OUT_PATH = "pre-selected/manifest.json"
OUT_DIR_PCAP = "pre-selected/replay_pcaps"
TIMING_REPORT_PATH = "pre-selected/timing_report.json"


TARGET_IDLE_TIMEOUT_DEFAULT = 2.0   # seconds; your current IDLE_TIMEOUT_SECONDS
GAP_PERCENTILE_DEFAULT = 99.0       # which percentile of inter-arrival gaps to design around
SAFETY_MARGIN_DEFAULT = 1.2         # headroom multiplier over the raw percentile


def flow_key(pkt):
    ip = pkt[IP]
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        sport, dport = 0, 0
    a, b = (ip.src, sport), (ip.dst, dport)
    endpoints = tuple(sorted([a, b]))
    return (ip.proto, endpoints[0], endpoints[1])


def analyze_flows(source_pcap, max_packets=None):
    #Single pass over the PCAP. For every flow, tracks Only a handful of floats are kept per flow - no packet objects, no list
    #of individual gaps     
    t0 = time.perf_counter()
    stats = {}
    packets_scanned = 0

    with PcapReader(str(source_pcap)) as reader:
        for i, pkt in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            if IP not in pkt:
                continue

            key = flow_key(pkt)
            t = float(pkt.time)
            packets_scanned += 1

            s = stats.get(key)
            if s is None:
                stats[key] = {
                    "count": 1, "first": t, "last": t, "prev": t,
                    "max_gap": 0.0, "sum_gap": 0.0, "n_gaps": 0,
                }
            else:
                gap = t - s["prev"]
                if gap > s["max_gap"]:
                    s["max_gap"] = gap
                s["sum_gap"] += gap
                s["n_gaps"] += 1
                s["prev"] = t
                s["last"] = t
                s["count"] += 1

    elapsed = time.perf_counter() - t0

    # finalize derived fields (duration, mean_gap) now that scanning is done
    for s in stats.values():
        s["duration"] = s["last"] - s["first"]
        s["mean_gap"] = (s["sum_gap"] / s["n_gaps"]) if s["n_gaps"] else 0.0
        del s["prev"]  # scratch field, not needed after the pass

    rate = packets_scanned / elapsed if elapsed > 0 else 0.0
    print(f"   analyze_flows: scanned {packets_scanned:,} packets to "
          f"{len(stats):,} flows in {elapsed:.3f}s ({rate:,.0f} pkts/sec)")

    return stats, packets_scanned


def collect_chosen_packets(source_pcap, chosen_keys, max_packets=None):
    #Stream the PCAP a second time, keeping only packets whose
    #flow_key is in chosen_keys. Memory usage is bounded by the size of the
    #sampled flows, not the whole file.
   
    t0 = time.perf_counter()
    chosen_keys = set(chosen_keys)
    packets = []
    packets_scanned = 0
    with PcapReader(str(source_pcap)) as reader:
        for i, pkt in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            if IP not in pkt:
                continue
            packets_scanned += 1
            key = flow_key(pkt)
            if key in chosen_keys:
                packets.append(pkt)
    elapsed = time.perf_counter() - t0

    rate = packets_scanned / elapsed if elapsed > 0 else 0.0
    print(f"   collect_chosen_packets: rescanned {packets_scanned:,} packets, "
          f"kept {len(packets):,} in {elapsed:.3f}s ({rate:,.0f} pkts/sec)")

    return packets


def _percentile(sorted_values, p):
    #Linear-interpolated percentile over an already-sorted list. Fine for the flow-count scales we deal with here.
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    idx = (p / 100) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _round_up_nice(value, step=0.5):
    if value <= 0:
        return step
    import math
    return math.ceil(value / step) * step


def summarize_timing(flow_stats, percentiles=(50, 90, 95, 99, 99.9, 100)):
    #Reduces a flow_stats dict down to percentile summaries of duration and
    #max_gap, plus the raw sorted arrays.
    
    durations = sorted(s["duration"] for s in flow_stats.values())
    max_gaps = sorted(s["max_gap"] for s in flow_stats.values())

    return {
        "num_flows": len(flow_stats),
        "duration_percentiles": {f"p{p}": round(_percentile(durations, p), 4) for p in percentiles},
        "max_gap_percentiles": {f"p{p}": round(_percentile(max_gaps, p), 4) for p in percentiles},
        "_durations_sorted": durations,   # kept for recommend_replay_params; strip before dumping to disk
        "_max_gaps_sorted": max_gaps,
    }


def recommend_replay_params(timing_summary,
                             target_idle_timeout=TARGET_IDLE_TIMEOUT_DEFAULT,
                             gap_percentile=GAP_PERCENTILE_DEFAULT,
                             safety_margin=SAFETY_MARGIN_DEFAULT):
    #Given a timing_summary 
    #This is deliberately a statistics-based estimate, not a prediction model:
    #there's no ground truth for "the correct" timeout/speed, so the goal is
    #a defensible, explainable choice rather than a learned one.
   
    gaps = timing_summary["_max_gaps_sorted"]
    if not gaps:
        return {
            "suggested_idle_timeout": target_idle_timeout,
            "suggested_replay_speed": 1.0,
            "expected_split_flows_pct": 0.0,
            "basis_gap_percentile_value": 0.0,
        }

    p_gap = _percentile(gaps, gap_percentile)

    suggested_idle_timeout = _round_up_nice(p_gap * safety_margin, step=0.5)

    if p_gap <= 0:
        suggested_replay_speed = 1.0
    else:
        suggested_replay_speed = max(1.0, (p_gap * safety_margin) / target_idle_timeout)
        suggested_replay_speed = round(suggested_replay_speed, 2)

    # re-estimate, after scaling by the suggested speed, what fraction of
    # flows would still exceed the *target* idle timeout
    scaled = [g / suggested_replay_speed for g in gaps]
    n_split = sum(1 for g in scaled if g > target_idle_timeout)
    expected_split_flows_pct = round(100.0 * n_split / len(scaled), 2)

    return {
        "suggested_idle_timeout": suggested_idle_timeout,
        "suggested_replay_speed": suggested_replay_speed,
        "expected_split_flows_pct": expected_split_flows_pct,
        "basis_gap_percentile": gap_percentile,
        "basis_gap_percentile_value": round(p_gap, 4),
        "target_idle_timeout_used": target_idle_timeout,
        "safety_margin_used": safety_margin,
    }


def _strip_internal_fields(timing_summary):
    # Returns a JSON-safe copy of a timing_summary 
    return {k: v for k, v in timing_summary.items() if not k.startswith("_")}


def sample_pcap(source_pcap, expected_label, out_dir, n_flows=150,
                 priority="normal", max_packets=2_000_000,
                 random_state=RANDOM_STATE,
                 target_idle_timeout=TARGET_IDLE_TIMEOUT_DEFAULT,
                 gap_percentile=GAP_PERCENTILE_DEFAULT,
                 safety_margin=SAFETY_MARGIN_DEFAULT):
    """Samples up to n_flows complete flows from source_pcap and writes them,
    sorted by original relative timestamp. Also returns per-file timing
    statistics and a replay-parameter recommendation derived from them.
    """
    t_total_start = time.perf_counter()

    source_pcap = Path(source_pcap)
    print(f" {source_pcap.name}  (expected_label={expected_label})")

    # Pass 1: index flow keys + timing stats in one streaming pass.
    t_index_start = time.perf_counter()
    flow_stats, _packets_scanned = analyze_flows(source_pcap, max_packets=max_packets)
    t_index_elapsed = time.perf_counter() - t_index_start

    available = len(flow_stats)
    if available == 0:
        print(f" no IP flows found in {source_pcap}")
        return None

    # Choose which flows to keep, based only on the lightweight index.
    t_sample_start = time.perf_counter()
    rng = random.Random(random_state)
    keys = list(flow_stats.keys())
    chosen = rng.sample(keys, k=min(n_flows, available))
    t_sample_elapsed = time.perf_counter() - t_sample_start

    # Pass 2: re-stream the file, materializing only the chosen flows' packets.
    t_collect_start = time.perf_counter()
    packets = collect_chosen_packets(source_pcap, chosen, max_packets=max_packets)
    packets.sort(key=lambda p: float(p.time))

    if packets:
        t0 = float(packets[0].time)
        for p in packets:
            p.time = float(p.time) - t0
    t_collect_elapsed = time.perf_counter() - t_collect_start

    out_dir = Path(out_dir) / expected_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_pcap.stem}_sample.pcap"

    t_write_start = time.perf_counter()
    wrpcap(str(out_path), packets)
    t_write_elapsed = time.perf_counter() - t_write_start

    del packets

    # Timing summary + recommendation, computed over ALL flows in this file more representative of the file's true
    # timing characteristics.
    timing_summary = summarize_timing(flow_stats)
    recommendation = recommend_replay_params(
        timing_summary, target_idle_timeout=target_idle_timeout,
        gap_percentile=gap_percentile, safety_margin=safety_margin,
    )

    t_total_elapsed = time.perf_counter() - t_total_start
    sampling_ratio = (len(chosen) / available * 100) if available else 0.0

    print(f"{source_pcap.name}: sampled {len(chosen)}/{available} flows - {out_path}")
    print(f"   Flows available     : {available:,}")
    print(f"   Flows sampled       : {len(chosen):,}  ({sampling_ratio:.2f}% of available)")
    print(f"   Time (analyze)      : {t_index_elapsed:.3f}s")
    print(f"   Time (sampling)     : {t_sample_elapsed:.3f}s")
    print(f"   Time (collect+write): {t_collect_elapsed:.3f}s (incl. {t_write_elapsed:.3f}s write)")
    print(f"   Time (TOTAL, file)  : {t_total_elapsed:.3f}s")
    print(f"   Gap p{gap_percentile}            : {recommendation['basis_gap_percentile_value']:.3f}s")
    print(f"   Suggested idle_timeout : {recommendation['suggested_idle_timeout']:.2f}s")
    print(f"   Suggested replay_speed : {recommendation['suggested_replay_speed']:.2f}x")
    print(f"   Expected split flows   : {recommendation['expected_split_flows_pct']:.2f}%")
    print("-" * 60)

    return {
        "pcap_path": str(out_path),
        "expected_label": expected_label,
        "num_flows_sampled": len(chosen),
        "num_flows_available": available,
        "priority": priority,
        "elapsed_seconds": t_total_elapsed,
        "index_seconds": t_index_elapsed,
        "sample_seconds": t_sample_elapsed,
        "collect_write_seconds": t_collect_elapsed,
        "timing_summary": _strip_internal_fields(timing_summary),
        "recommendation": recommendation,
        # kept only in-process (not written to manifest.json) so the whole
        # dataset's timing can be merged into one global recommendation
        "_flow_stats": flow_stats,
    }


def sample_directory(source_root, out_dir, n_flows_per_pcap=150, priority_map=None,
                      target_idle_timeout=TARGET_IDLE_TIMEOUT_DEFAULT,
                      gap_percentile=GAP_PERCENTILE_DEFAULT,
                      safety_margin=SAFETY_MARGIN_DEFAULT):
    # Samples every .pcap found under each label subfolder of source_root.
    priority_map = priority_map or {}
    records = []

    t_dir_start = time.perf_counter()

    label_dirs = [d for d in sorted(Path(source_root).iterdir()) if d.is_dir()]
    all_pcaps = [(d.name, p) for d in label_dirs for p in sorted(d.glob("*.pcap"))]
    total_files = len(all_pcaps)
    print("-" * 60)
    print(" FLOW SAMPLING EXPERIMENT LOG")
    print("-" * 60)
    print(f"Found {len(label_dirs)} label folders, {total_files} total PCAP files to process")
    print("-" * 60)

    for idx, (label, pcap_path) in enumerate(all_pcaps, start=1):
        print(f"[FILE {idx}/{total_files}]", end=" ")
        rec = sample_pcap(
            pcap_path, expected_label=label, out_dir=out_dir,
            n_flows=n_flows_per_pcap, priority=priority_map.get(label, "normal"),
            target_idle_timeout=target_idle_timeout, gap_percentile=gap_percentile,
            safety_margin=safety_margin,
        )
        if rec:
            records.append(rec)

    t_dir_elapsed = time.perf_counter() - t_dir_start

    total_available = sum(r["num_flows_available"] for r in records)
    total_sampled = sum(r["num_flows_sampled"] for r in records)
    overall_ratio = (total_sampled / total_available * 100) if total_available else 0.0
    avg_time_per_file = (t_dir_elapsed / len(records)) if records else 0.0

    print("-" * 60)
    print(" EXPERIMENT SUMMARY")
    print("-" * 60)
    print(f" Source PCAPs processed      : {len(records)}")
    print(f" Total flows available       : {total_available:,}")
    print(f" Total flows sampled         : {total_sampled:,}")
    print(f" Overall flow sampling ratio : {overall_ratio:.2f}%")
    print(f" Total wall-clock time       : {t_dir_elapsed:.3f}s")
    print(f" Average time per PCAP       : {avg_time_per_file:.3f}s")
    print("-" * 60)


    # Merge every file's per-flow stats into one big flow_stats dict so the
    # recommendation reflects the whole dataset, not just one shard.
    global_flow_stats = {}
    for i, rec in enumerate(records):
        for key, s in rec["_flow_stats"].items():
            # keys can collide across files (same 5-tuple in different
            # captures) - disambiguate by prefixing with the file index
            global_flow_stats[(i, key)] = s

    global_timing_summary = summarize_timing(global_flow_stats)
    global_recommendation = recommend_replay_params(
        global_timing_summary, target_idle_timeout=target_idle_timeout,
        gap_percentile=gap_percentile, safety_margin=safety_margin,
    )

    print(" DATASET-WIDE REPLAY PARAMETER RECOMMENDATION")
    print("-" * 60)
    print(f" Flows analysed          : {global_timing_summary['num_flows']:,}")
    print(f" Gap p{gap_percentile}                : {global_recommendation['basis_gap_percentile_value']:.3f}s")
    print(f" Suggested idle_timeout  : {global_recommendation['suggested_idle_timeout']:.2f}s")
    print(f" Suggested replay_speed  : {global_recommendation['suggested_replay_speed']:.2f}x")
    print(f" Expected split flows    : {global_recommendation['expected_split_flows_pct']:.2f}%")
    print("-" * 60)

    timing_report = {
        "per_file": [
            {
                "pcap_path": r["pcap_path"],
                "expected_label": r["expected_label"],
                "timing_summary": r["timing_summary"],
                "recommendation": r["recommendation"],
            }
            for r in records
        ],
        "dataset_wide": {
            "timing_summary": _strip_internal_fields(global_timing_summary),
            "recommendation": global_recommendation,
        },
    }

    timing_report_path = Path(TIMING_REPORT_PATH)
    timing_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_report_path, "w") as f:
        json.dump(timing_report, f, indent=2)
    print(f" Wrote timing report to {timing_report_path}")
    print("-" * 60)

    # drop the bulky internal field before returning records for manifest building
    for r in records:
        del r["_flow_stats"]

    return records, global_recommendation


def build_manifest(sample_records, out_path=OUT_PATH, global_recommendation=None):
    # Assigns a sequential replay_id in the given order — reorder sample_records
    # yourself beforehand if you want a specific run order.
    t0 = time.perf_counter()

    manifest = []
    for i, rec in enumerate(sample_records, start=1):
        manifest.append({
            "replay_id": i,
            "pcap_path": rec["pcap_path"],
            "expected_label": rec["expected_label"],
            "num_flows": rec["num_flows_sampled"],
            "priority": rec.get("priority", "normal"),
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "jobs": manifest,
            "recommended_replay_params": global_recommendation,  # None if not computed
        }, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"Wrote {len(manifest)} replay jobs to {out_path}  ({elapsed:.3f}s)")
    return manifest


def load_manifest(path=OUT_PATH):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    t_run_start = time.perf_counter()

    records, global_recommendation = sample_directory(SOURCE_ROOT, OUT_DIR_PCAP, n_flows_per_pcap=150)
    build_manifest(records, OUT_PATH, global_recommendation=global_recommendation)

    t_run_elapsed = time.perf_counter() - t_run_start
    print(f"\nFull run Timing: {t_run_elapsed:.3f}s")