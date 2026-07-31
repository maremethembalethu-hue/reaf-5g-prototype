# Samples a fixed number of *complete* flows from a source PCAP, writing them
# to a new, much smaller PCAP that replay_engine.py can replay

import random
import json
import time
from pathlib import Path

from scapy.utils import PcapReader, wrpcap
from scapy.all import IP, TCP, UDP



RANDOM_STATE = 42
SOURCE_ROOT = "./pcaps"
OUT_PATH =  "pre-selected/manifest.json"
OUT_DIR_PCAP = "pre-selected/replay_pcaps"

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

def group_flows(source_pcap, max_packets=None):
    #pass over the source PCAP, grouping packets by flow key
    #  extremely large files may need chunked or
    # streaming-to-disk grouping beyond what a single in-memory pass here
    # supports

    t0 = time.perf_counter()
    flows = {}
    packets_scanned = 0
    with PcapReader(str(source_pcap)) as reader:
        for i, pkt in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            if IP not in pkt:
                continue
            key = flow_key(pkt)
            flows.setdefault(key, []).append(pkt)
            packets_scanned += 1
    elapsed = time.perf_counter() - t0

    rate = packets_scanned / elapsed if elapsed > 0 else 0.0
    print(f"   group_flows: scanned {packets_scanned:,} packets -> "
          f"{len(flows):,} flows in {elapsed:.3f}s ({rate:,.0f} pkts/sec)")

    return flows

def sample_pcap(source_pcap, expected_label, out_dir, n_flows=150,
                 priority="normal", max_packets=2_000_000,
                 random_state=RANDOM_STATE):
    # Samples up to n_flows complete flows from source_pcap and writes them, sorted by original relative timestam
    t_total_start = time.perf_counter()

    source_pcap = Path(source_pcap)
    print(f"[FILE] {source_pcap.name}  (expected_label={expected_label})")

    t_group_start = time.perf_counter()
    flows = group_flows(source_pcap, max_packets=max_packets)
    t_group_elapsed = time.perf_counter() - t_group_start

    available = len(flows)
    if available == 0:
        print(f"[WARN] no IP flows found in {source_pcap}")
        return None

    t_sample_start = time.perf_counter()
    rng = random.Random(random_state)
    keys = list(flows.keys())
    chosen = rng.sample(keys, k=min(n_flows, available))

    packets = [pkt for key in chosen for pkt in flows[key]]
    packets.sort(key=lambda p: float(p.time))

    if packets:
        t0 = float(packets[0].time)
        for p in packets:
            p.time = float(p.time) - t0
    t_sample_elapsed = time.perf_counter() - t_sample_start

    out_dir = Path(out_dir) / expected_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_pcap.stem}_sample.pcap"

    t_write_start = time.perf_counter()
    wrpcap(str(out_path), packets)
    t_write_elapsed = time.perf_counter() - t_write_start

    t_total_elapsed = time.perf_counter() - t_total_start
    sampling_ratio = (len(chosen) / available * 100) if available else 0.0

    print(f"{source_pcap.name}: sampled {len(chosen)}/{available} flows "
          f"({len(packets)} packets) - {out_path}")
    print(f"   Flows available     : {available:,}")
    print(f"   Flows sampled       : {len(chosen):,}  ({sampling_ratio:.2f}% of available)")
    print(f"   Packets in sample   : {len(packets):,}")
    print(f"   Time (grouping)     : {t_group_elapsed:.3f}s")
    print(f"   Time (sampling)     : {t_sample_elapsed:.3f}s")
    print(f"   Time (writing pcap) : {t_write_elapsed:.3f}s")
    print(f"   Time (TOTAL, file)  : {t_total_elapsed:.3f}s")
    print("-" * 60)

    return {
        "pcap_path": str(out_path),
        "expected_label": expected_label,
        "num_flows_sampled": len(chosen),
        "num_flows_available": available,
        "num_packets": len(packets),
        "priority": priority,
        "elapsed_seconds": t_total_elapsed,
        "group_flows_seconds": t_group_elapsed,
        "sample_seconds": t_sample_elapsed,
        "write_seconds": t_write_elapsed,
    }


def sample_directory(source_root, out_dir, n_flows_per_pcap=150, priority_map=None):
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
        )
        if rec:
            records.append(rec)

    t_dir_elapsed = time.perf_counter() - t_dir_start

    total_available = sum(r["num_flows_available"] for r in records)
    total_sampled = sum(r["num_flows_sampled"] for r in records)
    total_packets = sum(r["num_packets"] for r in records)
    overall_ratio = (total_sampled / total_available * 100) if total_available else 0.0
    avg_time_per_file = (t_dir_elapsed / len(records)) if records else 0.0

    print("-" * 60)
    print(" EXPERIMENT SUMMARY")
    print("-" * 60)
    print(f" Source PCAPs processed      : {len(records)}")
    print(f" Total flows available       : {total_available:,}")
    print(f" Total flows sampled         : {total_sampled:,}")
    print(f" Overall flow sampling ratio : {overall_ratio:.2f}%")
    print(f" Total packets in samples    : {total_packets:,}")
    print(f" Total wall-clock time       : {t_dir_elapsed:.3f}s")
    print(f" Average time per PCAP       : {avg_time_per_file:.3f}s")
    print("-" * 60)

    return records




def build_manifest(sample_records, out_path=OUT_PATH):

    # Assigns a sequential replay_id in the given order — reorder sample_records yourself beforehand if you want a specific run order.
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
        json.dump(manifest, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"Wrote {len(manifest)} replay jobs to {out_path}  ({elapsed:.3f}s)")
    return manifest


def load_manifest(path=OUT_PATH):
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    t_run_start = time.perf_counter()

    records = sample_directory(SOURCE_ROOT, OUT_DIR_PCAP, n_flows_per_pcap=150)
    build_manifest(records, OUT_PATH)

    t_run_elapsed = time.perf_counter() - t_run_start
    print(f"\nFull run Timing: {t_run_elapsed:.3f}s")