# Samples a fixed number of *complete* flows from a source PCAP, writing them
# to a new, much smaller PCAP that replay_engine.py can replay

import random
import json
from pathlib import Path

from scapy.utils import PcapReader, wrpcap
from scapy.all import IP, TCP, UDP

RANDOM_STATE = 42
SOURCE_ROOT = "./iot-traffic-generator/pcaps"
OUT_PATH = "evaluation/manifest.json"

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
    
    flows = {}
    with PcapReader(str(source_pcap)) as reader:
        for i, pkt in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            if IP not in pkt:
                continue
            key = flow_key(pkt)
            flows.setdefault(key, []).append(pkt)
    return flows

def sample_pcap(source_pcap, expected_label, out_dir, n_flows=150,
                 priority="normal", max_packets=2_000_000,
                 random_state=RANDOM_STATE):
    # Samples up to n_flows complete flows from source_pcap and writes them, sorted by original relative timestam
    source_pcap = Path(source_pcap)
    flows = group_flows(source_pcap, max_packets=max_packets)
    available = len(flows)
    if available == 0:
        print(f"[WARN] no IP flows found in {source_pcap}")
        return None

    rng = random.Random(random_state)
    keys = list(flows.keys())
    chosen = rng.sample(keys, k=min(n_flows, available))

    packets = [pkt for key in chosen for pkt in flows[key]]
    packets.sort(key=lambda p: float(p.time))

    if packets:
        t0 = float(packets[0].time)
        for p in packets:
            p.time = float(p.time) - t0

    out_dir = Path(out_dir) / expected_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_pcap.stem}_sample.pcap"
    wrpcap(str(out_path), packets)

    print(f"{source_pcap.name}: sampled {len(chosen)}/{available} flows "
          f"({len(packets)} packets) -> {out_path}")

    return {
        "pcap_path": str(out_path),
        "expected_label": expected_label,
        "num_flows_sampled": len(chosen),
        "num_flows_available": available,
        "num_packets": len(packets),
        "priority": priority,
    }


def sample_directory(source_root, out_dir, n_flows_per_pcap=150, priority_map=None):
   # Samples every .pcap found under each label subfolder of source_root.
    priority_map = priority_map or {}
    records = []
    for label_dir in sorted(Path(source_root).iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for pcap_path in sorted(label_dir.glob("*.pcap")):
            rec = sample_pcap(
                pcap_path, expected_label=label, out_dir=out_dir,
                n_flows=n_flows_per_pcap, priority=priority_map.get(label, "normal"),
            )
            if rec:
                records.append(rec)
    return records




def build_manifest(sample_records, out_path=OUT_PATH):

    # Assigns a sequential replay_id in the given order — reorder sample_records yourself beforehand if you want a specific run order.
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

    print(f"Wrote {len(manifest)} replay jobs to {out_path}")
    return manifest


def load_manifest(path=OUT_PATH):
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    records = sample_directory(SOURCE_ROOT, "evaluation/replay_pcaps", n_flows_per_pcap=150)
    build_manifest(records,OUT_PATH)