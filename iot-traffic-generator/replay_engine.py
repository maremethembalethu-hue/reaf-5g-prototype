
# Loads a captured PCAP, rewrites its addresses onto the live Open5GS lab, reproduces the original inter-packet timing, and sends each packet out through uesimtun0.

import os
import time
import json
import random
import logging
import subprocess
from pathlib import Path

from scapy.all import rdpcap, send, IP

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
logging.getLogger("scapy.interactive").setLevel(logging.ERROR)
logging.getLogger("scapy.loading").setLevel(logging.ERROR)

log = logging.getLogger("TRAFFIC-GEN")

# Same defaults as traffic_generator.py, so both modules agree on the lab network
IFACE     = os.getenv("UE_TUNNEL_IFACE", "uesimtun0")
TARGET_IP = os.getenv("TARGET_IP", "192.168.100.1")
PCAP_DIR  = Path(os.getenv("PCAP_DIR", "pcaps"))


def get_ue_ip(iface=IFACE):
    # Same tunnel-IP lookup traffic_generator.py uses for synthetic mode.
    result = subprocess.run(["ip", "addr", "show", iface], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "inet " in line and "inet6" not in line:
            return line.strip().split()[1].split("/")[0]
    return None


def load_pcap(pcap_path):
    # Reads a PCAP and its optional metadata sidecar (same base name, .json extension).
    packets = rdpcap(str(pcap_path))
    meta_path = Path(str(pcap_path).rsplit(".", 1)[0] + ".json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    log.info(f"Loaded {len(packets)} packets from {pcap_path.name} "
             f"(expected_attack={metadata.get('expected_attack', 'unknown')})")
    return packets, metadata


def rewrite_packet(pkt, ue_ip, target_ip=TARGET_IP):
    # Drops non-IP layers (such as Ethernet, captured on a different network) and remaps src/dst onto the lab network, then forces checksum recalculation.
    if IP not in pkt:
        return None
    ip_pkt = pkt[IP].copy()
    ip_pkt.src = ue_ip
    ip_pkt.dst = target_ip
    del ip_pkt.chksum
    if hasattr(ip_pkt.payload, "chksum"):
        del ip_pkt.payload.chksum
    return ip_pkt


def replay_pcap(pcap_path, replay_speed=1.0, target_ip=TARGET_IP, iface=IFACE, ue_ip=None):
    # Replays one PCAP end-to-end: rewrite addresses, reproduce original timing, send.
    ue_ip = ue_ip or get_ue_ip(iface)
    if ue_ip is None:
        log.error(f"Could not resolve UE IP on {iface}; aborting replay")
        return

    packets, metadata = load_pcap(Path(pcap_path))
    if not packets:
        log.warning(f"No packets found in {pcap_path}")
        return

    log.info(f"REPLAY | {Path(pcap_path).name} | {ue_ip} -> {target_ip} | speed={replay_speed}x")
    prev_ts = float(packets[0].time)

    for pkt in packets:
        delay = (float(pkt.time) - prev_ts) / replay_speed
        if delay > 0:
            time.sleep(delay)
        prev_ts = float(pkt.time)

        out_pkt = rewrite_packet(pkt, ue_ip, target_ip)
        if out_pkt is None:
            continue
        send(out_pkt, iface=iface, verbose=False)
        log.info(f"REPLAY | {ue_ip} -> {target_ip} | {out_pkt.summary()}")

    log.info(f"REPLAY | {Path(pcap_path).name} complete "
             f"(expected_attack={metadata.get('expected_attack', 'unknown')})")


def replay_mixed(pcap_paths, replay_speed=1.0, gap=3.0, iface=IFACE, target_ip=TARGET_IP):
    # Replays a sequence of PCAPs back to back, e.g. [benign, recon, ddos, benign].
    ue_ip = get_ue_ip(iface)
    for p in pcap_paths:
        replay_pcap(p, replay_speed=replay_speed, target_ip=target_ip, iface=iface, ue_ip=ue_ip)
        time.sleep(gap)


def replay_random(pcap_dir=PCAP_DIR, replay_speed=1.0, iface=IFACE, target_ip=TARGET_IP):
    # Picks one random .pcap file anywhere under pcap_dir and replays it.
    candidates = list(Path(pcap_dir).rglob("*.pcap"))
    if not candidates:
        log.error(f"No .pcap files found under {pcap_dir}")
        return
    choice = random.choice(candidates)
    replay_pcap(choice, replay_speed=replay_speed, target_ip=target_ip, iface=iface)
