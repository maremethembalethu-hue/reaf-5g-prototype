# traffic_generator.py — REAF-5G IoT Traffic Generator / Orchestrator
import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(line_buffering=True)

import logging as _log
_log.getLogger("scapy.runtime").setLevel(_log.ERROR)
_log.getLogger("scapy.interactive").setLevel(_log.ERROR)
_log.getLogger("scapy.loading").setLevel(_log.ERROR)

from scapy.all import IP, TCP, UDP, ICMP, send
from replay_engine import replay_pcap, replay_mixed, replay_random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAFFIC-GEN] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
log = logging.getLogger(__name__)

IFACE        = os.getenv("UE_TUNNEL_IFACE", "uesimtun0")
TARGET_IP    = os.getenv("TARGET_IP", "192.168.100.1")
INTERVAL     = float(os.getenv("SEND_INTERVAL", "2"))
MODE         = os.getenv("MODE", "synthetic")
PCAP_PATH    = os.getenv("PCAP_PATH", "pcaps/recon/Recon-PortScan.pcap")
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "1.0"))

MIXED_SEQUENCE = [
    "pcaps/benign/benign.pcap",
    "pcaps/recon/Recon-PortScan.pcap",
    "pcaps/ddos/DDOS-TCP_Flood.pcap",
    "pcaps/benign/benign.pcap",
]

# MODE=manifest config. manifest.json and pcaps/ are both baked into this
# image's own /app folder by flow_sampler.py's output paths — no separate
# evaluation/ folder needs to be mounted into this container.
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "manifest.json")
GROUND_TRUTH_LOG = os.getenv("GROUND_TRUTH_LOG", "ground_truth_log.jsonl")
# Must be >= 5g-edge-node/flow_builder.py's IDLE_TIMEOUT_SECONDS plus a
# safety margin, or the next job's traffic can arrive before the previous
# job's last flow has gone idle at the live capture side (see doc 12).
SCHEDULER_DRAIN_SECONDS = float(os.getenv("SCHEDULER_DRAIN_SECONDS", "10.0"))


def get_ue_ip():
    result = subprocess.run(["ip", "addr", "show", IFACE], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "inet " in line and "inet6" not in line:
            return line.strip().split()[1].split("/")[0]
    return None


def tx(packet, msg):
    send(packet, iface=IFACE, verbose=False)
    log.info(msg)


def normal_ping(src):
    tx(IP(src=src, dst="8.8.8.8") / ICMP(), f"NORMAL | ICMP ping  | {src} -> 8.8.8.8")

def normal_http(src):
    tx(IP(src=src, dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="S"), f"NORMAL | HTTP SYN | {src}:12345 -> 8.8.8.8:80")

def normal_dns(src):
    tx(IP(src=src, dst="8.8.8.8") / UDP(sport=54321, dport=53), f"NORMAL | DNS query | {src}:54321 -> 8.8.8.8:53")

def attack_portscan(src):
    for port in [21, 22, 23, 25, 80, 443, 3306, 8080, 8443, 9999]:
        tx(IP(src=src, dst=TARGET_IP) / TCP(sport=11111, dport=port, flags="S"), f"ATTACK | SYN scan | {src} -> {TARGET_IP}:{port}")
        time.sleep(0.1)

def attack_ddos(src):
    for i in range(20):
        tx(IP(src=src, dst=TARGET_IP) / UDP(sport=i + 1000, dport=80), f"ATTACK | UDP flood | {src}:{i+1000} -> {TARGET_IP}:80")
        time.sleep(0.02)

def run_synthetic(src):
    cycle = 0
    while True:
        cycle += 1
        normal_ping(src); time.sleep(INTERVAL)
        normal_http(src); time.sleep(INTERVAL)
        normal_dns(src); time.sleep(INTERVAL)
        if cycle % 5 == 0:
            attack_portscan(src); time.sleep(INTERVAL)
            attack_ddos(src)
        time.sleep(INTERVAL)


def _priority_key(job):
    order = {"high": 0, "normal": 1, "low": 2}
    return order.get(job.get("priority", "normal"), 1)


def run_manifest(manifest_path=MANIFEST_PATH, replay_speed=REPLAY_SPEED):
    # replay every job in manifest.json in order, waiting
    # SCHEDULER_DRAIN_SECONDS between jobs so flows from different scenarios
    # never merge at the live capture side. Runs once, then returns unlike run_synthetic(), this does not loop forever.
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest = sorted(manifest, key=_priority_key)

    for job in manifest:
        log.info(f"--- Replay {job['replay_id']}: {job['pcap_path']} "
                 f"(expected={job['expected_label']}, flows={job['num_flows']}) ---")

        start_ts = datetime.now(timezone.utc).isoformat()
        start_time = time.time()

        replay_pcap(job["pcap_path"], replay_speed=replay_speed, target_ip=TARGET_IP, iface=IFACE)

        log.info(f"Replay {job['replay_id']} sent; waiting {SCHEDULER_DRAIN_SECONDS}s "
                 f"for the flow table to drain before the next job...")
        time.sleep(SCHEDULER_DRAIN_SECONDS)

        end_ts = datetime.now(timezone.utc).isoformat()
        end_time = time.time()

        record = {
            "replay_id": job["replay_id"], "pcap_path": job["pcap_path"],
            "expected_label": job["expected_label"], "num_flows_sent": job["num_flows"],
            "start_ts": start_ts, "end_ts": end_ts,
            "start_time": start_time, "end_time": end_time,
        }
        with open(GROUND_TRUTH_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    log.info(f"All {len(manifest)} replay jobs complete. Ground truth: {GROUND_TRUTH_LOG}")


def main():
    log.info(f"Mode: {MODE}")
    src = None
    while src is None:
        src = get_ue_ip()
        if src is None:
            time.sleep(5)

    if MODE == "synthetic":
        run_synthetic(src)
    elif MODE == "replay":
        replay_pcap(Path(PCAP_PATH), replay_speed=REPLAY_SPEED, target_ip=TARGET_IP, iface=IFACE, ue_ip=src)
    elif MODE == "mixed":
        replay_mixed(MIXED_SEQUENCE, replay_speed=REPLAY_SPEED, target_ip=TARGET_IP, iface=IFACE)
    elif MODE == "random":
        replay_random(replay_speed=REPLAY_SPEED, target_ip=TARGET_IP, iface=IFACE)
    elif MODE == "manifest":
        run_manifest(replay_speed=REPLAY_SPEED)
    else:
        log.error(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()