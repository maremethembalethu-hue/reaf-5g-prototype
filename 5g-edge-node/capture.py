
# REAF-5G Edge Node Stage 1 Real-time packet capture inside the UPF network namespace Capture-py.

import os
import sys
import time
import logging
import subprocess
from datetime import datetime, timezone
 
from scapy.all import sniff, IP
from fragment_reassembly import FragmentReassembler
 
from flow_builder import WindowBuilder
from model_engine import classify_flow
from trigger import should_acquire
from evidence_collect import collect_evidence
from prediction_log import log_prediction
 
 
sys.stdout.reconfigure(line_buffering=True)
 
import logging as _log
_log.getLogger("scapy.runtime").setLevel(_log.ERROR)
_log.getLogger("scapy.interactive").setLevel(_log.ERROR)
_log.getLogger("scapy.loading").setLevel(_log.ERROR)
 
 
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE-NODE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
log = logging.getLogger(__name__)
 
# Config
UE_SUBNET    = os.getenv("UE_SUBNET", "192.168.100.0/24")
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "/evidence")
UE_PREFIX    = ".".join(UE_SUBNET.split(".")[:3])
 
window_builder = WindowBuilder()
reassembler = FragmentReassembler()
 
 
def handle_finished_flow(window):
    result = classify_flow(window)
    log_prediction(window, result)
 
    ts = datetime.now(timezone.utc).isoformat()
    n_pkts = window["packet_count"]
    duration = window["last_time"] - window["start_time"]
    log.info(
        f"{result['label']} {ts} | window of {n_pkts} pkts over {duration:.2f}s | "
        f"proto={window['proto']} | attack={result['attack_type']} | "
        f"conf={result['confidence']:.3f} | model={result['model_used']}"
    )
 
    if should_acquire(result["attack_type"], result["confidence"]):
        collect_evidence(window["packets"], result, ts)
 
 
def on_packet(pkt):
    if IP not in pkt:
        return
 
    pkt = reassembler.feed(pkt, ts=time.time())
    if pkt is None:
        return  # mid-train fragment, or an incomplete set — wait or drop
 
    finished = window_builder.add_packet(pkt, ts=time.time())
    if finished is not None:
        handle_finished_flow(finished)
 
    # Windows complete purely by packet count, so
    # there's no per-flow idle-timeout eviction anymore just a single
    # global trailing partial window to flush during a in traffic.
    stale_window = window_builder.expire_stale_partial_window()
    if stale_window is not None:
        handle_finished_flow(stale_window)
 
    reassembler.expire_stale()
 
 
# Main
def main():
 
    log.info("REAF-5G Edge Node: Packet Capture")
    log.info(f"Method    : tcpdump pipe -> Scapy")
    log.info(f"UE filter : {UE_SUBNET}  (prefix: {UE_PREFIX}.*)")
    log.info(f"Evidence  : {EVIDENCE_DIR}")
    log.info(f"Window    : {window_builder.window_size} packets "
             f"(idle flush after {window_builder.idle_flush_seconds}s)")
 
    for subdir in ["packets", "memory", "processes", "syslogs"]:
        os.makedirs(os.path.join(EVIDENCE_DIR, subdir), exist_ok=True)
 
    log.info("Starting tcpdump capture pipe...")
 
    # tcpdump -i any: capture on ALL interfaces including ogstun
    # -n: do not resolve hostnames
    # -U: packet-buffered output
    # -w: write raw pcap to stdout
    # host 192.168.100: BPF filter: only UE subnet packets
    # 2>/dev/null: suppress tcpdump startup messages
    tcpdump = subprocess.Popen(
        [
            "tcpdump",
            "-i", "any",
            "-n",
            "-U",
            "-w", "-",
            f"src net {UE_PREFIX}.0/24 or dst net {UE_PREFIX}.0/24"  # Explicit directionality
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
 
    log.info("Waiting for traffic...")
 
    try:
        # sniff reads raw pcap bytes from the pipe; Scapy parses each
        # packet and calls on_packet(). This runs forever until tcpdump
        # exits or is killed.
        sniff(
            offline=tcpdump.stdout,
            prn=on_packet,
            store=False
        )
    except KeyboardInterrupt:
        log.info("Stopping capture...")
    finally:
        # Flush whatever's left in the current window rather than silently
        # dropping the last few packets on shutdown.
        trailing_window = window_builder.flush()
        if trailing_window is not None:
            handle_finished_flow(trailing_window)
        tcpdump.terminate()
 
 
if __name__ == "__main__":
    main()