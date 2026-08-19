
# REAF-5G Edge Node Real-time packet capture inside the UPF network namespace Capture-py.

import os
import sys
import time
import logging
import json
import subprocess
from datetime import datetime, timezone
 
from scapy.all import sniff, IP, UDP
from fragment_reassembly import FragmentReassembler
 
from flow_builder import WindowBuilder
from model_engine import classify_flow
from trigger import should_acquire
from evidence_collect import collect_evidence
from prediction_log import log_prediction
from pkt_debug import inspect_packet 
from envelope_ressemble import EnvelopeReassembler
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
EVAL_DIR = os.getenv("EVAL_DIR", "/evaluation")
UE_PREFIX    = ".".join(UE_SUBNET.split(".")[:3])
REAF_PORT = int(os.getenv("REAF_PORT", "9999"))

EVAL_DIR.mkdir(parents=True, exist_ok=True)
ITEMS_LOG = EVAL_DIR / "items_log.jsonl"
 
window_builder = WindowBuilder()
reassembler = FragmentReassembler()
envelope_reassembler = EnvelopeReassembler()  

def write_items(window, result, captured_ts):
  
    record = {
        "captured_ts": captured_ts,
        "flow_id": window.get("flow_id"),
        "mixed_flow": window.get("mixed_flow", False),
        "packet_ids": window.get("packet_ids", []),
        "packet_count": window["packet_count"],
        "predicted_label": result["attack_type"],
        "confidence": result["confidence"],
        "model_used": result["model_used"],
    }
    with open(ITEMS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()

    if window.get("mixed_flow"):
        log.warning(f"Window straddled more than one flow_id ")
 
def handle_finished_flow(window):
    result = classify_flow(window)
    log_prediction(window, result)
 
 
    #log.info(f"LIVE_FEATURES {json.dumps(window['features'], default=str)}")
    ts = datetime.now(timezone.utc).isoformat()
    n_pkts = window["packet_count"]
    duration = window["last_time"] - window["start_time"]
    log.info(
        f"{result['label']} {ts} | window of {n_pkts} pkts over {duration:.2f}s | "
        f"proto={window['proto']} | attack={result['attack_type']} | "
        f"conf={result['confidence']:.3f} | model={result['model_used']}"
    )
    write_items(window, result, ts)
    if should_acquire(result["attack_type"], result["confidence"]):
        collect_evidence(window["packets"], result, ts)
        
 
 

def process_original_packet(pkt):
    # DECAPSULATED original packet rather than directly on whatever

    pkt = reassembler.feed(pkt, ts=time.time())
    if pkt is None:
        return  # mid-train fragment, or an incomplete set — wait or drop

    finished = window_builder.add_packet(pkt, ts=float(pkt.time))
    if finished is not None:
        handle_finished_flow(finished)

    # stale_window = window_builder.expire_stale_partial_window()
    # if stale_window is not None:
    #     handle_finished_flow(stale_window)

    reassembler.expire_stale()


def on_packet(pkt):
    # What actually arrives here is the OUTER carrier packet 
    # It must be decapsulated before anything downstream (feature
    # extraction included) ever sees it.
    if IP not in pkt or UDP not in pkt:
        return
    if pkt[UDP].sport != REAF_PORT and pkt[UDP].dport != REAF_PORT:
        return

    udp_payload = bytes(pkt[UDP].payload)
    envelope = envelope_reassembler.feed(udp_payload, ts=time.time())
    if envelope is None:
        return  

    try:
        original_pkt = IP(envelope["payload"])
    except Exception as e:
        log.warning(f"Failed to decode decapsulated packet: {e}")
        return

    if IP not in original_pkt:
        return

    # Recover the ORIGINAL pcap timestamp
    original_pkt.time = envelope["timestamp"]

    # Carry envelope identity forward. Scapy's Packet.__setattr__ falls
    original_pkt.flow_id = envelope["flow_id"]
    original_pkt.packet_id = envelope["packet_id"]

    process_original_packet(original_pkt)

    envelope_reassembler.expire_stale()
 
 
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

    tcpdump = subprocess.Popen(
        [
            "tcpdump",
            "-i", "ogstun",
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
        # packet and calls on_packet()
        sniff(
            offline=tcpdump.stdout,
            prn=on_packet,
            store=False
        )
    except KeyboardInterrupt:
        log.info("Stopping capture...")
    finally:
        # Flush whatever's left in the current window 
        trailing_window = window_builder.flush()
        if trailing_window is not None:
            handle_finished_flow(trailing_window)
        tcpdump.terminate()
 
 
if __name__ == "__main__":
    main()