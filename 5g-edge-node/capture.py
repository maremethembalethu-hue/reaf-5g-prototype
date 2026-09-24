
import os
import sys
import time
import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
 
from scapy.all import sniff, IP, UDP, ARP, Ether, IPv6
from fragment_reassembly import FragmentReassembler
 
#from flow_builder import WindowBuilder
from model_engine import classify_flow, register_w100_result
from trigger import should_acquire
from evidence_collect import collect_evidence
from prediction_log import log_prediction
from pkt_debug import inspect_packet 
from envelope_ressemble import EnvelopeReassembler
from flow_builder import WindowBuilder
sys.stdout.reconfigure(line_buffering=True)
from log_buffer import recent_log_handler

import logging as _log
_log.getLogger("scapy.runtime").setLevel(_log.ERROR)
_log.getLogger("scapy.interactive").setLevel(_log.ERROR)
_log.getLogger("scapy.loading").setLevel(_log.ERROR)
 

 
_LINKTYPE_DECODERS = {1: Ether, 101: IP, 228: IP, 229: IPv6}  
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE-NODE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)


logging.getLogger().addHandler(recent_log_handler)
log = logging.getLogger(__name__)
 
# Config
UE_SUBNET    = os.getenv("UE_SUBNET", "192.168.100.0/24")
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "/evidence")
EVAL_DIR = Path(os.getenv("EVAL_DIR", "/evaluation"))
UE_PREFIX    = ".".join(UE_SUBNET.split(".")[:3])
REAF_PORT = int(os.getenv("REAF_PORT", "9999"))
 
EVAL_DIR.mkdir(parents=True, exist_ok=True)
ITEMS_LOG = os.path.join(EVAL_DIR, "items_log.jsonl")
TIMING_LOG = os.path.join(EVAL_DIR, "timing_log.jsonl") 
 
window_builder_10 = WindowBuilder(window_size=10)
window_builder_100 = WindowBuilder(window_size=100)
_total_packets_processed = 0
reassembler = FragmentReassembler()
envelope_reassembler = EnvelopeReassembler()
 
def write_items(window, result, captured_ts, incident_id=None):
  
    record = {
        "captured_ts": captured_ts,
        "flow_id": window.get("flow_id"),
        "mixed_flow": window.get("mixed_flow", False),
        "packet_ids": window.get("packet_ids", []),
        "packet_count": window["packet_count"],
        "predicted_label": result["attack_type"],
        "confidence": result["confidence"],
        "model_used": result["model_used"],
        "incident_id": incident_id,
    }
    with open(ITEMS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
 
    if window.get("mixed_flow"):
        log.warning(f"Window straddled more than one flow_id ")
 
 
def _insufficient_data_result():
    return {
        "label": "[?? PARTIAL ]", "attack_type": "InsufficientData", "confidence": 0.0,
        "model_used": "none", "cpu_percent": 0.0, "ram_percent": 0.0, "pred_class": "InsufficientData",
    }

def write_timing(flow_id, attack_type, detection_ts, evidence_start_ts, evidence_complete_ts, incident_id):
    # This is called from exactly the two places detection turns into evidence
    # acquisition, never anywhere else, since that's the only point both timestamps exist.
    record = {
        "flow_id": flow_id, "attack_type": attack_type, "incident_id": incident_id,
        "detection_ts": detection_ts, "evidence_start_ts": evidence_start_ts,
        "evidence_complete_ts": evidence_complete_ts,
        "detection_to_evidence_ms": round((evidence_complete_ts - evidence_start_ts) * 1000, 2),
    }
    with open(TIMING_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
 
def handle_finished_flow(window, position=None):
    if window["packet_count"] != window_builder_10.window_size:
        # Stale/shutdown partial window not a genuine 10-packet sample,
        # matches neither trained scale. 
        result = _insufficient_data_result()
        write_items(window, result, datetime.now(timezone.utc).isoformat())
        return
    detection_ts = time.time()  
    result = classify_flow(window, position=position)
    log_prediction(window, result)
 
    ts = datetime.now(timezone.utc).isoformat()
    
    n_pkts = window["packet_count"]
    duration = window["last_time"] - window["start_time"]
    log.info(
        f"{result['label']} {ts} | window of {n_pkts} pkts over {duration:.2f}s | "
        f" attack={result['attack_type']} | "
        f"conf={result['confidence']:.3f} | model={result['model_used']}"
    )
    write_items(window, result, ts)
    incident_id = None
    if should_acquire(result["attack_type"], result["confidence"]):
        evidence_start_ts = time.time()                      
        incident_id = collect_evidence(window["packets"], result, ts)
        evidence_complete_ts = time.time()                   
        write_timing(window.get("flow_id"), result["attack_type"],
                     detection_ts, evidence_start_ts, evidence_complete_ts, incident_id)  
    
    write_items(window, result, ts, incident_id)
 
 
def handle_finished_w100(window, position):
    # w=100 windows never gate on their own they only ever EXIST to
    # override the w=10 stage2 answer for Flood/Mirai.
    if window["packet_count"] != window_builder_100.window_size:
        result = _insufficient_data_result()
        write_items(window, result, datetime.now(timezone.utc).isoformat())
        return
    
    detection_ts = time.time()
    result = register_w100_result(window, position)
    log_prediction(window, result)
    ts = datetime.now(timezone.utc).isoformat()
    n_pkts = window["packet_count"]
    duration = window["last_time"] - window["start_time"]
    log.info(
        f"{result['label']} {ts} | [w100] window of {n_pkts} pkts over {duration:.2f}s | "
        f" attack={result['attack_type']} | "
        f"conf={result['confidence']:.3f} | model={result['model_used']}"
    )
    write_items(window, result, ts)
    if should_acquire(result["attack_type"], result["confidence"]):
        evidence_start_ts = time.time()                     
        incident_id = collect_evidence(window["packets"], result, ts)
        evidence_complete_ts = time.time()                   
        write_timing(window.get("flow_id"), result["attack_type"],
                     detection_ts, evidence_start_ts, evidence_complete_ts, incident_id)
 
def process_original_packet(pkt):
    # DECAPSULATED original packet rather than directly on whatever
    global _total_packets_processed
 
    pkt = reassembler.feed(pkt, ts=time.time())
    if pkt is None:
        return  # mid-train fragment, or an incomplete set, wait or drop
    inspect_packet(pkt, "capture")
    ts = float(pkt.time)
 
    _total_packets_processed += 1
    position = _total_packets_processed
 
    # Both builders see EVERY packet, independently accumulating toward
    # their own window_size. position is the shared, 1:1-synchronized packet
    # count both builders were fed up to, used to line up which w=10 windows
    # fall inside which w=100 window for the override check in classify_flow.
    finished_10 = window_builder_10.add_packet(pkt, ts=ts)
    if finished_10 is not None:
        handle_finished_flow(finished_10, position=position)
 
    finished_100 = window_builder_100.add_packet(pkt, ts=ts)
    if finished_100 is not None:
        handle_finished_w100(finished_100, position=position)
 
    stale_10 = window_builder_10.expire_stale_partial_window()
    if stale_10 is not None:
        handle_finished_flow(stale_10, position=position)
 
    stale_100 = window_builder_100.expire_stale_partial_window()
    if stale_100 is not None:
        handle_finished_w100(stale_100, position=position)
 
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
        decoder = _LINKTYPE_DECODERS.get(envelope["linktype"], Ether)
        original_pkt = decoder(envelope["payload"])
    except Exception as e:
        log.warning(f"Failed to decode decapsulated packet: {e}")
        return
 
    if IP not in original_pkt:
        return
 
    original_pkt.time = envelope["timestamp"]
    original_pkt.flow_id = envelope["flow_id"]
    original_pkt.packet_id = envelope["packet_id"]
    original_pkt.linktype = envelope["linktype"]
 
    process_original_packet(original_pkt)
 
    envelope_reassembler.expire_stale()
 
 
# Main
def main():
 
    log.info("REAF-5G Edge Node: Packet Capture")
    log.info(f"Method    : tcpdump pipe -> Scapy")
    log.info(f"UE filter : {UE_SUBNET}  (prefix: {UE_PREFIX}.*)")
    log.info(f"Evidence  : {EVIDENCE_DIR}")
    log.info(f"Window    : {window_builder_10.window_size} packets (stage1/stage2 fast path) "
             f"+ {window_builder_100.window_size} packets (stage2 Flood/Mirai override, dual-window)")
 
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
        # Flush whatever's left in either window
        trailing_10 = window_builder_10.flush()
        if trailing_10 is not None:
            handle_finished_flow(trailing_10, position=_total_packets_processed)
        trailing_100 = window_builder_100.flush()
        if trailing_100 is not None:
            handle_finished_w100(trailing_100, position=_total_packets_processed)
        tcpdump.terminate()
 
 
if __name__ == "__main__":
    main()