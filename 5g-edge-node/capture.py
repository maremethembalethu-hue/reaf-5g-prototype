import os
import sys
import time
import json
import logging
import signal
import subprocess
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

from scapy.all import sniff, IP, UDP, Ether, IPv6

from fragment_reassembly import FragmentReassembler
from envelope_ressemble import EnvelopeReassembler
from flow_builder import WindowBuilder
from model_engine import classify_flow, register_w100_result
from trigger import should_acquire
from evidence_collect import collect_evidence
from prediction_log import log_prediction
from pkt_debug import inspect_packet
from log_buffer import recent_log_handler

sys.stdout.reconfigure(line_buffering=True)

for _name in ("scapy.runtime", "scapy.interactive", "scapy.loading"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_LINKTYPE_DECODERS = {1: Ether, 101: IP, 228: IP, 229: IPv6}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE-NODE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger().addHandler(recent_log_handler)
log = logging.getLogger(__name__)

# Config
UE_SUBNET = os.getenv("UE_SUBNET", "192.168.100.0/24")
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "/evidence")
EVAL_DIR = Path(os.getenv("EVAL_DIR", "/evaluation"))
UE_PREFIX = ".".join(UE_SUBNET.split(".")[:3])
REAF_PORT = int(os.getenv("REAF_PORT", "9999"))

EVAL_DIR.mkdir(parents=True, exist_ok=True)
Path(EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)
ITEMS_LOG = os.path.join(EVAL_DIR, "items_log.jsonl")
TIMING_LOG = os.path.join(EVAL_DIR, "timing_log.jsonl")
RESOURCES_LOG = os.path.join(EVIDENCE_DIR, "resources.log")
COUNTERS_FILE = os.path.join(EVAL_DIR, "capture_counters.json")
TCPDUMP_STATS_FILE = os.path.join(EVAL_DIR, "tcpdump_stats.txt")
FLUSH_EVERY = 500

window_builder_10 = WindowBuilder(window_size=10)
window_builder_100 = WindowBuilder(window_size=100)
_total_packets_processed = 0
reassembler = FragmentReassembler()
envelope_reassembler = EnvelopeReassembler()


# Packet funnel: where each replayed packet stops between the sender and a classified window.
# totals: whole run. by_flow: per replay job, keyed by the flow_id carried in the envelope.
totals = Counter()
by_flow = {}
_tcpdump = None
_bumps = 0


def flush_counters():
    payload = {"written_ts": time.time(), "totals": dict(totals),
               "by_flow": {str(k): dict(v) for k, v in by_flow.items()}}
    tmp = COUNTERS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, COUNTERS_FILE)
    if _tcpdump is not None:
        try:
            _tcpdump.send_signal(signal.SIGUSR1)  # tcpdump prints captured/dropped counts to stderr
        except Exception:
            pass


def bump(name, flow_id=None):
    global _bumps
    totals[name] += 1
    if flow_id is not None:
        by_flow.setdefault(flow_id, Counter())[name] += 1
    _bumps += 1
    if _bumps % FLUSH_EVERY == 0:
        flush_counters()


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def write_items(window, result, captured_ts, incident_id=None):
    _append_jsonl(ITEMS_LOG, {
        "captured_ts": captured_ts,
        "flow_id": window.get("flow_id"),
        "mixed_flow": window.get("mixed_flow", False),
        "packet_ids": window.get("packet_ids", []),
        "packet_count": window["packet_count"],
        "predicted_label": result["attack_type"],
        "confidence": result["confidence"],
        "model_used": result["model_used"],
        "incident_id": incident_id,
    })
    if window.get("mixed_flow"):
        log.warning("Window straddled more than one flow_id")


def write_resources(result):
    # One CPU/RAM sample per classified window, taken from the classifier's own measurement.
    _append_jsonl(RESOURCES_LOG, {
        "timestamp": time.time(),
        "cpu": result.get("cpu_percent"),
        "ram": result.get("ram_percent"),
    })


def write_timing(flow_id, attack_type, detection_ts, evidence_start_ts, evidence_complete_ts,
                 incident_id, extra=None):
    # Called only where detection turns into evidence acquisition, the only point both timestamps exist.
    # detection_ts is taken when the window is complete, before classification starts.
    record = {
        "flow_id": flow_id, "attack_type": attack_type, "incident_id": incident_id,
        "detection_ts": detection_ts, "evidence_start_ts": evidence_start_ts,
        "evidence_complete_ts": evidence_complete_ts,
        "detection_to_evidence_ms": round((evidence_complete_ts - evidence_start_ts) * 1000, 2),
    }
    record.update(extra or {})
    _append_jsonl(TIMING_LOG, record)


def _insufficient_data_result():
    return {
        "label": "[?? PARTIAL ]", "attack_type": "InsufficientData", "confidence": 0.0,
        "model_used": "none", "cpu_percent": 0.0, "ram_percent": 0.0, "pred_class": "InsufficientData",
    }


def _acquire_if_triggered(window, result, ts, detection_ts, classification_done_ts):
    # Shared by the 10-packet and 100-packet handlers. Returns the incident id or None.
    if not should_acquire(result["attack_type"], result["confidence"]):
        return None
    trigger_done_ts = time.time()
    stage_ts = {}
    evidence_start_ts = time.time()
    try:
        incident_id = collect_evidence(window["packets"], result, ts, stage_ts)
    except Exception as e:
        log.error(f"Evidence acquisition failed: {e}")
        return None
    evidence_complete_ts = time.time()
    write_timing(
        window.get("flow_id"), result["attack_type"],
        detection_ts, evidence_start_ts, evidence_complete_ts, incident_id,
        extra={
            "window_first_packet_ts": window["start_time"],
            "window_complete_ts": window["last_time"],
            "classification_done_ts": classification_done_ts,
            "trigger_done_ts": trigger_done_ts,
            **stage_ts,
        },
    )
    return incident_id


def _handle_window(window, classify, tag, position):
    expected = window_builder_100.window_size if tag == "w100" else window_builder_10.window_size
    if window["packet_count"] != expected:
        # Stale or shutdown partial window: not a genuine sample at either trained scale.
        write_items(window, _insufficient_data_result(), datetime.now(timezone.utc).isoformat())
        return

    detection_ts = time.time()
    result = classify(window, position)
    classification_done_ts = time.time()
    log_prediction(window, result)
    write_resources(result)

    ts = datetime.now(timezone.utc).isoformat()
    duration = window["last_time"] - window["start_time"]
    prefix = "[w100] " if tag == "w100" else ""
    log.info(
        f"{result['label']} {ts} | {prefix}window of {window['packet_count']} pkts over {duration:.2f}s | "
        f" attack={result['attack_type']} | conf={result['confidence']:.3f} | model={result['model_used']}"
    )

    incident_id = _acquire_if_triggered(window, result, ts, detection_ts, classification_done_ts)
    write_items(window, result, ts, incident_id)


def handle_finished_flow(window, position=None):
    _handle_window(window, lambda w, p: classify_flow(w, position=p), "w10", position)


def handle_finished_w100(window, position):
    # w=100 windows never gate on their own; they exist to override the w=10 stage-2 answer for Flood/Mirai.
    _handle_window(window, register_w100_result, "w100", position)


def process_original_packet(pkt):
    # Works on the DECAPSULATED original packet.
    global _total_packets_processed

    flow_id = getattr(pkt, "flow_id", None)
    pkt = reassembler.feed(pkt, ts=time.time())
    if pkt is None:
        bump("held_fragment", flow_id)
        return  # mid-train fragment or incomplete set: wait or drop
    inspect_packet(pkt, "capture")
    ts = float(pkt.time)

    _total_packets_processed += 1
    position = _total_packets_processed
    bump("processed", flow_id)

    # Both builders see every packet. position is the shared packet count, used to line up
    # which w=10 windows fall inside which w=100 window for the override check in classify_flow.
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
    # What arrives here is the OUTER carrier packet; it must be decapsulated before anything downstream.
    if IP not in pkt or UDP not in pkt:
        return
    if pkt[UDP].sport != REAF_PORT and pkt[UDP].dport != REAF_PORT:
        bump("non_reaf_udp")
        return
    bump("udp_chunks")

    envelope = envelope_reassembler.feed(bytes(pkt[UDP].payload), ts=time.time())
    if envelope is None:
        return

    bump("envelopes", envelope["flow_id"])
    try:
        decoder = _LINKTYPE_DECODERS.get(envelope["linktype"], Ether)
        original_pkt = decoder(envelope["payload"])
    except Exception as e:
        log.warning(f"Failed to decode decapsulated packet: {e}")
        bump("decode_failed", envelope["flow_id"])
        return

    if IP not in original_pkt:
        bump("non_ip_dropped", envelope["flow_id"])  # ARP, IPv6 and other non-IPv4 packets stop here
        return

    original_pkt.time = envelope["timestamp"]
    original_pkt.flow_id = envelope["flow_id"]
    original_pkt.packet_id = envelope["packet_id"]
    original_pkt.linktype = envelope["linktype"]

    process_original_packet(original_pkt)
    envelope_reassembler.expire_stale()


def main():
    log.info("REAF-5G Edge Node: Packet Capture")
    log.info("Method    : tcpdump pipe -> Scapy")
    log.info(f"UE filter : {UE_SUBNET}  (prefix: {UE_PREFIX}.*)")
    log.info(f"Evidence  : {EVIDENCE_DIR}")
    log.info(f"Window    : {window_builder_10.window_size} packets (stage1/stage2 fast path) "
             f"+ {window_builder_100.window_size} packets (stage2 Flood/Mirai override, dual-window)")

    for subdir in ["packets", "memory", "processes", "syslogs"]:
        os.makedirs(os.path.join(EVIDENCE_DIR, subdir), exist_ok=True)

    log.info("Starting tcpdump capture pipe...")
    global _tcpdump
    stats_file = open(TCPDUMP_STATS_FILE, "w")
    tcpdump = _tcpdump = subprocess.Popen(
        ["tcpdump", "-i", "ogstun", "-n", "-U", "-B", "65536", "-w", "-",
         f"src net {UE_PREFIX}.0/24 or dst net {UE_PREFIX}.0/24"],
        stdout=subprocess.PIPE,
        stderr=stats_file,
    )

    log.info("Waiting for traffic...")
    try:
        sniff(offline=tcpdump.stdout, prn=on_packet, store=False)
    except KeyboardInterrupt:
        log.info("Stopping capture...")
    finally:
        trailing_10 = window_builder_10.flush()
        if trailing_10 is not None:
            handle_finished_flow(trailing_10, position=_total_packets_processed)
        trailing_100 = window_builder_100.flush()
        if trailing_100 is not None:
            handle_finished_w100(trailing_100, position=_total_packets_processed)
        flush_counters()
        tcpdump.terminate()


if __name__ == "__main__":
    main()
