
# REAF-5G Edge Node Stage 1 Real-time packet capture inside the UPF network namespace.

import os
import sys
import time
import logging
import subprocess
from datetime import datetime, timezone

from scapy.all import sniff, IP

from flow_builder import FlowTable
from model_engine import classify_flow
from trigger import should_acquire
from evidence_collect import collect_evidence

sys.stdout.reconfigure(line_buffering=True)

import logging as _log
_log.getLogger("scapy.runtime").setLevel(_log.ERROR)
_log.getLogger("scapy.interactive").setLevel(_log.ERROR)
_log.getLogger("scapy.loading").setLevel(_log.ERROR)


#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE-NODE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
log = logging.getLogger(__name__)

#  Config 
UE_SUBNET    = os.getenv("UE_SUBNET", "192.168.100.0/24")
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "/evidence")
UE_PREFIX    = ".".join(UE_SUBNET.split(".")[:3])

flow_table = FlowTable()

def handle_finished_flow(flow):
    result = classify_flow(flow)

    ts = datetime.now(timezone.utc).isoformat()
    n_pkts = flow["packet_count"]
    duration = flow["last_time"] - flow["start_time"]
    log.info(
        f"{result['label']} {ts} | flow of {n_pkts} pkts over {duration:.2f}s | "
        f"proto={flow['proto']} | attack={result['attack_type']} | "
        f"conf={result['confidence']:.3f} | model={result['model_used']}"
    )

    if should_acquire(result["attack_type"], result["confidence"]):
        collect_evidence(flow["packets"], result, ts)


def on_packet(pkt):
    if IP not in pkt:
        return

    finished = flow_table.add_packet(pkt, ts=time.time())
    if finished is not None:
        handle_finished_flow(finished)

    # check every packet at lab-scale traffic volumes; finalizes any flow that has gone idle even if it never hit the packet-count threshold.
    for stale_flow in flow_table.expire_stale_flows():
        handle_finished_flow(stale_flow)

#  Main 
def main():
   
    log.info("REAF-5G Edge Node: Packet Capture")
    log.info(f"Method    : tcpdump pipe: Scapy")
    log.info(f"UE filter : {UE_SUBNET}  (prefix: {UE_PREFIX}.*)")
    log.info(f"Evidence  : {EVIDENCE_DIR}")


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
        f"src net {UE_PREFIX}.0/24 or dst net {UE_PREFIX}.0/24" # Explicit directionality
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
    )

    log.info("Waiting for traffic...")

    try:
        # sniff reads raw pcap bytes from the pipe Scapy parses each packet and calls on_packet() This runs forever until tcpdump exits or is killed
        sniff(
            offline=tcpdump.stdout,
            prn=on_packet,
            store=False
        )
    except KeyboardInterrupt:
        log.info("Stopping capture...")
    finally:
        tcpdump.terminate()


if __name__ == "__main__":
    main()