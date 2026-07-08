
# capture.py — REAF-5G Edge Node Stage 1 Real-time packet capture inside the UPF network namespace.

import os
import sys
import logging
import subprocess
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

import logging as _log
_log.getLogger("scapy.runtime").setLevel(_log.ERROR)
_log.getLogger("scapy.interactive").setLevel(_log.ERROR)
_log.getLogger("scapy.loading").setLevel(_log.ERROR)

from scapy.all import sniff, IP, TCP, UDP, ICMP

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


#  Packet callback 
def on_packet(packet):
    try:
        if not packet.haslayer(IP):
            return

        ip_layer = packet[IP]
        src = ip_layer.src
        dst = ip_layer.dst

        if not (src.startswith(UE_PREFIX) or dst.startswith(UE_PREFIX)):
            return

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if TCP in packet:
            proto = (
                f"TCP "
                f"sport={packet[TCP].sport} "
                f"dport={packet[TCP].dport} "
                f"flags={packet[TCP].flags}"
            )
            label = "[!! RECON  ]" if int(packet[TCP].flags) == 0x02 else "[UE-TUNNEL ]"

        elif UDP in packet:
            proto = f"UDP sport={packet[UDP].sport} dport={packet[UDP].dport}"
            label = "[!! DDOS   ]" if packet[UDP].dport == 80 else "[UE-TUNNEL ]"

        elif ICMP in packet:
            proto = f"ICMP type={packet[ICMP].type}"
            label = "[UE-TUNNEL ]"

        else:
            proto = f"PROTO={packet[IP].proto}"
            label = "[UE-TUNNEL ]"

        log.info(f"{label} {ts} | {src} → {dst} | {proto} | len={len(packet)}")

    except Exception as e:
        log.error(f"Packet error: {e}")


#  Main 
def main():
    log.info("=" * 60)
    log.info("REAF-5G Edge Node — Stage 1: Packet Capture")
    log.info(f"Method    : tcpdump pipe → Scapy (TUN-compatible)")
    log.info(f"UE filter : {UE_SUBNET}  (prefix: {UE_PREFIX}.*)")
    log.info(f"Evidence  : {EVIDENCE_DIR}")
    log.info("=" * 60)

    for subdir in ["packets", "memory", "processes", "syslogs"]:
        os.makedirs(os.path.join(EVIDENCE_DIR, subdir), exist_ok=True)

    log.info("Starting tcpdump capture pipe...")

    # tcpdump -i any    — capture on ALL interfaces including ogstun (TUN)
    # -n                — do not resolve hostnames (faster)
    # -U                — packet-buffered output (flush each packet immediately)
    # -w -              — write raw pcap to stdout
    # host 192.168.100  — BPF filter: only UE subnet packets
    # 2>/dev/null       — suppress tcpdump startup messages
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