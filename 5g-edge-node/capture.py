
#capture.py — Minimal Packet Capture



import os
import sys
import logging
from datetime import datetime
from scapy.all import sniff, IP, TCP, UDP, ICMP

#  Logging setup 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE-NODE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

#  Configuration 
# The network interface inside the Docker container. 'eth0' is the default, inside the container to find the correct name.
INTERFACE = os.getenv("CAPTURE_INTERFACE", "eth0")

# Evidence folder mounted 
EVIDENCE_DIR = "/evidence"

#  Packet callback 
def on_packet(packet):
    #Called by Scapy for every packet that arrives on the network interface.
    
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Only process IP packets for now
        if IP not in packet:
            return

        src_ip   = packet[IP].src
        dst_ip   = packet[IP].dst
        protocol = packet[IP].proto

        # Identify transport layer
        if TCP in packet:
            transport = f"TCP  src_port={packet[TCP].sport} dst_port={packet[TCP].dport}"
        elif UDP in packet:
            transport = f"UDP  src_port={packet[UDP].sport} dst_port={packet[UDP].dport}"
        elif ICMP in packet:
            transport = f"ICMP type={packet[ICMP].type}"
        else:
            transport = f"PROTO={protocol}"

        log.info(f"PKT | {timestamp} | {src_ip} → {dst_ip} | {transport} | len={len(packet)}")

    except Exception as e:
        log.error(f"Error processing packet: {e}")


#  Main ─
def main():
    log.info("=" * 60)
    log.info("REAF-5G Edge Node — Stage 1: Packet Capture")
    log.info(f"Interface  : {INTERFACE}")
    log.info(f"Evidence   : {EVIDENCE_DIR}")
    log.info("Status     : Waiting for traffic...")
    log.info("=" * 60)

    # Confirm evidence directory exists
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    try:
        # Start capturing this runs forever until stopped
        # filter="ip" means only capture IP packets
        # store=False means do not store packets in RAM 
        # prn=on_packet means call on_packet() for every captured packet
        sniff(
            iface=INTERFACE,
            filter="ip",
            prn=on_packet,
            store=False
        )
    except PermissionError:
        log.error("Permission denied — container needs NET_ADMIN and privileged=true")
        log.error("Check your docker-compose.yml cap_add and privileged settings")
        sys.exit(1)
    except OSError as e:
        log.error(f"Interface '{INTERFACE}' not found: {e}")
        log.error("Run 'ip link show' inside the container to list interfaces")
        sys.exit(1)


if __name__ == "__main__":
    main()