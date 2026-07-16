# Runs inside nr_ue network namespace so it shares uesimtun0 with the UE container.
# Packets sent through uesimtun0 are GTP-U encapsulated by the gNB and decapsulated by the UPF onto ogstun where the forensic agent captures them.


import os
import sys
import time
import logging
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# Suppress Scapy IPv6/IPv4 mismatch warning on uesimtun0
import logging as _log
_log.getLogger("scapy.runtime").setLevel(_log.ERROR)
_log.getLogger("scapy.interactive").setLevel(_log.ERROR)
_log.getLogger("scapy.loading").setLevel(_log.ERROR)

from scapy.all import IP, TCP, UDP, ICMP, send
from replay_engine import replay_pcap, replay_mixed, replay_random

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAFFIC-GEN] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
log = logging.getLogger(__name__)

#  Config 
IFACE        = os.getenv("UE_TUNNEL_IFACE", "uesimtun0")
TARGET_IP    = os.getenv("TARGET_IP", "192.168.100.1")
INTERVAL     = float(os.getenv("SEND_INTERVAL", "2"))
MODE         = os.getenv("MODE", "synthetic")
PCAP_PATH    = os.getenv("PCAP_PATH", "pcaps/recon/Recon-PortScan.pcap")
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "1.0"))

# Fixed sequence used by MODE=mixed
MIXED_SEQUENCE = [
    "pcaps/benign/benign.pcap",
    "pcaps/recon/Recon-PortScan.pcap",
    "pcaps/ddos/DDOS-TCP_Flood.pcap",
    "pcaps/benign/benign.pcap",
]


#  Get UE IP from tunnel interface 
def get_ue_ip():
    result = subprocess.run(
        ["ip", "addr", "show", IFACE],
        capture_output=True, text=True
    )
    for line in result.stdout.split("\n"):
        if "inet " in line and "inet6" not in line:
            return line.strip().split()[1].split("/")[0]
    return None


#  Send helpers 
def tx(packet, msg):
   # Send packet through tunnel and log it.
    send(packet, iface=IFACE, verbose=False)
    log.info(msg)


#  Synthetic traffic functions (Stage 1, unchanged) 
def normal_ping(src):
    tx(
        IP(src=src, dst="8.8.8.8") / ICMP(),
        f"NORMAL | ICMP ping  | {src} - 8.8.8.8"
    )

def normal_http(src):
    tx(
        IP(src=src, dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="S"),
        f"NORMAL | HTTP SYN   | {src}:12345 - 8.8.8.8:80"
    )

def normal_dns(src):
    tx(
        IP(src=src, dst="8.8.8.8") / UDP(sport=54321, dport=53),
        f"NORMAL | DNS query  | {src}:54321 - 8.8.8.8:53"
    )

def attack_portscan(src):
    log.info(f"ATTACK | Port scan  | {src} - {TARGET_IP}")
    for port in [21, 22, 23, 25, 80, 443, 3306, 8080, 8443, 9999]:
        tx(
            IP(src=src, dst=TARGET_IP) / TCP(sport=11111, dport=port, flags="S"),
            f"ATTACK | SYN scan   | {src} - {TARGET_IP}:{port}"
        )
        time.sleep(0.1)

def attack_ddos(src):
    log.info(f"ATTACK | DDoS flood | {src} - {TARGET_IP} (20 pkts)")
    for i in range(20):
        tx(
            IP(src=src, dst=TARGET_IP) / UDP(sport=i + 1000, dport=80),
            f"ATTACK | UDP flood  | {src}:{i+1000} - {TARGET_IP}:80"
        )
        time.sleep(0.02)

def run_synthetic(src):
    cycle = 0
    while True:
        cycle += 1
        log.info(f"--- Cycle {cycle} ---")

        normal_ping(src)
        time.sleep(INTERVAL)

        normal_http(src)
        time.sleep(INTERVAL)

        normal_dns(src)
        time.sleep(INTERVAL)

        if cycle % 5 == 0:
            attack_portscan(src)
            time.sleep(INTERVAL)
            attack_ddos(src)

        time.sleep(INTERVAL)


#  Main 
def main():
    log.info("=" * 60)
    log.info("REAF-5G IoT Traffic Generator")
    log.info(f"Interface : {IFACE}")
    log.info(f"Target IP : {TARGET_IP}")
    log.info(f"Mode      : {MODE}")
    log.info("=" * 60)

    # Wait for uesimtun0 to be ready — needed for every mode
    log.info(f"Waiting for {IFACE} to be ready...")
    src = None
    while src is None:
        src = get_ue_ip()
        if src is None:
            time.sleep(5)
    log.info(f"Tunnel ready. UE IP: {src}")
    log.info("")

    if MODE == "synthetic":
        log.info("Starting synthetic traffic generation...")
        run_synthetic(src)

    elif MODE == "replay":
        log.info(f"Replaying single PCAP: {PCAP_PATH}")
        replay_pcap(Path(PCAP_PATH), replay_speed=REPLAY_SPEED,
                    target_ip=TARGET_IP, iface=IFACE, ue_ip=src)

    elif MODE == "mixed":
        log.info(f"Replaying mixed sequence: {MIXED_SEQUENCE}")
        replay_mixed(MIXED_SEQUENCE, replay_speed=REPLAY_SPEED,
                     target_ip=TARGET_IP, iface=IFACE)

    elif MODE == "random":
        log.info("Replaying a random PCAP...")
        replay_random(replay_speed=REPLAY_SPEED, target_ip=TARGET_IP, iface=IFACE)

    else:
        log.error(f"Unknown MODE: {MODE} (expected synthetic, replay, mixed, or random)")


if __name__ == "__main__":
    main()
