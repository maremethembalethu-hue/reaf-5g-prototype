# traffic_generator.py — REAF-5G IoT Traffic Generator: Runs inside nr_ue network namespace so it shares uesimtun0 with the UE container
# Packets sent through uesimtun0 are GTP-U encapsulated by the gNB and decapsulated by the UPF onto ogstun where the forensic agent captures them.

import os
import sys
import time
import logging
import subprocess

sys.stdout.reconfigure(line_buffering=True)

# Suppress Scapy IPv6/IPv4 mismatch warning on uesimtun0
import logging as _log
_log.getLogger("scapy.runtime").setLevel(_log.ERROR)
_log.getLogger("scapy.interactive").setLevel(_log.ERROR)
_log.getLogger("scapy.loading").setLevel(_log.ERROR)

from scapy.all import IP, TCP, UDP, ICMP, send

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAFFIC-GEN] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
log = logging.getLogger(__name__)

#  Config 
IFACE     = os.getenv("UE_TUNNEL_IFACE", "uesimtun0")
TARGET_IP = os.getenv("TARGET_IP", "192.168.100.1")
INTERVAL  = float(os.getenv("SEND_INTERVAL", "2"))


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
    """Send packet through tunnel and log it."""
    send(packet, iface=IFACE, verbose=False)
    log.info(msg)


#  Traffic functions 
def normal_ping(src):
    tx(
        IP(src=src, dst="8.8.8.8") / ICMP(),
        f"NORMAL | ICMP ping  | {src} → 8.8.8.8"
    )

def normal_http(src):
    tx(
        IP(src=src, dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="S"),
        f"NORMAL | HTTP SYN   | {src}:12345 → 8.8.8.8:80"
    )

def normal_dns(src):
    tx(
        IP(src=src, dst="8.8.8.8") / UDP(sport=54321, dport=53),
        f"NORMAL | DNS query  | {src}:54321 → 8.8.8.8:53"
    )

def attack_portscan(src):
    log.info(f"ATTACK | Port scan  | {src} → {TARGET_IP}")
    for port in [21, 22, 23, 25, 80, 443, 3306, 8080, 8443, 9999]:
        tx(
            IP(src=src, dst=TARGET_IP) / TCP(sport=11111, dport=port, flags="S"),
            f"ATTACK | SYN scan   | {src} → {TARGET_IP}:{port}"
        )
        time.sleep(0.1)

def attack_ddos(src):
    log.info(f"ATTACK | DDoS flood | {src} → {TARGET_IP} (20 pkts)")
    for i in range(20):
        tx(
            IP(src=src, dst=TARGET_IP) / UDP(sport=i + 1000, dport=80),
            f"ATTACK | UDP flood  | {src}:{i+1000} → {TARGET_IP}:80"
        )
        time.sleep(0.02)


#  Main 
def main():
    log.info("=" * 60)
    log.info("REAF-5G IoT Traffic Generator — Stage 1")
    log.info(f"Interface : {IFACE}")
    log.info(f"Target IP : {TARGET_IP}")
    log.info("=" * 60)

    # Wait for uesimtun0 to be ready
    log.info(f"Waiting for {IFACE} to be ready...")
    src = None
    while src is None:
        src = get_ue_ip()
        if src is None:
            time.sleep(5)

    log.info(f"Tunnel ready. UE IP: {src}")
    log.info("Starting traffic generation...")
    log.info("")

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


if __name__ == "__main__":
    main()