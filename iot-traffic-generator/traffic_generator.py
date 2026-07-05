# Send basic test packets to confirm the edge node is capturing traffic on the Docker network.

import time
import logging
from datetime import datetime
from scapy.all import IP, TCP, UDP, ICMP, send

#  Logging setup 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAFFIC-GEN] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# This is the Open5GS UPF container IP on the Docker network.
# When Open5GS is running, the UPF sits at this address.
# Your edge node will intercept this traffic as it flows past.
TARGET_IP   = "192.168.100.1"     # UPF internal IP — adjust if different
SOURCE_IP   = "192.168.100.2"     # Simulated UE IP assigned by Open5GS

# How long to wait between packets (seconds)
INTERVAL    = 2

#  Traffic functions 
def send_normal_http():
    # Simulate normal HTTP web browsing traffic.
    pkt = IP(src=SOURCE_IP, dst=TARGET_IP) / TCP(
        sport=12345,
        dport=80,
        flags="S"   # SYN — start of TCP handshake
    )
    send(pkt, verbose=False)
    log.info(f"NORMAL | HTTP SYN | {SOURCE_IP}:{12345} - {TARGET_IP}:80")


def send_normal_dns():
    # Simulate normal DNS lookup traffic.
    pkt = IP(src=SOURCE_IP, dst="8.8.8.8") / UDP(
        sport=54321,
        dport=53
    )
    send(pkt, verbose=False)
    log.info(f"NORMAL | DNS query | {SOURCE_IP} - 8.8.8.8:53")


def send_normal_ping():
    # Simulate normal ICMP ping.
    pkt = IP(src=SOURCE_IP, dst=TARGET_IP) / ICMP()
    send(pkt, verbose=False)
    log.info(f"NORMAL | ICMP ping | {SOURCE_IP} - {TARGET_IP}")


def send_simulated_portscan():
    
    # Simulate a basic port scan. Sends SYN packets to multiple ports rapidly. This should trigger your detection engine .   
    log.info("ATTACK | Port scan starting...")
    for port in [21, 22, 23, 25, 80, 443, 3306, 8080, 8443, 9999]:
        pkt = IP(src=SOURCE_IP, dst=TARGET_IP) / TCP(
            sport=11111,
            dport=port,
            flags="S"
        )
        send(pkt, verbose=False)
        log.info(f"ATTACK | SYN scan | {SOURCE_IP} - {TARGET_IP}:{port}")
        time.sleep(0.1)


def send_simulated_ddos():
    
    # Simulate a basic DDoS flood.
    # Sends rapid UDP packets — matches DDoS pattern in CICIoT2023.

    log.info("ATTACK | DDoS flood starting (10 packets)...")
    for i in range(10):
        pkt = IP(src=SOURCE_IP, dst=TARGET_IP) / UDP(
            sport=i + 1000,
            dport=80
        )
        send(pkt, verbose=False)
        time.sleep(0.05)
    log.info("ATTACK | DDoS flood complete")


#  Main loop 
def main():
    log.info("=" * 60)
    log.info("REAF-5G IoT Traffic Generator — Stage 1")
    log.info(f"Target     : {TARGET_IP}")
    log.info(f"Source     : {SOURCE_IP}")
    log.info(f"Mode       : Normal traffic + periodic attack simulation")
    log.info("=" * 60)

    # Wait for the network to settle before sending anything
    log.info("Waiting 15 seconds for network to be ready...")
    time.sleep(15)

    cycle = 0

    while True:
        cycle += 1
        log.info(f" Cycle {cycle} ")

        # Send normal traffic every cycle
        send_normal_ping()
        time.sleep(INTERVAL)

        send_normal_http()
        time.sleep(INTERVAL)

        send_normal_dns()
        time.sleep(INTERVAL)

        # Every 5th cycle simulate an attack
        # so your edge node has something to detect
        if cycle % 5 == 0:
            log.info("Injecting simulated attack traffic...")
            send_simulated_portscan()
            time.sleep(INTERVAL)
            send_simulated_ddos()

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()