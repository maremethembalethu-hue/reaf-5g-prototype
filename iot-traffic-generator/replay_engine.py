import os
import time
import json
import random
import logging
import socket
import struct
import fcntl
import hashlib
from pathlib import Path
from scapy.layers.inet import fragment

from scapy.all import RawPcapReader, Ether, IP, IPv6, L3RawSocket, TCP, UDP

from pkt_debug import inspect_packet

from build_envelope import build_envelope_chunks
# pcap link-layer type codes we know how to decode.
_LINKTYPE_DECODERS = {
    1:   Ether,   # DLT_EN10MB  standard Ethernet capture (most tcpdump/CICIoT2023 pcaps)
    101: IP,      # DLT_RAW     raw IP, no link-layer header
    228: IP,      # DLT_IPV4    raw IPv4, no link-layer header
    229: IPv6,    # DLT_IPV6    raw IPv6, no link-layer header
}

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
logging.getLogger("scapy.interactive").setLevel(logging.ERROR)
logging.getLogger("scapy.loading").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("TRAFFIC-GEN")
BASE_DIR = Path(__file__).resolve().parent

IFACE     = os.getenv("UE_TUNNEL_IFACE", "uesimtun0")
TARGET_IP = os.getenv("TARGET_IP", "192.168.100.1")
PCAP_DIR  = BASE_DIR / "pcaps"/ Path(os.getenv("PCAP_DIR", "pcaps"))
ORIGINAL_UE = "192.168.137.175"
MAX_CHUNK_PAYLOAD = int(os.getenv("REAF_MAX_CHUNK_PAYLOAD", "1300"))
REAF_PORT = int(os.getenv("REAF_PORT", "9999"))

    # BASE_DIR / "pcaps" / "BenignTraffic.pcap",
    # BASE_DIR / "pcaps" / "DDoS-UDP_Flood.pcap",
def get_ue_ip(iface=IFACE):
    # ioctl(SIOCGIFADDR) lookup instead of shelling out to `ip` and parsing
    # text  no dependency on the `ip` binary, no locale/format parsing risk.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed_iface = struct.pack("256s", iface[:15].encode("utf-8"))
            addr = fcntl.ioctl(s.fileno(), 0x8915, packed_iface)  # SIOCGIFADDR
            return socket.inet_ntoa(addr[20:24])
        finally:
            s.close()
    except OSError as e:
        log.error(f"Could not read address for {iface}: {e}")
        return None


def load_metadata(pcap_path):
    # rdpcap is gone, but the .json sidecar (expected_attack label etc) is
    # still worth reading  it's tiny, unrelated to the packet-count problem.
    meta_path = Path(str(pcap_path).rsplit(".", 1)[0] + ".json")
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}

def id_for(pcap_path):
  
    digest = hashlib.sha1(str(pcap_path).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")

def iter_packets(pcap_path):
    reader = RawPcapReader(str(pcap_path))
    linktype = reader.linktype
    decoder = _LINKTYPE_DECODERS.get(reader.linktype)
    if decoder is None:
        log.warning(f"Unrecognized linktype {reader.linktype} for {pcap_path}, "
                     f"defaulting to Ethernet decode")
        decoder = Ether

    for raw_bytes, pkt_meta in reader:
        pkt = decoder(raw_bytes)
        if IP not in pkt:
            continue
        ts = pkt_meta.sec + pkt_meta.usec / 1e6
        ip_bytes = bytes(pkt[IP])
        yield ip_bytes, ts, linktype



def rewrite_packet(pkt, ue_ip, target_ip):

    if IP not in pkt:
        return None

    ip_pkt = pkt[IP].copy()

    # Outbound
    if ip_pkt.src == ORIGINAL_UE:
        ip_pkt.src = ue_ip
        ip_pkt.dst = target_ip

    # Inbound
    elif ip_pkt.dst == ORIGINAL_UE:
        ip_pkt.src = target_ip
        ip_pkt.dst = ue_ip

    # Any other packet
    else:
        # Still replay it
        ip_pkt.src = ue_ip
        ip_pkt.dst = target_ip

    del ip_pkt.len
    del ip_pkt.chksum

    if TCP in ip_pkt:
        del ip_pkt[TCP].chksum

    if UDP in ip_pkt:
        del ip_pkt[UDP].chksum

    return ip_pkt


def make_socket(iface=IFACE):
   
    return L3RawSocket(iface=iface)

def send_envelope_chunk(sock, ue_ip, target_ip, chunk_bytes):
    # The OUTER packet is the only thing that gets new addressing
    outer = IP(src=ue_ip, dst=target_ip) / UDP(sport=REAF_PORT, dport=REAF_PORT) / chunk_bytes
    del outer.len
    del outer.chksum
    del outer[UDP].chksum
    sock.outs.sendto(bytes(outer), (target_ip, 0))

def replay_single_packet(pcap_path, iface=IFACE, target_ip=TARGET_IP, ue_ip=None):
   
    ue_ip = ue_ip or get_ue_ip(iface)
    if ue_ip is None:
        print(f"FAIL: could not resolve IP on {iface}")
        return False
    flow_id = id_for(pcap_path)
    print(f"Opening {pcap_path} with RawPcapReader...")
    sock = make_socket(iface)
    try:
        for packet_id, (original_bytes, ts, linktype) in enumerate(iter_packets(pcap_path)):
            print(f"Read one packet, ts={ts}, {len(original_bytes)} original bytes")
            chunks = build_envelope_chunks(
                flow_id, packet_id, ts, linktype, original_bytes,
                max_chunk_payload=MAX_CHUNK_PAYLOAD,
            )
            print(f"  encapsulated into {len(chunks)} envelope chunk(s), original untouched")
            print("Sending...")
            try:
                for chunk in chunks:
                    send_envelope_chunk(sock, ue_ip, target_ip, chunk)
                print("Sent")
                return True
            except OSError as e:
                print(f"FAIL: sock.send() raised {e}")
                return False
    finally:
        sock.close()

    print("FAIL: pcap contained no IP packets (or was empty)")
    return False

def replay_pcap(pcap_path, replay_speed=1.0, target_ip=TARGET_IP, iface=IFACE, ue_ip=None, sock=None,
                 deadline=None, MAX_DELAY =None):
    ue_ip = ue_ip or get_ue_ip(iface)
    if ue_ip is None:
        log.error(f"Could not resolve UE IP on {iface}; aborting replay")
        return 0
    log.info(f"REPLAY | effective replay_speed={replay_speed}, MAX_DELAY={MAX_DELAY}")
    metadata = load_metadata(pcap_path)
    flow_id = id_for(pcap_path)
    own_socket = sock is None
    if own_socket:
        sock = make_socket(iface)

    log.info(f"REPLAY | {Path(pcap_path).name} | {ue_ip} -> {target_ip} | speed={replay_speed}x")
    prev_ts = None
    sent = 0
    skipped = 0
    errors = 0
    total = 0

    try:
        for packet_id, (original_bytes, ts, linktype) in enumerate(iter_packets(pcap_path)):
            if deadline is not None and time.time() >= deadline:
                log.warning(f"REPLAY | {Path(pcap_path).name} | deadline reached — "
                            f"stopping mid-file at packet {total} (sent={sent})")
                break
            total += 1
            # if prev_ts is not None:
            #     delay = (ts - prev_ts) / replay_speed
                
    
            #     # if MAX_DELAY is not None and delay > MAX_DELAY:
            #     #     delay = MAX_DELAY
                
            #     # #MAX_DELAY = 0.05      # 50 ms

            #     # if delay > 0:

            #     time.sleep(delay)
            #     log.info(f"Delay = {delay:.3f}s")
               
            prev_ts = ts

            chunks = build_envelope_chunks(
                flow_id, packet_id, ts, linktype, original_bytes,
                max_chunk_payload=MAX_CHUNK_PAYLOAD,
            )

            try:
                for chunk in chunks:
                    send_envelope_chunk(sock, ue_ip, target_ip, chunk)
                sent += 1
                log.info(f"REPLAY | {ue_ip} to {target_ip} | packet_id={packet_id} "
                         f"({len(original_bytes)}B original, {len(chunks)} chunk(s))")
            except OSError as e:
                errors += 1
                if errors <= 3 or errors % 50 == 0:
                    log.error(f"send() failed on packet {total}: {e}")
    finally:
        if own_socket:
            sock.close()

    log.info(f"REPLAY | {Path(pcap_path).name} complete "
             f"(read={total}, sent={sent}, errors={errors}, "
             f"expected_attack={metadata.get('expected_attack', 'unknown')})")
    return {"packets_read": total, "packets_sent": sent}


def replay_mixed(pcap_paths, replay_speed=1.0, gap=3.0, iface=IFACE, target_ip=TARGET_IP):
    ue_ip = get_ue_ip(iface)
    sock = make_socket(iface)
    try:
        for p in pcap_paths:
            replay_pcap(p, replay_speed=replay_speed, target_ip=target_ip, iface=iface, ue_ip=ue_ip, sock=sock)
            time.sleep(gap)
    finally:
        sock.close()


def replay_random(pcap_dir=PCAP_DIR, replay_speed=1.0, iface=IFACE, target_ip=TARGET_IP):
    candidates = list(Path(pcap_dir).rglob("*.pcap"))
    if not candidates:
        log.error(f"No .pcap files found under {pcap_dir}")
        return
    choice = random.choice(candidates)
    replay_pcap(choice, replay_speed=replay_speed, target_ip=target_ip, iface=iface)