import os
import time
import json
import random
import logging
import socket
import struct
import fcntl
from pathlib import Path
from scapy.layers.inet import fragment

from scapy.all import RawPcapReader, Ether, IP, IPv6, L3RawSocket, TCP, UDP

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


def iter_packets(pcap_path):
    reader = RawPcapReader(str(pcap_path))
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
        yield pkt, ts


def rewrite_packet(pkt, ue_ip, target_ip=TARGET_IP):
    if IP not in pkt:
        return None    
    ip_pkt = pkt[IP].copy()

    ip_pkt.src = ue_ip
    ip_pkt.dst = target_ip

    if hasattr(ip_pkt, "len"):
        del ip_pkt.len

    if hasattr(ip_pkt, "chksum"):
        del ip_pkt.chksum
    if hasattr(ip_pkt.payload, "chksum"):
        del ip_pkt.payload.chksum

    if TCP in ip_pkt:
        del ip_pkt[TCP].chksum

    elif UDP in ip_pkt:
        del ip_pkt[UDP].chksum
    return ip_pkt


def make_socket(iface=IFACE):
   
    return L3RawSocket(iface=iface)


def replay_single_packet(pcap_path, iface=IFACE, target_ip=TARGET_IP, ue_ip=None):
   
    ue_ip = ue_ip or get_ue_ip(iface)
    if ue_ip is None:
        print(f"FAIL: could not resolve IP on {iface}")
        return False

    print(f"Opening {pcap_path} with RawPcapReader...")
    for pkt, ts in iter_packets(pcap_path):
        print(f"Read one packet, ts={ts}")
        if IP not in pkt:
            print("  not an IP packet, skipping")
            continue

        print(f"  original: {pkt.summary()}")
        out_pkt = rewrite_packet(pkt, ue_ip, target_ip)
        print(f"  rewritten: {out_pkt.summary()}")

        sock = make_socket(iface)
        print("Sending...")
        try:
            
            if len(bytes(out_pkt)) > 1300:
                fragments = fragment(out_pkt, fragsize=1300)

                for frag in fragments:
                    sock.send(frag)
            else:
                sock.send(out_pkt)
            print("Sent")
            return True
        except OSError as e:
            print(f"FAIL: sock.send() raised {e}")
            return False
        finally:
            sock.close()

    print("FAIL: pcap contained no IP packets (or was empty)")
    return False


def replay_pcap(pcap_path, replay_speed=1.0, target_ip=TARGET_IP, iface=IFACE, ue_ip=None, sock=None):
    ue_ip = ue_ip or get_ue_ip(iface)
    if ue_ip is None:
        log.error(f"Could not resolve UE IP on {iface}; aborting replay")
        return 0

    metadata = load_metadata(pcap_path)
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
        for pkt, ts in iter_packets(pcap_path):
            total += 1
            if prev_ts is not None:
                delay = (ts - prev_ts) / replay_speed
                if delay > 0:
                    time.sleep(delay)
            prev_ts = ts

            out_pkt = rewrite_packet(pkt, ue_ip, target_ip)
            if out_pkt is None:
                skipped += 1
                continue

            try:
                
                if len(bytes(out_pkt)) > 1300:
                    fragments = fragment(out_pkt, fragsize=1300)

                    for frag in fragments:
                        sock.send(frag)
                else:
                    sock.send(out_pkt)
        
                sent += 1
                log.info(f"REPLAY | {ue_ip} - {target_ip} | {out_pkt.summary()}")
            except OSError as e:
                errors += 1
                log.error(f"send() failed on packet {total}: {e}")
    finally:
        if own_socket:
            sock.close()

    log.info(f"REPLAY | {Path(pcap_path).name} complete "
             f"(read={total}, sent={sent}, skipped_non_ip={skipped}, errors={errors}, "
             f"expected_attack={metadata.get('expected_attack', 'unknown')})")
    return sent


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