import os
import time
import json
import random
import logging
import socket
import struct
import fcntl
from pathlib import Path

from scapy.all import RawPcapReader, Ether, IP, IPv6, L3RawSocket

# pcap link-layer type codes we know how to decode.
# https://www.tcpdump.org/linktypes.html
_LINKTYPE_DECODERS = {
    1:   Ether,   # DLT_EN10MB -- standard Ethernet capture (most tcpdump/CICIoT2023 pcaps)
    101: IP,      # DLT_RAW    -- raw IP, no link-layer header
    228: IP,      # DLT_IPV4   -- raw IPv4, no link-layer header
    229: IPv6,    # DLT_IPV6   -- raw IPv6, no link-layer header
}

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
logging.getLogger("scapy.interactive").setLevel(logging.ERROR)
logging.getLogger("scapy.loading").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("TRAFFIC-GEN")

IFACE     = os.getenv("UE_TUNNEL_IFACE", "uesimtun0")
TARGET_IP = os.getenv("TARGET_IP", "192.168.100.1")
PCAP_DIR  = Path(os.getenv("PCAP_DIR", "pcaps"))


def get_ue_ip(iface=IFACE):
    """
    ioctl(SIOCGIFADDR) lookup instead of shelling out to `ip` and parsing
    text -- no dependency on the `ip` binary, no locale/format parsing risk.
    """
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
    # still worth reading -- it's tiny, unrelated to the packet-count problem.
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
        ts = pkt_meta.sec + pkt_meta.usec / 1e6
        yield pkt, ts


def rewrite_packet(pkt, ue_ip, target_ip=TARGET_IP):
    if IP not in pkt:
        return None
    ip_pkt = pkt[IP].copy()
    ip_pkt.src = ue_ip
    ip_pkt.dst = target_ip
    del ip_pkt.chksum
    if hasattr(ip_pkt.payload, "chksum"):
        del ip_pkt.payload.chksum
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
                sock.send(out_pkt)
                sent += 1
                log.info(f"REPLAY | {ue_ip} -> {target_ip} | {out_pkt.summary()}")
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














"""
Test harness for the streaming replay_engine.py.

WHY THIS VERSION USES A REAL INTERFACE, NOT `lo`:
`lo` is link-type EN10MB internally but behaves specially in the kernel
(loopback fast-path, no real ARP/routing). Testing purely on `lo` proves
L3RawSocket doesn't raise, but doesn't prove packets survive a real NIC's
send path the way they'll need to over `uesimtun0`. This script auto-picks
a real interface unless you override it.

USAGE
-----
Auto-detect a real interface + its IP:

    sudo .venv/bin/python test_replay_engine.py

Force a specific interface/target (e.g. once you're back in Docker):

    export UE_TUNNEL_IFACE=uesimtun0
    export TARGET_IP=192.168.100.1
    sudo .venv/bin/python test_replay_engine.py

Raw sockets need root -- run with sudo or you'll get PermissionError.

WHAT EACH STAGE ISOLATES
-------------------------
1. Interface check       -- can we even resolve an IP on this iface?
2. PCAP directory check  -- are there .pcap files to replay?
3. Single-packet debug   -- read ONE packet, rewrite it, send it, with
                             explicit prints at every step (this is the
                             "which of these 4 things is failing" test).
4. Full streaming replay -- run replay_pcap() (RawPcapReader-based) across
                             the whole file and report read/sent/skipped/
                             error counts.

Cross-check step 4's counts against a concurrent tcpdump on the interface
you're actually sending out of -- that's the real ground truth, not the
sent-count alone.
"""

import os
import sys
import logging
from pathlib import Path


def detect_real_iface():
   # Pick the first non-loopback interface that has an IPv4 address."""
    import socket, fcntl, struct
    candidates = [i for i in os.listdir("/sys/class/net") if i != "lo"]
    for iface in candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            packed = struct.pack("256s", iface[:15].encode("utf-8"))
            addr = fcntl.ioctl(s.fileno(), 0x8915, packed)
            ip = socket.inet_ntoa(addr[20:24])
            s.close()
            return iface, ip
        except OSError:
            continue
    return None, None


# IMPORTANT: env vars must be set BEFORE `import replay_engine`
if "UE_TUNNEL_IFACE" not in os.environ:
    real_iface, real_ip = detect_real_iface()
    if real_iface is None:
        print("Could not auto-detect a real interface; falling back to lo.")
        os.environ["UE_TUNNEL_IFACE"] = "lo"
        os.environ.setdefault("TARGET_IP", "127.0.0.1")
    else:
        print(f"Auto-detected real interface: {real_iface} ({real_ip})")
        os.environ["UE_TUNNEL_IFACE"] = real_iface
        # Send to this host's own IP on the real NIC -- proves L3RawSocket
        os.environ.setdefault("TARGET_IP", real_ip)
else:
    os.environ.setdefault("TARGET_IP", "127.0.0.1")

import replay_engine as re

log = logging.getLogger("TEST")


def step(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")


def test_interface():
    step("1. Interface check")
    print(f"Using UE_TUNNEL_IFACE={re.IFACE}  TARGET_IP={re.TARGET_IP}")
    ip = re.get_ue_ip()
    if ip is None:
        print(f"FAIL: could not resolve IP on {re.IFACE}.")
        return None
    print(f"OK: {re.IFACE} -> {ip}")
    return ip


def test_pcap_dir():
    step("2. PCAP directory check")
    if not re.PCAP_DIR.exists():
        print(f"FAIL: {re.PCAP_DIR} does not exist")
        return []
    pcaps = sorted(re.PCAP_DIR.rglob("*.pcap"))
    if not pcaps:
        print(f"FAIL: no .pcap files under {re.PCAP_DIR}")
        return []
    print(f"OK: found {len(pcaps)} pcap file(s)")
    for p in pcaps[:10]:
        print(f"  - {p}")
    return pcaps


def test_single_packet(pcap_path, ue_ip):
    step(f"3. Single-packet debug on {pcap_path.name}")
    ok = re.replay_single_packet(pcap_path, ue_ip=ue_ip)
    if not ok:
        print("FAIL: single-packet test did not complete -- see output above "
              "for exactly which stage failed (read / rewrite / send).")
    return ok


def test_full_replay(pcap_path, ue_ip):
    step(f"4. Full streaming replay of {pcap_path.name} (sped up 20x)")
    sent = re.replay_pcap(pcap_path, replay_speed=20.0, ue_ip=ue_ip)
    if sent == 0:
        print("FAIL: 0 packets sent. Check the errors/skipped counts logged above.")
    else:
        print(f"OK: replay_pcap() reports {sent} packets sent.")
        print("  Cross-check this number against a concurrent tcpdump.")
    return sent


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ue_ip = test_interface()
    pcaps = test_pcap_dir()

    if not pcaps:
        print("\nCannot continue without at least one pcap file.")
        sys.exit(1)

    if ue_ip is None:
        print("\nSkipping send tests: no UE IP resolved.")
        sys.exit(1)

    if not test_single_packet(pcaps[0], ue_ip):
        print("\nStopping here -- fix the single-packet failure before "
              "attempting a full replay.")
        sys.exit(1)

    test_full_replay(pcaps[0], ue_ip)

    step("Done")
    print("If everything above is OK but a concurrent tcpdump on the target "
          "interface sees nothing, the issue has moved out of Scapy and into "
          "routing/firewall rules on that interface.")


if __name__ == "__main__":
    main()