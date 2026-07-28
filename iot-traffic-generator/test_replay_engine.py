
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