import sys
import logging
from pathlib import Path
 
import replay_engine as re
 
log = logging.getLogger("TEST")
 
 
def step(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    
def test_interface():
    step("1. Interface check")
    ip = re.get_ue_ip()
    if ip is None:
        print(f"FAIL: could not resolve IP on {re.IFACE}. "
              f"Is uesimtun0 up? Run 'ip addr show {re.IFACE}' manually.")
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

def test_load_pcap(pcap_path):
    step(f"3. Loading {pcap_path.name}")
    packets, metadata = re.load_pcap(pcap_path)
    if not packets:
        print("FAIL: 0 packets loaded")
        return None
    print(f"OK: {len(packets)} packets loaded, metadata={metadata}")
    ip_count = sum(1 for p in packets if re.IP in p)
    print(f"  {ip_count}/{len(packets)} packets have an IP layer (non-IP ones get skipped)")
    return packets


 
def test_raw_socket_send(ue_ip):
    step("4. Raw socket send test (bypasses replay logic entirely)")
    from scapy.all import ICMP
    try:
        sock = re.make_socket(re.IFACE)
        pkt = re.IP(src=ue_ip, dst=re.TARGET_IP) / ICMP()
        sock.send(pkt)
        sock.close()
        print(f"OK: sent one raw ICMP packet {ue_ip} -> {re.TARGET_IP} with no exception")
        print("  Confirm it actually arrived via tcpdump on the UPF side.")
    except OSError as e:
        print(f"FAIL: socket.send() raised {e}")
        print("  This means the fix isn't enough on your kernel/container -- ")
        print("  check that the container has NET_RAW/NET_ADMIN capability,")
        print("  and that TARGET_IP is actually routable via uesimtun0.")
 
 
def test_replay(pcap_path, ue_ip):
    step(f"5. Full replay_pcap() test on {pcap_path.name} (sped up 20x)")
    sent = re.replay_pcap(pcap_path, replay_speed=20.0, ue_ip=ue_ip)
    if sent == 0:
        print("FAIL: 0 packets sent. Check the errors/skipped counts logged above.")
    else:
        print(f"OK: replay_pcap() reports {sent} packets sent.")
        print("  Cross-check this number against tcpdump on the UPF interface.")
 
 
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
 
    ue_ip = test_interface()
    pcaps = test_pcap_dir()
 
    if not pcaps:
        print("\nCannot continue without at least one pcap file. Set PCAP_DIR "
              "or check pcaps/ contains .pcap files.")
        sys.exit(1)
 
    packets = test_load_pcap(pcaps[0])
    if not packets:
        sys.exit(1)
 
    if ue_ip is None:
        print("\nSkipping send tests: no UE IP resolved.")
        sys.exit(1)
 
    test_raw_socket_send(ue_ip)
    test_replay(pcaps[0], ue_ip)
 
    step("Done")
    print("If step 4/5 show OK but tcpdump on the UPF sees nothing, the issue")
    print("has moved out of Scapy and into the UPF/N3 GTP-U tunnel or your")
    print("iptables/routing rules on the ogstun side.")
 
 
if __name__ == "__main__":
    main()