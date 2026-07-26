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