from scapy.all import *

for i, (raw_bytes, _) in enumerate(RawPcapReader(str("pcaps/BenignTraffic.pcap"))):

    pkt = Ether(raw_bytes)

    if IP not in pkt:
        continue

    print(i, pkt.summary())

    if i == 20:
        break