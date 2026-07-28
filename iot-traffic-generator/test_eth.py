# from scapy.all import *

# for i, (raw_bytes, _) in enumerate(RawPcapReader(str("pcaps/BenignTraffic.pcap"))):

#     pkt = Ether(raw_bytes)

#     if IP not in pkt:
#         continue

#     print(i, pkt.summary())

#     if i == 20:
#         break

from scapy.all import RawPcapReader, Ether

reader = RawPcapReader("./pcaps/BenignTraffic.pcap")

for i, (raw, meta) in enumerate(reader):
    pkt = Ether(raw)
    print(i, len(raw), pkt.summary())

    if i == 10:
        break