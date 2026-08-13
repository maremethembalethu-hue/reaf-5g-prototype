# from pathlib import Path
# from scapy.all import IP, PcapReader

# path = Path("evidence/incident_20260812_103531_834491/network_capture.pcap")

# # Convert the PosixPath object to a string via str(path)
# with PcapReader(str(path)) as reader:
#     print(f"linktype: {reader.linktype}")
#     for i, pkt in enumerate(reader):
#         print(f"[{i}] top layer: {pkt.__class__.__name__}, len(pkt)={len(pkt)}, "
#               f"len(pkt[IP]) if IP in pkt else N/A: "
#               f"{len(pkt[IP]) if IP in pkt else 'no IP'}")
#         if i >= 9:
#             break

# from scapy.utils import PcapReader
# from scapy.all import IP, TCP

# def flag_summary(pcap_path, label, decoder=None):
#     counts = {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "n": 0}
#     with PcapReader(str(pcap_path)) as reader:
#         for pkt in reader:
#             if TCP not in pkt:
#                 continue
#             counts["n"] += 1
#             f = pkt[TCP].flags
#             if f & 0x02: counts["SYN"] += 1
#             if f & 0x10: counts["ACK"] += 1
#             if f & 0x01: counts["FIN"] += 1
#             if f & 0x04: counts["RST"] += 1
#             if f & 0x08: counts["PSH"] += 1
#     print(f"{label}: {counts}")

# flag_summary("iot-traffic-generator/pre-selected/replay_pcaps_contiguous/Benign_Final/BenignTraffic_contiguous_sample.pcap", "ORIGINAL")
# flag_summary("evidence/incident_20260812_120900_624124/network_capture.pcap", "LIVE-CAPTURED")

from collections import Counter
from scapy.utils import PcapReader
from scapy.all import IP, TCP

ports = Counter()
with PcapReader("iot-traffic-generator/pre-selected/replay_pcaps_contiguous/Benign_Final/BenignTraffic_contiguous_sample.pcap") as reader:
    for pkt in reader:
        if TCP in pkt:
            ports[pkt[TCP].dport] += 1
print(ports.most_common(10))