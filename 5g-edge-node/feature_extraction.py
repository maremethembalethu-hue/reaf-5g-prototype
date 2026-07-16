
# Extracts the 23feature subset (Heavy Model) and 10feature subset (Lite Model) from a live Scapy packet.

# Feature order must exactly match what was used during training


import numpy as np
from scapy.all import IP, TCP, UDP, ICMP


#  Heavy Model feature names 
HEAVY_FEATURES = [
    "Header_Length", "Protocol Type", "Rate", "fin_flag_number",
    "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "TCP", "UDP", "ICMP", "HTTP", "DNS", "Tot sum",
    "Min", "Max", "AVG"
]

#  Lite Model feature names  
LITE_FEATURES = [
    "Protocol Type", "syn_flag_number", "ack_flag_number",
    "fin_flag_number", "rst_flag_number", "TCP", "UDP",
    "Tot sum", "Rate", "Header_Length"
]


def extract_heavy(packet) :
  
    #Extract 23 features for the Heavy Model.
    features = _extract_base(packet)
    values = [features.get(f, 0.0) for f in HEAVY_FEATURES]
    return np.array(values, dtype=np.float32).reshape(1, 1)


def extract_lite(packet) :
    
    # Extract 10 features for the Lite Model.
    features = _extract_base(packet)
    values = [features.get(f, 0.0) for f in LITE_FEATURES]
    return np.array(values, dtype=np.float32).reshape(1, 1)


def _extract_base(packet):

   # Extract all available features from a single packet.   
    f = {}

    if IP not in packet:
        return {k: 0.0 for k in HEAVY_FEATURES}

    ip = packet[IP]
    pkt_len = len(packet)

    #  IP / Protocol 
    f["Header_Length"] = ip.ihl * 4      # IP header length in bytes
    f["Protocol Type"] = ip.proto        # 6=TCP, 17=UDP, 1=ICMP
    f["Tot sum"]       = pkt_len
    f["Min"]           = pkt_len         # single packet: min=max=avg
    f["Max"]           = pkt_len
    f["AVG"]           = pkt_len

    # Approximate rate from packet length
    f["Rate"]          = pkt_len / 1500.0

    #  Protocol onehot flags 
    f["TCP"]  = 1.0 if TCP  in packet else 0.0
    f["UDP"]  = 1.0 if UDP  in packet else 0.0
    f["ICMP"] = 1.0 if ICMP in packet else 0.0

    #  TCP flags 
    f["fin_flag_number"] = 0.0
    f["syn_flag_number"] = 0.0
    f["rst_flag_number"] = 0.0
    f["psh_flag_number"] = 0.0
    f["ack_flag_number"] = 0.0
    f["ece_flag_number"] = 0.0
    f["cwr_flag_number"] = 0.0
    f["ack_count"]       = 0.0
    f["syn_count"]       = 0.0
    f["fin_count"]       = 0.0
    f["rst_count"]       = 0.0

    if TCP in packet:
        tcp = packet[TCP]
        flags = int(tcp.flags)
        f["fin_flag_number"] = 1.0 if flags & 0x01 else 0.0
        f["syn_flag_number"] = 1.0 if flags & 0x02 else 0.0
        f["rst_flag_number"] = 1.0 if flags & 0x04 else 0.0
        f["psh_flag_number"] = 1.0 if flags & 0x08 else 0.0
        f["ack_flag_number"] = 1.0 if flags & 0x10 else 0.0
        f["ece_flag_number"] = 1.0 if flags & 0x40 else 0.0
        f["cwr_flag_number"] = 1.0 if flags & 0x80 else 0.0
        # Cumulative counts, same as flag presence for single packet
        f["syn_count"] = f["syn_flag_number"]
        f["ack_count"] = f["ack_flag_number"]
        f["fin_count"] = f["fin_flag_number"]
        f["rst_count"] = f["rst_flag_number"]

    #  Application protocol hints
    dport = packet[TCP].dport if TCP in packet else (
            packet[UDP].dport if UDP in packet else 0)

    f["HTTP"] = 1.0 if dport in (80, 8080) else 0.0
    f["DNS"]  = 1.0 if dport == 53 else 0.0

    return f