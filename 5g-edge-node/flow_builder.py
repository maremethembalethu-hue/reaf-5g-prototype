#Groups incoming packets into fixed-size WINDOWS of WINDOW_SIZE consecutive packets and 
#extracts the raw per-packet values feature_extractor.py needs to aggregate each window into a named feature vector.


import time
import logging
from scapy.all import IP, TCP, UDP, ARP

from feature_extraction import aggregate_window


WINDOW_SIZE = 30

IDLE_FLUSH_SECONDS = 60.0

ASSUMED_L2_HEADER_LEN = 14
ETHERNET_HEADER_LEN = 14


log = logging.getLogger(__name__)


def wire_size(pkt):
    size = len(pkt)
    
    size += ETHERNET_HEADER_LEN
    return size

def get_flag_values(tcp):
    flags = int(tcp.flags)
    return [
        int((flags & 0x01) != 0),  # FIN
        int((flags & 0x02) != 0),  # SYN
        int((flags & 0x04) != 0),  # RST
        int((flags & 0x08) != 0),  # PSH
        int((flags & 0x10) != 0),  # ACK
        int((flags & 0x20) != 0),  # URG
        int((flags & 0x40) != 0),  # ECE
        int((flags & 0x80) != 0),  # CWR
    ]


def build_packet_row(pkt, ts, last_pac_time):
    has_ip = IP in pkt
    has_arp = (not has_ip) and (ARP in pkt)

    if not has_ip and not has_arp:
        return None, last_pac_time

    iat = 0.0 if last_pac_time is None else max(0.0, float(ts) - float(last_pac_time))

    row = {
        "ts": ts,
        "Header_Length": 0.0,
        "Protocol Type": int(pkt[IP].proto) if has_ip else 0,
        # CICIoT2023's "Duration" column is the IP Time-To-Live, not a time span.
        "Duration": float(pkt[IP].ttl) if has_ip else 0.0,
        "IPv": 1 if has_ip else 0,
        "ARP": 1 if has_arp else 0,
         "Rate": 0.0,
    
        "fin_flag_number": 0,
        "syn_flag_number": 0,
        "rst_flag_number": 0,
        "psh_flag_number": 0,
        "ack_flag_number": 0,
        "ece_flag_number": 0,
        "cwr_flag_number": 0,
        "ack_count": 0,
        "syn_count": 0,
        "fin_count": 0,
        "rst_count": 0,
        "urg_count": 0,

        "Tot size": wire_size(pkt),
        "IAT": iat,
        "Number": 1,
        "TCP": 0,
        "UDP": 0,
        "ICMP": 0,
        "HTTP": 0,
        "HTTPS": 0,
        "SSH": 0,
        "IRC": 0,
        "DNS": 0,
        "Telnet": 0,
        "SMTP": 0,
        "DHCP": 0,
    }

    if not has_ip:
        # ARP-only packet: everything else stays at the defaults above.
        return row, ts

    ip = pkt[IP]
    
    if ip.proto == 1:  # ICMP
        row["ICMP"] = 1

    if ip.proto == 17 and UDP in pkt:
        row["Header_Length"] = 8.0
        row["UDP"] = 1

        udp = pkt[UDP]
        usport, udport = int(udp.sport), int(udp.dport)
        if 53 in (usport, udport):
            row["DNS"] = 1
        if 67 in (usport, udport) or 68 in (usport, udport):
            row["DHCP"] = 1
    elif ip.proto == 6 and TCP in pkt:
        tcp = pkt[TCP]
        dataofs = tcp.dataofs if tcp.dataofs else 5
        row["Header_Length"] = int(dataofs) * 4
        row["TCP"] = 1
        
        sport, dport = int(tcp.sport), int(tcp.dport)
        if 80 in (sport, dport):
            row["HTTP"] = 1
        if 443 in (sport, dport):
            row["HTTPS"] = 1
        if 22 in (sport, dport):
            row["SSH"] = 1
        if sport in (194, 6667, 6697) or dport in (194, 6667, 6697):
            row["IRC"] = 1
        if 53 in (sport, dport):
            row["DNS"] = 1
        if 23 in (sport, dport):
            row["Telnet"] = 1
        if 25 in (sport, dport):
            row["SMTP"] = 1

        flag_values = get_flag_values(tcp)
        row["fin_flag_number"] = flag_values[0]
        row["syn_flag_number"] = flag_values[1]
        row["rst_flag_number"] = flag_values[2]
        row["psh_flag_number"] = flag_values[3]
        row["ack_flag_number"] = flag_values[4]
        row["ece_flag_number"] = flag_values[6]
        row["cwr_flag_number"] = flag_values[7]

        row["ack_count"] = row["ack_flag_number"]
        row["syn_count"] = row["syn_flag_number"]
        row["fin_count"] = row["fin_flag_number"]
        row["rst_count"] = row["rst_flag_number"]
        row["urg_count"] = flag_values[5]
      

    return row, ts


# LITE_FEATURES = [
#     "Tot size", "Protocol Type", 
#     "fin_flag_number", "syn_flag_number",
#      "Header_Length","UDP",
#       "Min", "Max", "AVG",
#       "Number","Std","TCP",]

# HEAVY_FEATURES = [
#     "Header_Length", "Protocol Type", 
#     "fin_flag_number", "syn_flag_number", "rst_flag_number",
#     "psh_flag_number", "ack_flag_number", 
#     "cwr_flag_number", "ack_count",
#      "HTTP", "HTTPS", "IAT",
#     "SSH", "IRC", "TCP", "UDP",  "ICMP",
#        "Tot sum", "Min", "Max", "AVG",
#       "Number", "Variance",]

class WindowBuilder:
    #Buffers packets into fixed-size windows and hands back an aggregated window record once WINDOW_SIZE packets have arrived.
    def __init__(self, window_size=WINDOW_SIZE, idle_flush_seconds=IDLE_FLUSH_SECONDS):
        self.window_size = window_size
        self.idle_flush_seconds = idle_flush_seconds
        self._rows = []
        self._packets = []
        self._last_pac_time = None
        self._identity = []

    def add_packet(self, pkt, ts=None):
        #Feed one packet in. Returns a finished window record once window_size packets have been buffered, else None.
        if IP not in pkt:
            return None
        ts = ts if ts is not None else time.time()
        
        row, self._last_pac_time = build_packet_row(pkt, ts, self._last_pac_time)
        self._rows.append(row)
        self._packets.append(pkt)
        
        self._identity.append({
            "flow_id": getattr(pkt, "flow_id", None),
            "packet_id": getattr(pkt, "packet_id", None),
        })

        if len(self._rows) >= self.window_size:
            return self._finalize()
        return None

    # def expire_stale_partial_window(self, now=None):
    #     #Call periodically to flush a trailing partial window during a traffic.         
	    
    #     if not self._rows:
    #         return None
        
    #     now = now if now is not None else time.time()
    #     if now - self._rows[-1]["ts"] >= self.idle_flush_seconds:
    #         return self._finalize()
    #     return None

    def flush(self):
        #Unconditionally finalize whatever's currently buffered, even if it's short of window_size.
        if self._rows:
            return self._finalize()
        return None

    def _finalize(self):
        features = aggregate_window(self._rows)
        features["flow_duration"] = self._rows[-1]["ts"] - self._rows[0]["ts"]
        features["Covariance"] = features.get("Variance", 0.0)
        features["Drate"] = features.get("Rate", 0.0)
        flow_ids = sorted({i["flow_id"] for i in self._identity if i["flow_id"] is not None})
        packet_ids = sorted({i["packet_id"] for i in self._identity if i["packet_id"] is not None})
        mixed_flow = len(flow_ids) > 1

        if mixed_flow:
            log.warning(
                f"Window mixes packets from {len(flow_ids)} different flow_ids "
                
            )

        window = {
            "packet_count": len(self._rows),
            "start_time": self._rows[0]["ts"],
            "last_time": self._rows[-1]["ts"],
            "proto": features["Protocol Type"],
            "packets": self._packets,
            "features": features,
            "flow_id": flow_ids[0] if len(flow_ids) == 1 else None,
            "flow_ids": flow_ids,
            "mixed_flow": mixed_flow,
            "packet_ids": packet_ids,
        }
        self._rows = []
        self._packets = []
        self._identity = []
        return window