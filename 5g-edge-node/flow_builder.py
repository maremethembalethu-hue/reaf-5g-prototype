#Groups incoming packets into fixed-size WINDOWS of WINDOW_SIZE consecutive packets and 
#extracts the raw per-packet values feature_extractor.py needs to aggregate each window into a named feature vector.


import time
from scapy.all import IP, TCP, UDP

from feature_extractor import aggregate_window


WINDOW_SIZE = 10

IDLE_FLUSH_SECONDS = 2.0

SERVICE_PORTS_NOTE = (
    "IRC below intentionally checks port 21 (FTP's control port, not IRC's) "
    "-- this reproduces a quirk/bug in the ground-truth extractor "
    "(Layered_features.py's L4.IRC()) on purpose, so our numbers match the "
    "training data's."
)


def _is_port(sport, dport, port):
    return 1 if (sport == port or dport == port) else 0


def build_packet_row(pkt, ts, last_pac_time):
    #Computes one row of raw per-packet values a direct port of the per-packet section of Feature_extraction.py's pcap_evaluation() loop.
    
    ip = pkt[IP]
    iat = 0.0 if last_pac_time is None else max(0.0, ts - last_pac_time)

    row = {
        "ts": ts,
        "Header_Length": 0,          # only set for TCP/UDP below; 0 for ICMP etc,
                                      # matching the original code's per-packet reset
        "Protocol Type": ip.proto,
        "Time_To_Live": getattr(ip, "ttl", 0),
        "Rate": 0.0,                 # filled in at window-aggregation time only
        "fin_flag_number": 0, "syn_flag_number": 0, "rst_flag_number": 0,
        "psh_flag_number": 0, "ack_flag_number": 0, "ece_flag_number": 0,
        "cwr_flag_number": 0,
        "ack_count": 0, "syn_count": 0, "fin_count": 0, "rst_count": 0,
        "HTTP": 0, "HTTPS": 0, "DNS": 0, "Telnet": 0, "SMTP": 0, "SSH": 0,
        "IRC": 0, "TCP": 0, "UDP": 0, "DHCP": 0, "ARP": 0, "ICMP": 0,
        "IGMP": 0, "IPv": 1, "LLC": 0,
        "Tot size": len(pkt),        # full captured frame size, matching the
                                      # original extractor's len(buf); no
                                      # Ethernet-header compensation needed here
                                      # since this is real sniffed traffic, not a
                                      # reconstructed IP-only packet
        "IAT": iat,
        "Number": 1,
    }

    if ip.proto == 1:
        row["ICMP"] = 1
    elif ip.proto == 2:
        row["IGMP"] = 1

    if UDP in pkt:
        row["UDP"] = 1
        row["Header_Length"] = 8   # fixed for UDP, matches Connectivity_features_basic
        udp = pkt[UDP]
        sport, dport = udp.sport, udp.dport
        row["DNS"] = _is_port(sport, dport, 53)
        row["DHCP"] = 1 if ((sport, dport) == (67, 68) or (sport, dport) == (68, 67)) else 0

    elif TCP in pkt:
        row["TCP"] = 1
        tcp = pkt[TCP]
        dataofs = tcp.dataofs if tcp.dataofs else 5   # 5 * 4 = 20 bytes, TCP's default
        row["Header_Length"] = int(dataofs) * 4

        flags = tcp.flags
        row["fin_flag_number"] = int(bool(flags & 0x01))
        row["syn_flag_number"] = int(bool(flags & 0x02))
        row["rst_flag_number"] = int(bool(flags & 0x04))
        row["psh_flag_number"] = int(bool(flags & 0x08))
        row["ack_flag_number"] = int(bool(flags & 0x10))
        row["ece_flag_number"] = int(bool(flags & 0x40))
        row["cwr_flag_number"] = int(bool(flags & 0x80))

        # per-packet *_count == the corresponding *_flag_number (0 or 1);
        # the distinction only appears after window aggregation, where
        # *_count gets SUMMED and *_flag_number gets AVERAGED (see
        # aggregate_window() in feature_extractor.py).
        row["ack_count"] = row["ack_flag_number"]
        row["syn_count"] = row["syn_flag_number"]
        row["fin_count"] = row["fin_flag_number"]
        row["rst_count"] = row["rst_flag_number"]

        sport, dport = tcp.sport, tcp.dport
        row["HTTP"] = _is_port(sport, dport, 80)
        row["HTTPS"] = _is_port(sport, dport, 443)
        row["SSH"] = _is_port(sport, dport, 22)
        row["IRC"] = _is_port(sport, dport, 21)  # see SERVICE_PORTS_NOTE above
        row["Telnet"] = _is_port(sport, dport, 23)
        row["SMTP"] = _is_port(sport, dport, 25)

    return row, ts


class WindowBuilder:
    #Buffers packets into fixed-size windows and hands back an aggregated window record once WINDOW_SIZE packets have arrived.
    def __init__(self, window_size=WINDOW_SIZE, idle_flush_seconds=IDLE_FLUSH_SECONDS):
        self.window_size = window_size
        self.idle_flush_seconds = idle_flush_seconds
        self._rows = []
        self._packets = []
        self._last_pac_time = None

    def add_packet(self, pkt, ts=None):
        #Feed one packet in. Returns a finished window record once window_size packets have been buffered, else None.
        if IP not in pkt:
            return None
        ts = ts if ts is not None else time.time()

        row, self._last_pac_time = build_packet_row(pkt, ts, self._last_pac_time)
        self._rows.append(row)
        self._packets.append(pkt)

        if len(self._rows) >= self.window_size:
            return self._finalize()
        return None

    def expire_stale_partial_window(self, now=None):
        #Call periodically to flush a trailing partial window during a traffic.         
	    
        if not self._rows:
            return None
        
        now = now if now is not None else time.time()
        if now - self._rows[-1]["ts"] >= self.idle_flush_seconds:
            return self._finalize()
        return None

    def flush(self):
        #Unconditionally finalize whatever's currently buffered, even if it's short of window_size.
        if self._rows:
            return self._finalize()
        return None

    def _finalize(self):
        features = aggregate_window(self._rows)
        window = {
            "packet_count": len(self._rows),
            "start_time": self._rows[0]["ts"],
            "last_time": self._rows[-1]["ts"],
            "proto": features["Protocol Type"],
            "packets": self._packets,
            "features": features,
        }
        self._rows = []
        self._packets = []
        return window