
import time
import logging
from scapy.all import IP, TCP, UDP, ARP
 
from feature_extraction import aggregate_window
 

WINDOW_SIZE = 10
 
IDLE_FLUSH_SECONDS = 60.0
 
ETHERNET_HEADER_LEN = 14
 
 
log = logging.getLogger(__name__)
 
 
def wire_size(pkt):
    size = len(pkt)
    if getattr(pkt, "linktype", 1) == 1:
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
        return row, ts
 
    ip = pkt[IP]
 
    if ip.proto == 1:  # ICMP
        row["ICMP"] = 1
        row["Header_Length"] = 4.0
 
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
 
    return row, ts
 
 
class WindowBuilder:
    
    def __init__(self, window_size=WINDOW_SIZE, idle_flush_seconds=IDLE_FLUSH_SECONDS):
        self.window_size = window_size
        self.idle_flush_seconds = idle_flush_seconds
        self._rows = []
        self._packets = []
        self._last_pac_time = None
        self._identity = []
        self._window_start_ts = None    # arrival-order: embedded ts of the FIRST packet added
        self._true_min_ts = None        # chronological: min embedded ts seen so far
        self._true_max_ts = None        # chronological: max embedded ts seen so far
        self._real_window_opened_at = None 
 
    def add_packet(self, pkt, ts=None):
        if IP not in pkt:
            return None
        ts = ts if ts is not None else time.time()
 
        if self._window_start_ts is None:
            self._window_start_ts = ts
        if self._real_window_opened_at is None:
            self._real_window_opened_at = time.time()
        if self._true_min_ts is None:
            self._true_min_ts, self._true_max_ts = ts, ts
        else:
            if ts < self._true_min_ts:
                self._true_min_ts = ts
            if ts > self._true_max_ts:
                self._true_max_ts = ts
 
        row, self._last_pac_time = build_packet_row(pkt, ts, self._last_pac_time)
        self._rows.append(row)
        self._packets.append(pkt)
 
        self._identity.append({
            "flow_id": getattr(pkt, "flow_id", None),
            "packet_id": getattr(pkt, "packet_id", None),
        })
 
        if len(self._rows) >= self.window_size:
            return self._finalize(reason="window_size")
        return None
 
    def expire_stale_partial_window(self):
        # Uses REAL wall-clock time only
        if not self._rows:
            return None
        if time.time() - self._real_window_opened_at >= self.idle_flush_seconds:
            return self._finalize(reason="idle_stale")
        return None
 
    def flush(self):
        if self._rows:
            return self._finalize(reason="shutdown_flush")
        return None
 
    def _finalize(self, reason="unknown"):
        features = aggregate_window(self._rows)
        features["flow_duration"] = self._rows[-1]["ts"] - self._rows[0]["ts"]
 
        # Arrival-order span Counted and surfaced here
        # rather than silently baked into a wrong feature value.
        ts_values = [r["ts"] for r in self._rows]
        true_min_ts, true_max_ts = self._true_min_ts, self._true_max_ts
        out_of_order_count = sum(
            1 for i in range(1, len(ts_values)) if ts_values[i] < ts_values[i - 1]
        )
        arrival_order_span = self._rows[-1]["ts"] - self._rows[0]["ts"]
 
        flow_ids = sorted({i["flow_id"] for i in self._identity if i["flow_id"] is not None})
        packet_ids = sorted({i["packet_id"] for i in self._identity if i["packet_id"] is not None})
        mixed_flow = len(flow_ids) > 1
 
        # Mixing flows within a window is EXPECTED and matches training
        
        if mixed_flow:
            log.debug(f"Window spans {len(flow_ids)} different flow_ids (expected — "
                      f"windows are a fixed packet count, not per-flow)")
        if out_of_order_count > 0:
            log.warning(
                f"Window has {out_of_order_count}/{len(ts_values)} packets out of "
                f"chronological order (flow_id={flow_ids}). Arrival-order span="
                f"{arrival_order_span:.3f}s, true chronological span="
                f"{true_max_ts - true_min_ts:.3f}s. Window CLOSING is now based on "
                f"true chronological span, so grouping should still be correct — "
                f"this warning just quantifies how scrambled arrival order is."
            )
 
        window = {
            "packet_count": len(self._rows),
            "start_time": self._rows[0]["ts"],
            "last_time": self._rows[-1]["ts"],
            "close_reason": reason,   # "window_size" | "idle_stale" | "shutdown_flush"
            "true_chronological_span": true_max_ts - true_min_ts,
            "arrival_order_span": arrival_order_span,
            "out_of_order_packets": out_of_order_count,
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
        self._window_start_ts = None
        self._real_window_opened_at = None
        self._true_min_ts = None
        self._true_max_ts = None
        return window