# Aggregates individual packets into flow records, the same unit of analysis CICIoT2023 and CSE-CIC-IDS2018 were built on. A "flow" is identified by the
# 5-tuple (protocol, endpoint A, endpoint B) with both directions folded into one record

import time
from scapy.all import IP, TCP, UDP

MAX_PACKETS_PER_FLOW    = 20
IDLE_TIMEOUT_SECONDS     = 2.0
MAX_FLOW_DURATION_SECONDS = 30.0
MAX_STORED_PACKETS       = 200

TCP_FLAG_BITS = {
    "fin": 0x01, "syn": 0x02, "rst": 0x04, "psh": 0x08,
    "ack": 0x10, "urg": 0x20, "ece": 0x40, "cwr": 0x80,
}


def flow_key(pkt):
   # 5-tuple flow key, both directions of a conversation map to the same flow (protocol, sorted (ip, port) pair, sorted (ip, port) pair).
    ip = pkt[IP]
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        sport, dport = 0, 0
    a, b = (ip.src, sport), (ip.dst, dport)
    endpoints = tuple(sorted([a, b]))
    return (ip.proto, endpoints[0], endpoints[1])


def new_flow(pkt, ts):
    ip = pkt[IP]
    if TCP in pkt:
        sport = pkt[TCP].sport
    elif UDP in pkt:
        sport = pkt[UDP].sport
    else:
        sport = 0
    return {
        "start_time": ts,
        "last_time": ts,
        "proto": ip.proto,
        "initiator": (ip.src, sport),   # defines the "forward" direction
        "packet_count": 0,
        "lengths": [],
        "header_lengths": [],
        "interarrival": [],
        "fwd_count": 0, "bwd_count": 0,
        "fwd_bytes": 0, "bwd_bytes": 0,
        "flags": {k: 0 for k in TCP_FLAG_BITS},
        "packets": [],
    }


class FlowTable:
    def __init__(self, max_packets=MAX_PACKETS_PER_FLOW,
                 idle_timeout=IDLE_TIMEOUT_SECONDS,
                 max_duration=MAX_FLOW_DURATION_SECONDS):
        self.flows = {}
        self.max_packets = max_packets
        self.idle_timeout = idle_timeout
        self.max_duration = max_duration

    def add_packet(self, pkt, ts=None):
        # Feed one packet in. Returns a finished flow record if this packet completed a flow else None.
        if IP not in pkt:
            return None
        ts = ts if ts is not None else time.time()
        key = flow_key(pkt)

        flow = self.flows.get(key)
        if flow is None:
            flow = new_flow(pkt, ts)
            self.flows[key] = flow

        ip = pkt[IP]
        length = len(pkt)

        if flow["packet_count"] > 0:
            flow["interarrival"].append(ts - flow["last_time"])
        flow["last_time"] = ts
        flow["packet_count"] += 1
        flow["lengths"].append(length)
        flow["header_lengths"].append(ip.ihl * 4)

        if TCP in pkt:
            tcp_flags = int(pkt[TCP].flags)
            for name, bit in TCP_FLAG_BITS.items():
                if tcp_flags & bit:
                    flow["flags"][name] += 1
            sport = pkt[TCP].sport
        elif UDP in pkt:
            sport = pkt[UDP].sport
        else:
            sport = 0

        if (ip.src, sport) == flow["initiator"]:
            flow["fwd_count"] += 1
            flow["fwd_bytes"] += length
        else:
            flow["bwd_count"] += 1
            flow["bwd_bytes"] += length

        if len(flow["packets"]) < MAX_STORED_PACKETS:
            flow["packets"].append(pkt)

        duration = flow["last_time"] - flow["start_time"]
        if flow["packet_count"] >= self.max_packets or duration >= self.max_duration:
            del self.flows[key]
            return flow
        return None

    def expire_stale_flows(self, now=None):
       # Call periodically to finalize flows that have gone idle. Returns a list of finished flow records.
        now = now if now is not None else time.time()
        finished = []
        for key in list(self.flows.keys()):
            flow = self.flows[key]
            if now - flow["last_time"] >= self.idle_timeout:
                finished.append(flow)
                del self.flows[key]
        return finished
