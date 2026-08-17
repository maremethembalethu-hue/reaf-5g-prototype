
# Wraps an original PCAP packet's raw bytes inside a small
# header so it can be carried across the 5G tunnel as UDP payload without
# disturbing the packet it is carrying.

import struct
import time

MAGIC = b"RE5G"
VERSION = 1

HEADER_FMT = "!4sBIIdHHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def build_envelope_chunks(flow_id, packet_id, timestamp, linktype, payload, max_chunk_payload=1200):
    # Split payload into one or more envelope-wrapped chunks, each small enough to fit in a single
    # outer UDP datagram without IP fragmentation.

    if max_chunk_payload <= 0:
        raise ValueError("max_chunk_payload must be positive")

    if len(payload) == 0:
        chunk_payloads = [b""]
    else:
        chunk_payloads = [
            payload[i:i + max_chunk_payload]
            for i in range(0, len(payload), max_chunk_payload)
        ]

    chunk_count = len(chunk_payloads)
    out = []
    for idx, chunk_payload in enumerate(chunk_payloads):
        header = struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            flow_id & 0xFFFFFFFF,
            packet_id & 0xFFFFFFFF,
            float(timestamp),
            linktype & 0xFFFF,
            idx,
            chunk_count,
            len(chunk_payload),
        )
        out.append(header + chunk_payload)
    return out

