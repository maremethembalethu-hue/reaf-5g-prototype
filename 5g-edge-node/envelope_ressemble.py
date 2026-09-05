
import struct
import time

MAGIC = b"RE5G"
VERSION = 1

HEADER_FMT = "!4sBIIdHHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def parse_envelope_chunk(raw):
    # Decode a single envelope chunk. 
    if len(raw) < HEADER_SIZE:
        return None

    (magic, version, flow_id, packet_id, timestamp, linktype,
     chunk_index, chunk_count, payload_length) = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])

    if magic != MAGIC or version != VERSION:
        return None

    payload = raw[HEADER_SIZE:HEADER_SIZE + payload_length]
    if len(payload) != payload_length:
        return None  # truncated mid-payload

    return {
        "flow_id": flow_id,
        "packet_id": packet_id,
        "timestamp": timestamp,
        "linktype": linktype,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "payload": payload,
    }

class EnvelopeReassembler:
    # Receiver-side buffer that reassembles chunked envelopes back into
    # one original packet's raw bytes.

    def __init__(self, timeout=10):
        # (flow_id, packet_id) -> {chunks: {idx: bytes}, count, first_seen, timestamp, linktype}
        self._buffers = {}
        self.timeout = timeout

    def feed(self, raw, ts=None):
        # Add one received outer UDP payload. Returns
        # once every chunk for that packet_id has arrived
        
        envelope = parse_envelope_chunk(raw)
        if envelope is None:
            return None

        ts = ts if ts is not None else time.time()
        key = (envelope["flow_id"], envelope["packet_id"])

        if envelope["chunk_count"] == 1:
            self._buffers.pop(key, None)
            return {
                "flow_id": envelope["flow_id"],
                "packet_id": envelope["packet_id"],
                "timestamp": envelope["timestamp"],
                "linktype": envelope["linktype"],
                "payload": envelope["payload"],
            }

        entry = self._buffers.setdefault(key, {
            "chunks": {},
            "count": envelope["chunk_count"],
            "first_seen": ts,
            "timestamp": envelope["timestamp"],
            "linktype": envelope["linktype"],
        })
        entry["chunks"][envelope["chunk_index"]] = envelope["payload"]

        if len(entry["chunks"]) < entry["count"]:
            return None

        self._buffers.pop(key)
        original_bytes = b"".join(entry["chunks"][i] for i in range(entry["count"]))
        return {
            "flow_id": envelope["flow_id"],
            "packet_id": envelope["packet_id"],
            "timestamp": entry["timestamp"],
            "linktype": entry["linktype"],
            "payload": original_bytes,
        }

    def expire_stale(self, now=None):
        # Drops chunk sets whose remaining pieces
        # never arrived, so the buffer can't grow unbounded.
        now = now if now is not None else time.time()
        for key in list(self._buffers.keys()):
            if now - self._buffers[key]["first_seen"] > self.timeout:
                del self._buffers[key]
