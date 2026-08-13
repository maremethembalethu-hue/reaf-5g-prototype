
# Buffers IP fragments and hands back a single reassembled packet once a
# fragment train completes Unfragmented packets pass straight
# through untouched.
 

import time
from scapy.layers.inet import IP, defragment


class FragmentReassembler:
    def __init__(self, timeout=30):
        self._buffers = {}   # (src, dst, proto, ip.id) = {"frags": [...], "first_seen": ts}
        self.timeout = timeout

    def _key(self, ip):
        return (ip.src, ip.dst, ip.proto, ip.id)

    def feed(self, pkt, ts=None):
        # Returns a whole packet, or None if still waiting on more fragments / the set never completed cleanly.
        if IP not in pkt:
            return pkt

        ip = pkt[IP]
        more_fragments = bool(ip.flags & 0x1)   # MF bit
        frag_offset = ip.frag

        if not more_fragments and frag_offset == 0:
            return pkt  # ordinary packet, nothing to reassemble

        ts = ts if ts is not None else time.time()
        key = self._key(ip)
        entry = self._buffers.setdefault(key, {"frags": [], "first_seen": ts})
        entry["frags"].append(pkt)

        if more_fragments:
            return None  # last fragment (MF=0) hasn't arrived yet

        frags = self._buffers.pop(key)["frags"]
        try:
            reassembled = defragment(frags)
        except Exception:
            return None
        return reassembled[0] if len(reassembled) == 1 else None

    def expire_stale(self, now=None):
        # Call periodically drops fragment trains whose final fragment never arrived, so the buffer can't grow unbounded.
        now = now if now is not None else time.time()
        for key in list(self._buffers.keys()):
            if now - self._buffers[key]["first_seen"] > self.timeout:
                del self._buffers[key]