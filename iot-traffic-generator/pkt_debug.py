import io
import json
import time
from pathlib import Path

MAX_INSPECT = 10
_count = 0

OUT_DIR = Path(__file__).resolve().parent / "pkt_debug_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT_DIR / "packets.json"
TEXT_PATH = OUT_DIR / "packets.log"

_records = []


def _layers(pkt):
    layers = []
    p = pkt
    while p is not None:
        fields = {}
        try:
            for name, val in p.fields.items():
                fields[name] = str(val)
        except Exception as e:
            fields["_error"] = str(e)
        layers.append({"layer": p.__class__.__name__, "fields": fields})
        p = p.payload if hasattr(p, "payload") and p.payload else None
        if p is not None and p.__class__.__name__ == "NoPayload":
            break
    return layers


def _show_str(pkt):
    buf = io.StringIO()
    try:
        pkt.show(dump=True)
    except TypeError:
        pass
    try:
        return pkt.show(dump=True)
    except Exception as e:
        return f"show() failed: {e}"


def _hexdump(pkt):
    try:
        raw = bytes(pkt)
    except Exception as e:
        return f"hex failed: {e}"
    hexstr = raw.hex()
    return " ".join(hexstr[i:i + 2] for i in range(0, len(hexstr), 2))


_done_printed = False
_locked = False


def is_done():
    return _locked


def inspect_packet(pkt, label=""):
    global _count, _done_printed, _locked
    if _locked:
        return
    if _count >= MAX_INSPECT:
        _locked = True
        if not _done_printed:
            _done_printed = True
            print(f"[pkt_debug] {MAX_INSPECT} packets captured. No further packets will be recorded.")
            print(f"[pkt_debug] Output written to: {OUT_DIR.resolve()}")
            print(f"[pkt_debug]   JSON: {JSON_PATH.resolve()}")
            print(f"[pkt_debug]   LOG:  {TEXT_PATH.resolve()}")
        return
    if _count == 0:
        print(f"[pkt_debug] writing output to: {OUT_DIR.resolve()}")
    _count += 1

    record = {
        "index": _count,
        "label": label,
        "time": getattr(pkt, "time", None),
        "wallclock": time.time(),
        "summary": None,
        "length": None,
        "layers": [],
        "hex": None,
        "show": None,
    }

    try:
        record["summary"] = pkt.summary()
    except Exception as e:
        record["summary"] = f"failed: {e}"

    try:
        record["length"] = len(bytes(pkt))
    except Exception as e:
        record["length"] = f"failed: {e}"

    record["layers"] = _layers(pkt)
    record["hex"] = _hexdump(pkt)
    record["show"] = _show_str(pkt)

    _records.append(record)

    # incremental json write
    try:
        JSON_PATH.write_text(json.dumps(_records, indent=2, default=str))
    except Exception as e:
        print(f"json write failed: {e}")

    # incremental text log
    try:
        with open(TEXT_PATH, "a") as f:
            f.write(f"\n=== [{label}] packet #{_count} ===\n")
            f.write(f"summary: {record['summary']}\n")
            f.write(f"length: {record['length']} time: {record['time']}\n")
            f.write("layers:\n")
            for layer in record["layers"]:
                f.write(f"  {layer['layer']}: {layer['fields']}\n")
            f.write(f"hex: {record['hex']}\n")
            f.write("show:\n")
            f.write(record["show"] or "")
            f.write("\n")
    except Exception as e:
        print(f"text log write failed: {e}")


def reset():
    global _count, _records, _locked, _done_printed
    _count = 0
    _records = []
    _locked = False
    _done_printed = False
    JSON_PATH.write_text("[]")
    TEXT_PATH.write_text("")