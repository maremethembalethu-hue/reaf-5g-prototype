
import os
import json
import time
 
PACKET_TRACE_ENABLED = os.getenv("PACKET_TRACE", "0") == "1"
TRACE_LOG_PATH = os.getenv("TRACE_LOG_PATH", os.path.join(
    os.getenv("EVAL_DIR", "/evaluation"), "packet_trace.jsonl"))
MAX_TRACE_RECORDS = int(os.getenv("MAX_TRACE_RECORDS", "2000"))
 
_trace_count = 0
_cap_notice_shown = False
 
 
def trace(stage, **fields):
    global _trace_count, _cap_notice_shown
    if not PACKET_TRACE_ENABLED:
        return
    if _trace_count >= MAX_TRACE_RECORDS:
        if not _cap_notice_shown:
            print(f"[packet_trace] reached MAX_TRACE_RECORDS={MAX_TRACE_RECORDS}, "
                  f"no further records written to {TRACE_LOG_PATH}")
            _cap_notice_shown = True
        return
    record = {"ts": time.time(), "stage": stage, **fields}
    try:
        os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)
        with open(TRACE_LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
        _trace_count += 1
    except Exception as e:
        print(f"[packet_trace] failed to write: {e}")
 