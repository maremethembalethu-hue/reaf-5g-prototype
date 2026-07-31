
# Appends one record per classified flow to a JSONL log, regardless of
# whether it triggered evidence acquisition

import os
import json
from pathlib import Path

PREDICTIONS_LOG = Path(os.getenv("PREDICTIONS_LOG", "/evidence/predictions_log.jsonl"))


def log_prediction(flow: dict, result: dict):
    record = {
        "flow_start_time": flow["start_time"],
        "flow_last_time": flow["last_time"],
        "packet_count": flow["packet_count"],
        "proto": flow["proto"],
        "attack_type": result["attack_type"],
        "confidence": result["confidence"],
        "model_used": result["model_used"],
    }
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
