
# Appends one record per classified flow to a JSONL log, regardless of
# whether it triggered evidence acquisition

import os
import json
from pathlib import Path

PREDICTIONS_LOG = Path(os.getenv("PREDICTIONS_LOG", "/evaluation/predictions_log.jsonl"))
PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)



def log_prediction(flow, result):
    record = {
        "flow_start_time": flow["start_time"],
        "flow_last_time": flow["last_time"],

        "packet_count": flow["packet_count"],
        "proto": flow["proto"],
        "attack_type": result["attack_type"],
        "confidence": result["confidence"],
        "model_used": result["model_used"],
    }

    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")