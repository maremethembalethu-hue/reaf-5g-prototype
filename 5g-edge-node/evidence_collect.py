# Captures all four evidence layers when an attack is detected.

import os
import json
import time
import logging
from datetime import datetime, timezone

import psutil
from scapy.all import wrpcap

from chain_of_custody import preserve_bundle
from log_buffer import recent_log_handler

log = logging.getLogger(__name__)

EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "/evidence")
MB = 1024 * 1024


def _save_network(bundle_dir, packets):
    path = os.path.join(bundle_dir, "network_capture.pcap")
    wrpcap(path, packets)
    log.info(f"  1/4 Network layer saved: {path}")
    return path


def _save_processes(bundle_dir, timestamp):
    path = os.path.join(bundle_dir, "processes.json")
    processes = []
    for proc in psutil.process_iter(["pid", "name", "status", "cpu_percent", "memory_percent"]):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    with open(path, "w") as f:
        json.dump({"timestamp": timestamp, "processes": processes}, f, indent=2)
    log.info(f"  2/4 Process layer saved: {path} ({len(processes)} processes)")
    return path


def _save_memory(bundle_dir, timestamp):
    path = os.path.join(bundle_dir, "memory.json")
    vm, sm = psutil.virtual_memory(), psutil.swap_memory()
    snapshot = {
        "timestamp": timestamp,
        "total_mb": vm.total // MB,
        "available_mb": vm.available // MB,
        "used_mb": vm.used // MB,
        "percent": vm.percent,
        "swap_total_mb": sm.total // MB,
        "swap_used_mb": sm.used // MB,
        "swap_percent": sm.percent,
    }
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    log.info(f"  3/4 Memory layer saved: {path}")
    return path


def _save_syslog(bundle_dir):
    path = os.path.join(bundle_dir, "syslog.txt")
    lines = recent_log_handler.get_recent(100)
    with open(path, "w") as f:
        f.write("\n".join(lines) if lines else "No log entries captured yet")
    log.info(f"  4/4 System layer saved: {path} ({len(lines)} lines)")
    return path


def collect_evidence(packets, result, timestamp, stage_ts=None):
    # Called when trigger.py fires. Collects the four evidence layers and preserves them.
    # stage_ts: optional dict that is filled with wall-clock times (seconds) of the internal steps,
    # so capture.py can write them to timing_log.jsonl. Return value is unchanged: the incident id.
    stage_ts = stage_ts if stage_ts is not None else {}
    incident_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    bundle_dir = os.path.join(EVIDENCE_DIR, f"incident_{incident_id}")
    os.makedirs(bundle_dir, exist_ok=True)

    log.info(f"EVIDENCE ACQUISITION STARTED | incident={incident_id} | attack={result['attack_type']}")

    layers = [
        ("network", lambda: _save_network(bundle_dir, packets)),
        ("processes", lambda: _save_processes(bundle_dir, timestamp)),
        ("memory", lambda: _save_memory(bundle_dir, timestamp)),
        ("syslog", lambda: _save_syslog(bundle_dir)),
    ]
    paths = {}
    for name, save in layers:
        try:
            paths[name] = save()
        except Exception as e:
            log.error(f"  {name} layer failed: {e}")

    meta = {
        "incident_id": incident_id,
        "timestamp": timestamp,
        "attack_type": result["attack_type"],
        "confidence": result["confidence"],
        "model_used": result["model_used"],
        "cpu_percent": result.get("cpu_percent"),
        "ram_percent": result.get("ram_percent"),
        "packet_count": len(packets),
        "evidence_files": paths,
    }
    with open(os.path.join(bundle_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # preserve_bundle hashes the bundle and writes the custody entry; both happen inside this call.
    stage_ts["bundle_written_ts"] = time.time()
    preserve_bundle(bundle_dir, meta)
    stage_ts["custody_done_ts"] = time.time()

    log.info(f"EVIDENCE ACQUISITION COMPLETE | incident={incident_id}")
    return incident_id
