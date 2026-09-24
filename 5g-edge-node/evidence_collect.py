# Captures all four evidence layers when an attack is detected.

import os
import json
import logging
import subprocess
from datetime import datetime, timezone

import psutil
from scapy.all import wrpcap

from chain_of_custody import preserve_bundle

log = logging.getLogger(__name__)

EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "/evidence")


def collect_evidence(packets, result, timestamp):
    
    # Called when trigger.py fires.
    # Collects all four evidence layers and preserves them.
    
    incident_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    bundle_dir  = os.path.join(EVIDENCE_DIR, f"incident_{incident_id}")
    os.makedirs(bundle_dir, exist_ok=True)

    log.info(f"EVIDENCE ACQUISITION STARTED | incident={incident_id} | attack={result['attack_type']}")

    paths = {}

    #  Layer 1: Network 
    pcap_path = os.path.join(bundle_dir, "network_capture.pcap")
    try:
        wrpcap(pcap_path, packets)
        paths["network"] = pcap_path
        log.info(f"  1/4 Network layer saved: {pcap_path}")
    except Exception as e:
        log.error(f"  1/4 Network capture failed: {e}")

    #  Layer 2: Process 
    proc_path = os.path.join(bundle_dir, "processes.json")
    try:
        processes = []
        for proc in psutil.process_iter(["pid", "name", "status", "cpu_percent", "memory_percent"]):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        with open(proc_path, "w") as f:
            json.dump({"timestamp": timestamp, "processes": processes}, f, indent=2)
        paths["processes"] = proc_path
        log.info(f"  2/4 Process layer saved: {proc_path} ({len(processes)} processes)")
    except Exception as e:
        log.error(f" 2/4 Process capture failed: {e}")

    #  Layer 3: Memory 
    mem_path = os.path.join(bundle_dir, "memory.json")
    try:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        memory_snapshot = {
            "timestamp":       timestamp,
            "total_mb":        vm.total  // (1024 * 1024),
            "available_mb":    vm.available // (1024 * 1024),
            "used_mb":         vm.used   // (1024 * 1024),
            "percent":         vm.percent,
            "swap_total_mb":   sm.total  // (1024 * 1024),
            "swap_used_mb":    sm.used   // (1024 * 1024),
            "swap_percent":    sm.percent
        }
        with open(mem_path, "w") as f:
            json.dump(memory_snapshot, f, indent=2)
        paths["memory"] = mem_path
        log.info(f"  3/4 Memory layer saved: {mem_path}")
    except Exception as e:
        log.error(f"  3/4 Memory capture failed: {e}")

    #  Layer 4: System logs 
    from log_buffer import recent_log_handler
 
    syslog_path = os.path.join(bundle_dir, "syslog.txt")
    try:
        recent_lines = recent_log_handler.get_recent(100)
        with open(syslog_path, "w") as f:
            f.write("\n".join(recent_lines) if recent_lines else "No log entries captured yet")
        paths["syslog"] = syslog_path
        log.info(f"  4/4 System layer saved: {syslog_path} ({len(recent_lines)} lines)")
    except Exception as e:
        log.warning(f"  4/4 Syslog capture failed: {e}")

    #  Metadata file 
    meta = {
        "incident_id":  incident_id,
        "timestamp":    timestamp,
        "attack_type":  result["attack_type"],
        "confidence":   result["confidence"],
        "model_used":   result["model_used"],
        "ram_percent":  result["ram_percent"],
        "packet_count": len(packets),
        "evidence_files": paths
    }
    meta_path = os.path.join(bundle_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    #  Chain of custody
    preserve_bundle(bundle_dir, meta)
    log.info(f"EVIDENCE ACQUISITION COMPLETE | incident={incident_id}")
    
    return incident_id  