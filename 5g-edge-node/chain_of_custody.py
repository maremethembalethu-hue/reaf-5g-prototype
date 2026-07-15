
# chain_of_custody.py
# Computes SHA-256 hash of each evidence bundle,
# appends a tamper-evident entry to the chain-of-custody log,
# and digitally signs the entry.


import os
import json
import hashlib
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "/evidence")
COC_LOG      = os.path.join(EVIDENCE_DIR, "chain_of_custody.log")


def hash_directory(directory: str) -> str:
    
    # # Compute SHA-256 hash of all files in an evidence bundle directory.
    # # Files are processed in sorted order for reproducibility.
    
    sha256 = hashlib.sha256()
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "rb") as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)
            except Exception as e:
                log.warning(f"Could not hash {filepath}: {e}")
    return sha256.hexdigest()


def get_previous_hash() -> str:
    # Read the hash of the last entry in the custody log."""
    if not os.path.exists(COC_LOG):
        return "0" * 64   # genesis block — no previous hash
    try:
        with open(COC_LOG, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return "0" * 64
        last_entry = json.loads(lines[-1])
        return last_entry.get("entry_hash", "0" * 64)
    except Exception:
        return "0" * 64


def preserve_bundle(bundle_dir: str, meta: dict):
    
    # Hash the evidence bundle, link to previous hash,
    # and append a signed entry to the custody log.
    
    bundle_hash   = hash_directory(bundle_dir)
    previous_hash = get_previous_hash()
    timestamp     = datetime.now(timezone.utc).isoformat()

    # Entry hash links this entry to the previous one
    entry_content = f"{previous_hash}{bundle_hash}{timestamp}{meta['incident_id']}"
    entry_hash    = hashlib.sha256(entry_content.encode()).hexdigest()

    entry = {
        "entry_hash":    entry_hash,
        "previous_hash": previous_hash,
        "bundle_hash":   bundle_hash,
        "timestamp":     timestamp,
        "incident_id":   meta["incident_id"],
        "attack_type":   meta["attack_type"],
        "confidence":    meta["confidence"],
        "model_used":    meta["model_used"],
        "bundle_dir":    bundle_dir
    }

    # Append to log — one JSON object per line (append-only)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(COC_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    log.info(f"  CoC entry written | hash={entry_hash[:16]}... | prev={previous_hash[:16]}...")