
import os
import json
import base64
import hashlib
import logging
import fcntl
from datetime import datetime, timezone
 
log = logging.getLogger(__name__)
 
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "/evidence")
COC_LOG      = os.path.join(EVIDENCE_DIR, "chain_of_custody.log")
ANCHOR_LOG   = os.getenv("ANCHOR_LOG", os.path.join(EVIDENCE_DIR, "chain_anchors.jsonl"))
ANCHOR_EVERY = int(os.getenv("ANCHOR_EVERY", "0"))
CUSTODY_FSYNC = os.getenv("CUSTODY_FSYNC", "false").lower() == "true"
 
USE_RFC3161 = os.getenv("USE_RFC3161", "false").lower() == "true"
TSA_URL     = os.getenv("TSA_URL", "https://freetsa.org/tsr")
TSA_TIMEOUT = float(os.getenv("TSA_TIMEOUT", "5"))
 
KEY_DIR          = os.getenv("KEY_DIR", "/etc/reaf")
SIGNING_KEY_PATH = os.getenv("SIGNING_KEY_PATH", os.path.join(KEY_DIR, "custody_signing_key.pem"))
 
SIG_ALG = "Ed25519"
_key_cache = {}
 
 
# - signing key
def _load_or_create_key():
    """Return (private_key, public_key_raw_bytes, key_id). Creates a key on first use."""
    if "k" in _key_cache:
        return _key_cache["k"]
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 
    path = SIGNING_KEY_PATH
    if os.path.exists(path):
        with open(path, "rb") as f:
            priv = serialization.load_pem_private_key(f.read(), password=None)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        priv = Ed25519PrivateKey.generate()
        pem = priv.private_bytes(serialization.Encoding.PEM,
                                 serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        log.warning(f"Generated new custody signing key at {path}. "
                    f"Publish the public key (export_public_key) OUTSIDE this node.")
    pub_raw = priv.public_key().public_bytes(serialization.Encoding.Raw,
                                             serialization.PublicFormat.Raw)
    key_id = hashlib.sha256(pub_raw).hexdigest()
    _key_cache["k"] = (priv, pub_raw, key_id)
    return _key_cache["k"]
 
 
def export_public_key(dest_path):
    """Write the public key (PEM) so a verifier can pin it. Returns key_id."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    _, pub_raw, key_id = _load_or_create_key()
    pem = Ed25519PublicKey.from_public_bytes(pub_raw).public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    with open(dest_path, "wb") as f:
        f.write(pem)
    return key_id
 
 
def canonical(obj):
    """Deterministic bytes for signing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
 
 
def _sign(data):
    priv, _, key_id = _load_or_create_key()
    return priv.sign(data).hex(), key_id
 
 
# Fields covered by the signature. entry_hash itself only covers four fields, so
# signing the whole core also protects attack_type, confidence, model_used, bundle_dir.
SIGNED_FIELDS = ("entry_hash", "previous_hash", "bundle_hash", "timestamp",
                 "incident_id", "attack_type", "confidence", "model_used",
                 "bundle_dir", "key_id")
 
 
def signed_core(entry):
    return {k: entry.get(k) for k in SIGNED_FIELDS}
 
 
#  hashing
def hash_directory(directory):
    # SHA-256 over all files of an evidence bundle, sorted for reproducibility.
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
 
 
def _read_entries():
    if not os.path.exists(COC_LOG):
        return []
    with open(COC_LOG, "r") as f:
        return [json.loads(l) for l in f if l.strip()]
 
 
def get_previous_hash():
    if not os.path.exists(COC_LOG):
        return "0" * 64
    try:
        with open(COC_LOG, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return "0" * 64
        return json.loads(lines[-1]).get("entry_hash", "0" * 64)
    except Exception as e:
        # Restarting at genesis silently would break the chain; make it loud.
        log.error(f"Could not read last custody entry, restarting chain at genesis: {e}")
        return "0" * 64
 
 
# -- trusted timestamp
def _rfc3161_timestamp(digest):
    """RFC 3161 token over `digest` (raw sha256 bytes). Never raises: a failed
    timestamp must not block evidence preservation, but the failure is recorded."""
    if not USE_RFC3161:
        return {"trusted": False, "reason": "disabled"}
    try:
        import rfc3161ng
        stamper = rfc3161ng.RemoteTimestamper(TSA_URL, hashname="sha256", timeout=TSA_TIMEOUT)
        token = stamper.timestamp(digest=digest, include_tsa_certificate=True)
        return {"trusted": True, "tsa_url": TSA_URL, "hash_alg": "sha256",
                "digest_hex": digest.hex(),
                "token_b64": base64.b64encode(token).decode()}
    except Exception as e:
        log.warning(f"RFC3161 timestamp unavailable: {e}")
        return {"trusted": False, "reason": f"{type(e).__name__}: {e}"}
 
 
# "$PYTHON" entries
def preserve_bundle(bundle_dir, meta):
    # Hash the bundle, link to the previous entry, sign, (optionally) time-stamp,
    # and append. The read-previous/append step is under a file lock so concurrent
    # incidents cannot fork the chain.
    bundle_hash = hash_directory(bundle_dir)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(COC_LOG + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous_hash = get_previous_hash()
        timestamp     = datetime.now(timezone.utc).isoformat()
 
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
            "bundle_dir":    bundle_dir,
        }
        try:
            _, _, key_id = _load_or_create_key()
            entry["key_id"] = key_id
            entry["signature_alg"] = SIG_ALG
            entry["signature"], _ = _sign(canonical(signed_core(entry)))
        except Exception as e:
            log.error(f"Signing failed, entry will be UNSIGNED: {e}")
            entry["signature"] = None
            entry["signature_error"] = f"{type(e).__name__}: {e}"
 
        entry["trusted_timestamp"] = _rfc3161_timestamp(bytes.fromhex(entry_hash))
 
        with open(COC_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            if CUSTODY_FSYNC:
                os.fsync(f.fileno())
        count = len(_read_entries()) if ANCHOR_EVERY else 0
 
    log.info(f"  CoC entry written | hash={entry_hash[:16]}... | prev={previous_hash[:16]}... "
             f"| signed={bool(entry.get('signature'))} | tsa={entry['trusted_timestamp'].get('trusted')}")
    if ANCHOR_EVERY and count % ANCHOR_EVERY == 0:
        anchor_chain(reason=f"every_{ANCHOR_EVERY}")
    return entry
 
 
# "$PYTHON"- anchors
def anchor_chain(reason="manual"):
    """Sign (and time-stamp) the current chain head and append it to ANCHOR_LOG.
    Verification later proves: the log held >= entry_count entries, and entry
    entry_count-1 had hash head_hash, as of anchored_at. This is what exposes
    tail truncation and rewrite-and-rehash of anchored entries."""
    try:  # make the anchored head durable on disk
        fd = os.open(COC_LOG, os.O_RDONLY); os.fsync(fd); os.close(fd)
    except OSError:
        pass
    entries = _read_entries()
    if not entries:
        return None
    _, _, key_id = _load_or_create_key()
    record = {"anchored_at": datetime.now(timezone.utc).isoformat(),
              "entry_count": len(entries),
              "head_hash": entries[-1]["entry_hash"],
              "key_id": key_id, "reason": reason}
    record["signature_alg"] = SIG_ALG
    record["signature"], _ = _sign(canonical(record))
    digest = hashlib.sha256(canonical({k: record[k] for k in
             ("anchored_at", "entry_count", "head_hash", "key_id", "reason")})).digest()
    record["trusted_timestamp"] = _rfc3161_timestamp(digest)
    os.makedirs(os.path.dirname(ANCHOR_LOG) or ".", exist_ok=True)
    with open(ANCHOR_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
    log.info(f"  Chain anchored | count={record['entry_count']} head={record['head_hash'][:16]}...")
    return record
 
 
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) >= 2 and sys.argv[1] == "anchor":
        anchor_chain("cli")
    elif len(sys.argv) >= 3 and sys.argv[1] == "export-pubkey":
        print(export_public_key(sys.argv[2]))
    else:
        print("usage: chain_of_custody.py anchor | export-pubkey <dest.pem>")
 