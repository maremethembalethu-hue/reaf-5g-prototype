# custody_verify.py  -- independent verifier for signed/anchored custody logs.
# Trusts ONLY the public key(s) you pass in (pin the key published off-node),
# never the key material found in the log itself.
import json, hashlib, base64, sys
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

SIGNED_FIELDS = ("entry_hash", "previous_hash", "bundle_hash", "timestamp",
                 "incident_id", "attack_type", "confidence", "model_used",
                 "bundle_dir", "key_id")

def canonical(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def load_pubkeys(pem_paths):
    keys = {}
    for p in pem_paths:
        pub = serialization.load_pem_public_key(open(p, "rb").read())
        raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        keys[hashlib.sha256(raw).hexdigest()] = pub
    return keys

def _jsonl(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []

def _check_sig(pub, sig_hex, data):
    try:
        pub.verify(bytes.fromhex(sig_hex), data); return True
    except (InvalidSignature, ValueError, TypeError):
        return False

def _check_tsa(ts, digest, tsa_cert_path):
    """Returns 'verified' | 'present-unverified' | 'absent'."""
    if not ts or not ts.get("trusted"):
        return "absent"
    if not tsa_cert_path:
        return "present-unverified"
    import rfc3161ng
    cert = open(tsa_cert_path, "rb").read()
    ok = rfc3161ng.check_timestamp(base64.b64decode(ts["token_b64"]),
                                   certificate=cert, digest=digest, hashname="sha256")
    return "verified" if ok else "INVALID"

def verify(coc_log, anchor_log, pubkeys, tsa_cert=None):
    problems, entries, anchors = [], _jsonl(coc_log), _jsonl(anchor_log)
    prev, ts_state = "0" * 64, {"verified": 0, "present-unverified": 0, "absent": 0, "INVALID": 0}
    for i, e in enumerate(entries):
        # 1. chain linkage + entry_hash
        exp = hashlib.sha256(f"{e['previous_hash']}{e['bundle_hash']}{e['timestamp']}{e['incident_id']}".encode()).hexdigest()
        if e["previous_hash"] != prev: problems.append((i, "BROKEN_LINK"))
        if exp != e["entry_hash"]:     problems.append((i, "ENTRY_HASH_MISMATCH"))
        prev = e["entry_hash"]
        # 2. signature by a pinned key
        pub = pubkeys.get(e.get("key_id"))
        if not e.get("signature"):            problems.append((i, "UNSIGNED"))
        elif pub is None:                     problems.append((i, "UNKNOWN_KEY_ID"))
        elif not _check_sig(pub, e["signature"], canonical({k: e.get(k) for k in SIGNED_FIELDS})):
            problems.append((i, "BAD_SIGNATURE"))
        # 3. external time
        ts_state[_check_tsa(e.get("trusted_timestamp"), bytes.fromhex(e["entry_hash"]), tsa_cert)] += 1
    # 4. anchors
    for j, a in enumerate(anchors):
        core = {k: a[k] for k in ("anchored_at", "entry_count", "head_hash", "key_id", "reason")}
        pub = pubkeys.get(a.get("key_id"))
        if pub is None or not _check_sig(pub, a.get("signature", ""), canonical({**core, "signature_alg": a.get("signature_alg")})):
            problems.append((f"anchor{j}", "BAD_ANCHOR_SIGNATURE")); continue
        n = a["entry_count"]
        if len(entries) < n:                       problems.append((f"anchor{j}", "TAIL_TRUNCATED"))
        elif entries[n - 1]["entry_hash"] != a["head_hash"]: problems.append((f"anchor{j}", "ANCHORED_ENTRY_REWRITTEN"))
        ts_state[_check_tsa(a.get("trusted_timestamp"), hashlib.sha256(canonical(core)).digest(), tsa_cert)] += 1
    return {"entries": len(entries), "anchors": len(anchors), "problems": problems,
            "timestamps": ts_state, "ok": not problems}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("coc_log"); ap.add_argument("--anchors", default="")
    ap.add_argument("--pubkey", action="append", required=True)
    ap.add_argument("--tsa-cert", default=None)
    a = ap.parse_args()
    r = verify(a.coc_log, a.anchors, load_pubkeys(a.pubkey), a.tsa_cert)
    print(json.dumps(r, indent=2, default=str)); sys.exit(0 if r["ok"] else 1)