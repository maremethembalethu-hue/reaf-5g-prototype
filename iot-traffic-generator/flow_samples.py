import json
import math
import random
import time
from pathlib import Path

import dpkt

RANDOM_STATE = 42
SOURCE_ROOT = "./pcaps"

CONTIGUOUS_OUT_DIR_PCAP = "pre-selected/replay_pcaps_contiguous"
CONTIGUOUS_MANIFEST_PATH = "pre-selected/manifest_contiguous.json"

CONTIGUOUS_TARGET_PACKETS = 5000
CONTIGUOUS_NUM_BLOCKS = 5

TARGET_IDLE_TIMEOUT_DEFAULT = 2.0
GAP_PERCENTILE_DEFAULT = 99.0
SAFETY_MARGIN_DEFAULT = 1.2

WINDOW_SIZE = 10  # must match flow_builder.py — live windows are exactly this many packets, no flow grouping


def _iter_ip_packets(source_pcap, max_packets=None):
    with open(source_pcap, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for i, (ts, buf) in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                continue
            yield i, ts, buf, eth, eth.data


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    idx = (p / 100) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def round_up_nice(value, step=0.5):
    if value <= 0:
        return step
    return math.ceil(value / step) * step


def recommend_replay_params(gaps_sorted, target_idle_timeout=TARGET_IDLE_TIMEOUT_DEFAULT,
                             gap_percentile=GAP_PERCENTILE_DEFAULT, safety_margin=SAFETY_MARGIN_DEFAULT):
    if not gaps_sorted:
        return {"suggested_idle_timeout": target_idle_timeout, "suggested_replay_speed": 1.0,
                "suggested_max_delay": target_idle_timeout, "basis_gap_percentile": gap_percentile,
                "basis_gap_percentile_value": 0.0, "max_gap_seconds": 0.0}

    p_gap = percentile(gaps_sorted, gap_percentile)
    suggested_idle_timeout = round_up_nice(p_gap * safety_margin, step=0.5)
    suggested_replay_speed = 1.0 if p_gap <= 0 else round(max(1.0, (p_gap * safety_margin) / target_idle_timeout), 2)
    max_gap = gaps_sorted[-1]
    suggested_max_delay = round_up_nice(
        min(max_gap, p_gap * safety_margin) / suggested_replay_speed if suggested_replay_speed else 0.0, step=0.05)

    return {
        "suggested_idle_timeout": suggested_idle_timeout,
        "suggested_replay_speed": suggested_replay_speed,
        "suggested_max_delay": suggested_max_delay,
        "basis_gap_percentile": gap_percentile,
        "basis_gap_percentile_value": round(p_gap, 4),
        "max_gap_seconds": round(max_gap, 4),
    }


def sample_pcap_contiguous(source_pcap, expected_label, out_dir, target_packets=CONTIGUOUS_TARGET_PACKETS,
                            num_blocks=CONTIGUOUS_NUM_BLOCKS, max_packets=2_000_000, random_state=RANDOM_STATE,
                            target_idle_timeout=TARGET_IDLE_TIMEOUT_DEFAULT, gap_percentile=GAP_PERCENTILE_DEFAULT,
                            safety_margin=SAFETY_MARGIN_DEFAULT, max_block_packets=1000):
    # Several contiguous blocks spread across the file, not one slice that
    # could land on an unrepresentative burst. block_size is forced to a
    # multiple of WINDOW_SIZE so a block boundary never splits a live window
    # across two unrelated parts of the source capture.
    #
    # block_size is additionally CAPPED at max_block_packets, growing the
    # number of blocks instead of their size once target_packets gets large.
    # A larger single block dwelling longer in one part of the file is more
    # likely to land entirely inside a degenerate, repetitive stretch (e.g.
    # a burst of near-duplicate, same-timestamp packets) — confirmed
    # directly: at target_packets=20,000 (block_size=4,000), Benign accuracy
    # collapsed to 65% with dozens of windows showing Rate=0 and nearly
    # identical feature vectors, all traced to one such stretch. Spreading
    # the same total across more, smaller blocks keeps each one's chance of
    # sitting entirely inside a bad stretch roughly constant instead of
    # scaling up with the sample size.
    t0 = time.perf_counter()
    source_pcap = Path(source_pcap)

    total_ip_packets = sum(1 for _ in _iter_ip_packets(source_pcap, max_packets))
    if total_ip_packets == 0:
        print(f"  no IP packets found in {source_pcap}")
        return None

    take = min(target_packets, total_ip_packets)
    n_blocks = max(1, min(num_blocks, take))
    block_size = (take // n_blocks // WINDOW_SIZE) * WINDOW_SIZE
    if block_size > max_block_packets:
        block_size = (max_block_packets // WINDOW_SIZE) * WINDOW_SIZE
        n_blocks = max(1, take // block_size)
    if block_size == 0:
        block_size = WINDOW_SIZE
    segment_len = total_ip_packets // n_blocks

    rng = random.Random(random_state)
    blocks = []
    for b in range(n_blocks):
        seg_start = b * segment_len
        seg_end = total_ip_packets if b == n_blocks - 1 else seg_start + segment_len
        max_start = max(seg_start, seg_end - block_size)
        start = rng.randint(seg_start, max_start) if max_start > seg_start else seg_start
        blocks.append((start, start + block_size))

    out_dir = Path(out_dir) / expected_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_pcap.stem}_contiguous_sample.pcap"

    ip_seen = 0
    block_idx = 0
    written = 0
    baseline_ts = None
    prev_written_ts = None
    gaps = []

    with open(source_pcap, "rb") as f_in, open(out_path, "wb") as f_out:
        reader = dpkt.pcap.Reader(f_in)
        writer = dpkt.pcap.Writer(f_out)

        for i, (ts, buf) in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                continue

            if block_idx < len(blocks):
                b_start, b_end = blocks[block_idx]
                if b_start <= ip_seen < b_end:
                    if baseline_ts is None:
                        baseline_ts = float(ts)
                    written_ts = float(ts) - baseline_ts
                    if prev_written_ts is not None:
                        gaps.append(written_ts - prev_written_ts)
                    prev_written_ts = written_ts
                    writer.writepkt(buf, ts=written_ts)
                    written += 1
                elif ip_seen >= b_end:
                    block_idx += 1

            ip_seen += 1
            if block_idx >= len(blocks):
                break

        writer.close()

    elapsed = time.perf_counter() - t0
    print(f"{source_pcap.name}: took {written:,}/{total_ip_packets:,} IP packets across "
          f"{n_blocks} spread blocks -> {out_path}  ({elapsed:.3f}s)")

    negative_gap_count = sum(1 for g in gaps if g < 0)
    gaps_sorted = sorted(g for g in gaps if g >= 0)
    recommendation = recommend_replay_params(gaps_sorted, target_idle_timeout, gap_percentile, safety_margin)
    recommendation["negative_gap_count"] = negative_gap_count

    print(f"   Gap p{gap_percentile}: {recommendation['basis_gap_percentile_value']:.3f}s  "
          f"replay_speed: {recommendation['suggested_replay_speed']:.2f}x  "
          f"MAX_DELAY: {recommendation['suggested_max_delay']:.2f}s")
    if negative_gap_count:
        print(f"   [WARN] {negative_gap_count} negative gap(s) — source pcap has non-monotonic timestamps")

    return {
        "pcap_path": str(out_path),
        "expected_label": expected_label,
        "num_packets": written,
        "num_packets_available": total_ip_packets,
        "num_blocks": n_blocks,
        "recommendation": recommendation,
    }


def sample_directory_contiguous(source_root, out_dir, target_packets=CONTIGUOUS_TARGET_PACKETS,
                                 target_idle_timeout=TARGET_IDLE_TIMEOUT_DEFAULT,
                                 gap_percentile=GAP_PERCENTILE_DEFAULT, safety_margin=SAFETY_MARGIN_DEFAULT):
    records = []
    label_dirs = [d for d in sorted(Path(source_root).iterdir()) if d.is_dir()]
    all_pcaps = [(d.name, p) for d in label_dirs for p in sorted(d.glob("*.pcap"))]
    print(f"Found {len(label_dirs)} label folders, {len(all_pcaps)} PCAP files")
    for label, pcap_path in all_pcaps:
        rec = sample_pcap_contiguous(pcap_path, expected_label=label, out_dir=out_dir,
                                      target_packets=target_packets, target_idle_timeout=target_idle_timeout,
                                      gap_percentile=gap_percentile, safety_margin=safety_margin)
        if rec:
            records.append(rec)
    return records


def build_manifest_contiguous(sample_records, out_path=CONTIGUOUS_MANIFEST_PATH):
    manifest = [
        {"replay_id": i, "pcap_path": r["pcap_path"], "expected_label": r["expected_label"],
         "num_packets": r["num_packets"], "recommended_replay_params": r.get("recommendation")}
        for i, r in enumerate(sample_records, start=1)
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"jobs": manifest}, f, indent=2)
    print(f"Wrote {len(manifest)} contiguous replay jobs to {out_path}")
    return manifest


def load_manifest(path=CONTIGUOUS_MANIFEST_PATH):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    records = sample_directory_contiguous(SOURCE_ROOT, CONTIGUOUS_OUT_DIR_PCAP,
                                           target_packets=CONTIGUOUS_TARGET_PACKETS)
    build_manifest_contiguous(records, CONTIGUOUS_MANIFEST_PATH)