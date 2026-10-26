#!/bin/bash
# Usage: ./experiment.sh 5K|10K|25K [--dry-run]
#        ./experiment.sh --evaluate-only evaluation/experiments/<run>
# The size is the packet target PER REPLAY JOB (per PCAP), as flow_samples.py applies it;
# the total traffic is that target times the number of jobs in the manifest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
EVAL_DIR="$SCRIPT_DIR/evaluation"
FLOW_SAMPLES_DIR="${FLOW_SAMPLES_DIR:-$SCRIPT_DIR}"
MANIFEST_RELATIVE="pre-selected/manifest_contiguous.json"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"
MAX_SETTLE_SECONDS="${MAX_SETTLE_SECONDS:-900}"
REPLAY_TIMEOUT_SECONDS="${REPLAY_TIMEOUT_SECONDS:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { echo "Usage: $0 5K|10K|25K [--dry-run] | $0 --evaluate-only <run_dir>" >&2; exit 2; }

evaluate_run() {
    python3 "$EVAL_DIR/evaluate_run.py" --run-dir "$1"
}

count_evidence_state() {
    local incidents timing items
    incidents=$(find "$SCRIPT_DIR/evidence" -maxdepth 1 -name 'incident_*' 2>/dev/null | wc -l)
    timing=$(wc -l < "$EVAL_DIR/timing_log.jsonl" 2>/dev/null || echo 0)
    items=$(wc -l < "$EVAL_DIR/items_log.jsonl" 2>/dev/null || echo 0)
    echo "$incidents/$timing/$items"
}

wait_for_replay() {
    echo "Waiting for the traffic-generator replay (reaf-traffic) to finish..."
    if [ "$REPLAY_TIMEOUT_SECONDS" -gt 0 ]; then
        timeout "$REPLAY_TIMEOUT_SECONDS" docker wait reaf-traffic >/dev/null 2>&1 \
            || die "replay did not finish within ${REPLAY_TIMEOUT_SECONDS}s"
    else
        docker wait reaf-traffic >/dev/null 2>&1 || die "docker wait failed for reaf-traffic"
    fi
    echo "Replay finished; waiting for evidence collection to settle..."
    local previous="" current waited=0 stable=0
    while [ "$waited" -lt "$MAX_SETTLE_SECONDS" ]; do
        current=$(count_evidence_state)
        if [ "$current" = "$previous" ]; then
            stable=$((stable + 5))
            [ "$stable" -ge "$SETTLE_SECONDS" ] && { echo "Evidence stable at incidents/timing/windows = $current"; return 0; }
        else
            stable=0
        fi
        previous="$current"
        sleep 5
        waited=$((waited + 5))
    done
    echo "WARNING: evidence still changing after ${MAX_SETTLE_SECONDS}s; archiving what exists." >&2
}

[ $# -ge 1 ] || usage
if [ "$1" = "--evaluate-only" ]; then
    [ -n "${2:-}" ] && [ -d "$2/raw" ] || die "--evaluate-only needs an existing run directory containing raw/"
    evaluate_run "$2"; exit $?
fi

case "$1" in
    5K)  PACKETS_PER_JOB=5000 ;;
    10K) PACKETS_PER_JOB=10000 ;;
    25K) PACKETS_PER_JOB=25000 ;;
    *)   usage ;;
esac
SIZE_LABEL="$1"
DRY_RUN=false
[ "${2:-}" = "--dry-run" ] && DRY_RUN=true

if $DRY_RUN; then
    echo "Dry run: would sample $PACKETS_PER_JOB packets per PCAP in $FLOW_SAMPLES_DIR, rebuild the traffic generator,"
    echo "replay all jobs, archive raw data under $EVAL_DIR/experiments/<timestamp>_${SIZE_LABEL}_${PACKETS_PER_JOB}pkt_per_job,"
    echo "then run evaluate_run.py."
    exit 0
fi

source "$SCRIPT_DIR/scripts/stack.sh"
RUN_DIR=$(python3 "$EVAL_DIR/run_layout.py" create --size-label "$SIZE_LABEL" --packets-per-job "$PACKETS_PER_JOB") \
    || die "could not create the run directory"
echo "Run directory: $RUN_DIR"
exec > >(tee -a "$RUN_DIR/logs/experiment.log") 2>&1

echo "Sampling $PACKETS_PER_JOB packets per PCAP (flow_samples.py)..."
(cd "$FLOW_SAMPLES_DIR" && TARGET_PACKETS_PER_PCAP="$PACKETS_PER_JOB" python3 "$SCRIPT_DIR/iot-traffic-generator/flow_samples.py") \
    || die "flow_samples.py failed"
MANIFEST_PATH="$FLOW_SAMPLES_DIR/$MANIFEST_RELATIVE"
python3 "$EVAL_DIR/validate_run.py" --manifest "$MANIFEST_PATH" --packets-per-job "$PACKETS_PER_JOB" \
    > "$RUN_DIR/config/manifest_check.json" || die "manifest is empty or missing: $MANIFEST_PATH"
cp "$MANIFEST_PATH" "$RUN_DIR/config/manifest_contiguous.json"
EXPECTED_JOBS=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['jobs']))" "$MANIFEST_PATH")
python3 "$EVAL_DIR/run_layout.py" configure --run-dir "$RUN_DIR" --size-label "$SIZE_LABEL" \
    --packets-per-job "$PACKETS_PER_JOB" --expected-jobs "$EXPECTED_JOBS" || die "could not write run_config.json"

enable_ip_forwarding
start_core_stack || die "core stack failed to start"
reset_live_data
start_reaf_containers || die "REAF containers failed to start"
follow_container_logs "$RUN_DIR/logs/edge_node.log" "$RUN_DIR/logs/traffic_generator.log"

wait_for_replay

echo "Archiving raw data..."
sudo python3 "$EVAL_DIR/run_layout.py" archive --run-dir "$RUN_DIR" --live-eval-dir "$EVAL_DIR" \
    --live-evidence-dir "$SCRIPT_DIR/evidence" || die "archiving failed"
sudo chown -R "$(id -u):$(id -g)" "$RUN_DIR"

echo "Removing containers after the experiment..."
./end.sh

evaluate_run "$RUN_DIR" | tee "$RUN_DIR/logs/evaluate.log"
status=${PIPESTATUS[0]}
echo "Run directory: $RUN_DIR"
exit "$status"
