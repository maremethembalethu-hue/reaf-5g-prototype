#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/.venv/bin/python"
# USE_RFC3161=true   time-stamp EVERY custody entry with the TSA (adds a network round trip per incident)
# ANCHOR_EVERY=N     also anchor the chain head every N entries (an anchor is always written at the end of a run)
# TSA_URL            RFC 3161 time-stamp authority
KEYS_DIR="$SCRIPT_DIR/keys"          # persists across runs, NEVER inside evidence/ (evidence/ is wiped every run)
ANCHORS_DIR="$SCRIPT_DIR/anchors"    # chain anchors; for real protection place this where the edge node cannot rewrite it
export USE_RFC3161="${USE_RFC3161:-false}"
export ANCHOR_EVERY="${ANCHOR_EVERY:-0}"
export TSA_URL="${TSA_URL:-https://freetsa.org/tsr}"
export CUSTODY_KEYS_DIR="$KEYS_DIR"
export CUSTODY_ANCHORS_DIR="$ANCHORS_DIR"
CUSTODY_OVERRIDE="$SCRIPT_DIR/.custody.override.yml"

# Menu: Normal Run / Run Experiment (5K, 10K, 25K) / Exit
MODE=""; SIZE_LABEL=""; PACKETS=""; RUNTIME=""
while [ -z "$MODE" ]; do
    echo ""
    echo " REAF-5G Prototype"
    echo "  1) Normal Run"
    echo "  2) Run Experiment"
    echo "  3) Exit"
    read -r -p "Select [1-3]: " choice || exit 1
    case "$choice" in
        1) MODE="normal" ;;
        2) while [ -z "$SIZE_LABEL" ]; do
               echo "  1) 5K   2) 10K   3) 25K   4) Back"
               read -r -p "Select [1-4]: " size_choice || exit 1
               case "$size_choice" in
                   1) SIZE_LABEL="5K";  PACKETS=1000;  RUNTIME=1800 ;;
                   2) SIZE_LABEL="10K"; PACKETS=10000; RUNTIME=3000 ;;
                   3) SIZE_LABEL="25K"; PACKETS=25000; RUNTIME=6000 ;;
                   4) break ;;
                   *) echo "Invalid selection '$size_choice'. Enter 1, 2, 3 or 4." ;;
               esac
           done
           [ -n "$SIZE_LABEL" ] && MODE="experiment" ;;
        3) echo "Exit."; exit 0 ;;
        *) echo "Invalid selection '$choice'. Enter 1, 2 or 3." ;;
    esac
done


echo " REAF-5G Prototype: Startup"

# Step 1: IP Forwarding
echo "1/6 Enabling IP forwarding..."
sudo sysctl -w net.ipv4.ip_forward=1

# Step 2: Start Open5GS Core
echo "2/6 Starting Open5GS 5G Core..."
cd "$SCRIPT_DIR/docker_open5gs"
set -a; source .env; set +a
docker compose -f sa-deploy.yaml up -d
echo "      Waiting 25 seconds for core to be ready..."
sleep 25

# Step 3: Start gNB
echo "3/6 Starting UERANSIM gNB..."
docker compose -f nr-gnb.yaml up -d
sleep 10

# Step 4: Start UE
echo "4/6 Starting UERANSIM UE..."
docker compose -f nr-ue.yaml up -d
echo "      Waiting 15 seconds for UE registration..."
sleep 15

#  replace the default route with uesimtun0 so ALL traffic goes through the 5G tunnel to the UPF
echo "      nr_ue routing: replacing eth0 default with uesimtun0..."
docker exec nr_ue ip route del default 2>/dev/null || true
docker exec nr_ue ip route add default dev uesimtun0
echo "      nr_ue default route: $(docker exec nr_ue ip route | grep default)"

# Get UE IP from tunnel interface
UE_IP=$(docker exec nr_ue ip addr show uesimtun0 2>/dev/null \
    | grep "inet " \
    | awk '{print $2}' \
    | cut -d'/' -f1)

# Derive UPF gateway from UE IP
UPF_GW=$(echo "$UE_IP" | sed 's/\.[0-9]*$/.1/')

echo "      UE IP   : $UE_IP"
echo "      UPF GW  : $UPF_GW"



echo "Removing old REAF containers..."

docker rm -f reaf-traffic 2>/dev/null || true
docker rm -f 5g-edge-node 2>/dev/null || true
docker rm -f dashboard 2>/dev/null || true

echo "Clearing Evidence from previous runs..."
sudo rm -rf "$SCRIPT_DIR/evidence"
echo "Clearing anchors from previous runs (the signing key in $KEYS_DIR is kept)..."
sudo rm -rf "$ANCHORS_DIR"
echo "Clearing evaluation logs from previous runs..."
sudo rm -rf \
        "$SCRIPT_DIR/evaluation/items_log.jsonl" \
        "$SCRIPT_DIR/evaluation/truth_log.jsonl" \
        "$SCRIPT_DIR/evaluation/debug_predictions.jsonl" \
        "$SCRIPT_DIR/evaluation/predictions_log.jsonl" \
        "$SCRIPT_DIR/evaluation/experiment_suite_report.json" \
        "$SCRIPT_DIR/evaluation/resource_report.json" \
        "$SCRIPT_DIR/evaluation/timing_log.jsonl" \
        "$SCRIPT_DIR/evaluation/tamper_test_report.json" \
        "$SCRIPT_DIR/evaluation/flow_recall_summary.csv" \
        "$SCRIPT_DIR/evaluation/joined_results.csv" \
        "$SCRIPT_DIR/evaluation/capture_counters.json" \
        "$SCRIPT_DIR/evaluation/tcpdump_stats.txt" \
        "$SCRIPT_DIR/evaluation/validation" \
        "$SCRIPT_DIR/edge_logs.txt" \
        "$SCRIPT_DIR/traffic_logs.txt"
sleep 10

# Custody signing key: created once on the host (so the same key_id is used in every run), public key exported
echo "Preparing custody signing key and anchor folder..."
mkdir -p "$KEYS_DIR" "$ANCHORS_DIR"
KEY_ID=$(KEY_DIR="$KEYS_DIR" "$PYTHON" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR/5g-edge-node')
import chain_of_custody as c
print(c.export_public_key('$KEYS_DIR/custody_public.pem'))") \
    || { echo "ERROR: cannot create the custody signing key (pip install cryptography on the host)"; exit 1; }
echo "      custody key_id : $KEY_ID"
echo "      public key     : $KEYS_DIR/custody_public.pem  (publish a copy OUTSIDE this machine)"

# Experiment only: create the run folder, set CONTIGUOUS_TARGET_PACKETS (the only line changed in flow_samples.py) and sample the PCAPs
if [ "$MODE" = "experiment" ]; then
    RUN_DIR=$("$PYTHON" "$SCRIPT_DIR/evaluation/sum_results/scripts/run_layout.py" create --size-label "$SIZE_LABEL" --packets-per-job "$PACKETS") \
        || { echo "ERROR: cannot create the run directory"; exit 1; }
    echo "Run directory: $RUN_DIR"
    FLOW_PY=$(find "$SCRIPT_DIR" -maxdepth 3 -name flow_samples.py -path "*iot-traffic-generator*" | head -1)
    [ -n "$FLOW_PY" ] || { echo "ERROR: flow_samples.py not found"; exit 1; }
    sed -i -E "s/^(CONTIGUOUS_TARGET_PACKETS = )[0-9_]+/\1$PACKETS/" "$FLOW_PY"
    FLOW_ROOT="$(dirname "$(dirname "$FLOW_PY")")"
    (cd "$FLOW_ROOT" && "$PYTHON" iot-traffic-generator/flow_samples.py) || { echo "ERROR: flow_samples.py failed"; exit 1; }
    EXPECTED_JOBS=$("$PYTHON" -c "import json;print(len(json.load(open('$FLOW_ROOT/pre-selected/manifest_contiguous.json'))['jobs']))") \
        || { echo "ERROR: pre-selected/manifest_contiguous.json missing"; exit 1; }
    "$PYTHON" "$SCRIPT_DIR/evaluation/sum_results/scripts/run_layout.py" configure --run-dir "$RUN_DIR" --size-label "$SIZE_LABEL" \
        --packets-per-job "$PACKETS" --expected-jobs "$EXPECTED_JOBS"
    cp "$FLOW_ROOT/pre-selected/manifest_contiguous.json" "$RUN_DIR/config/"
    cp "$FLOW_ROOT/pre-selected/sampling_provenance.json" "$RUN_DIR/config/" 2>/dev/null || echo "WARNING: sampling_provenance.json not produced (is the new flow_samples.py in place?)"
fi

# Step 5: Start edge node inside UPF network namespace
echo "5/6 Building and starting Edge Node and Traffic Generator..."
cd "$SCRIPT_DIR"
export TARGET_IP="$UPF_GW"
[ -n "$RUNTIME" ] && export MAX_RUNTIME_SECONDS="$RUNTIME"   # replay time budget; the generator default of 1200 s cut the 25K run short

# Compose override: mounts the key and anchor folders into the edge node and passes the custody settings.
# Written here so docker-compose.yml itself does not have to change.
cat > "$CUSTODY_OVERRIDE" <<'YAML'
services:
  5g-edge-node:
    environment:
      - KEY_DIR=/keys
      - ANCHOR_LOG=/anchors/chain_anchors.jsonl
      - USE_RFC3161=${USE_RFC3161:-false}
      - ANCHOR_EVERY=${ANCHOR_EVERY:-0}
      - TSA_URL=${TSA_URL:-https://freetsa.org/tsr}
    volumes:
      - ${CUSTODY_KEYS_DIR}:/keys
      - ${CUSTODY_ANCHORS_DIR}:/anchors
YAML
BASE_COMPOSE=""
for f in docker-compose.yml docker-compose.yaml compose.yml compose.yaml; do
    [ -f "$SCRIPT_DIR/$f" ] && { BASE_COMPOSE="$f"; break; }
done
[ -n "$BASE_COMPOSE" ] || { echo "ERROR: no compose file found next to start.sh"; exit 1; }
DC="docker compose -f $BASE_COMPOSE -f $CUSTODY_OVERRIDE"

# Build first, then make sure the image can actually sign. Otherwise every entry would silently be UNSIGNED.
$DC build 5g-edge-node || { echo "ERROR: edge node build failed"; exit 1; }
NEEDED="import cryptography"
[ "$USE_RFC3161" = "true" ] && NEEDED="import cryptography, rfc3161ng"
# Test the built image directly (no compose network or dependencies involved) and show the real error if it fails
EDGE_IMG=$($DC config --images 2>/dev/null | grep -E '5g-edge-node' | head -1)
[ -n "$EDGE_IMG" ] || EDGE_IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '5g-edge-node' | head -1)
[ -n "$EDGE_IMG" ] || { echo "ERROR: cannot determine the edge-node image name after build"; exit 1; }
echo "      checking custody-signing packages in image $EDGE_IMG ..."
if ! CHECK_OUT=$(docker run --rm --entrypoint python3 "$EDGE_IMG" -c "$NEEDED; print('ok')" 2>&1); then
    echo "ERROR: the edge-node image cannot import the packages needed for custody signing ($NEEDED)."
    echo "       Python said:"; echo "$CHECK_OUT" | sed 's/^/         /'
    echo "       If cryptography/rfc3161ng are missing, check 5g-edge-node/requirements.txt and rebuild with: docker compose build --no-cache 5g-edge-node"
    exit 1
fi
$DC up -d 5g-edge-node iot-traffic-generator dashboard

# reaf-traffic shares nr_ue network namespace but Docker resets the route on container start so  fix it here as well
echo "      Waiting 5 seconds for traffic generator to initialise..."
sleep 5
echo "       replacing eth0 default with uesimtun0..."
docker exec reaf-traffic ip route del default 2>/dev/null || true
docker exec reaf-traffic ip route add default dev uesimtun0 2>/dev/null || true
echo "      reaf-traffic default route: $(docker exec reaf-traffic ip route | grep default)"


echo " Done"
echo " UE IP       : $UE_IP"
echo " UPF Gateway : $UPF_GW"
echo " Custody key : $KEY_ID  (RFC3161 per entry: $USE_RFC3161, anchor every: $ANCHOR_EVERY)"
echo ""
echo " docker logs -f 5g-edge-node"
echo " docker logs -f reaf-traffic"


echo "Writing the log to a .txt files"
docker logs -f 5g-edge-node > edge_logs.txt 2>&1 &
docker logs -f reaf-traffic > traffic_logs.txt 2>&1 &
# Experiment only: wait for the replay, archive, remove containers, then evaluate
if [ "$MODE" = "normal" ]; then
    echo "Normal run: stack is running, no experiment evaluation. Stop it with ./end.sh"
    echo "To anchor the custody chain now, run:  ./anchor_now.sh"
    exit 0
fi

echo ""
echo "Waiting for the traffic-generator replay (reaf-traffic) to finish..."
docker wait reaf-traffic >/dev/null 2>&1
echo "Replay finished; waiting for evidence collection to settle..."
previous=""; stable=0; waited=0
while [ "$waited" -lt "${MAX_SETTLE_SECONDS:-900}" ]; do
    current="$(find "$SCRIPT_DIR/evidence" -maxdepth 1 -name 'incident_*' 2>/dev/null | wc -l)/$(wc -l < "$SCRIPT_DIR/evaluation/timing_log.jsonl" 2>/dev/null || echo 0)/$(wc -l < "$SCRIPT_DIR/evaluation/items_log.jsonl" 2>/dev/null || echo 0)"
    if [ "$current" = "$previous" ]; then stable=$((stable + 5)); else stable=0; fi
    [ "$stable" -ge "${SETTLE_SECONDS:-30}" ] && break
    previous="$current"; sleep 5; waited=$((waited + 5))
done

# Anchor the finished chain (signed head + entry count, time-stamped if USE_RFC3161=true)
echo "Anchoring the custody chain..."
docker exec 5g-edge-node sh -c 'cd "$(dirname "$(find / -name chain_of_custody.py -not -path "*/site-packages/*" 2>/dev/null | head -1)")" && python3 chain_of_custody.py anchor' \
    || echo "WARNING: chain anchoring failed; truncation and rewrite of the whole chain will NOT be detectable for this run"

# Put the pinned public key and the anchors next to the evidence so they are archived with the run
sudo cp "$KEYS_DIR/custody_public.pem" "$SCRIPT_DIR/evidence/custody_public.pem" 2>/dev/null || true
sudo cp "$ANCHORS_DIR/chain_anchors.jsonl" "$SCRIPT_DIR/evidence/chain_anchors.jsonl" 2>/dev/null || true

echo "Archiving raw data into $RUN_DIR ..."
cp edge_logs.txt traffic_logs.txt "$RUN_DIR/logs/" 2>/dev/null || true
sudo "$PYTHON" "$SCRIPT_DIR/evaluation/sum_results/scripts/run_layout.py" archive --run-dir "$RUN_DIR" \
    --live-eval-dir "$SCRIPT_DIR/evaluation" --live-evidence-dir "$SCRIPT_DIR/evidence"
sudo chown -R "$(id -u):$(id -g)" "$RUN_DIR"

echo "Removing Containers After The Experiment.."
./end.sh

# Independent custody verification (pinned public key + anchors) and the threat-model probes, both on the archived copy
echo "Verifying custody signatures and anchors..."
"$PYTHON" "$SCRIPT_DIR/evaluation/custody_verify.py" "$RUN_DIR/raw/evidence/chain_of_custody.log" \
    --anchors "$RUN_DIR/raw/evidence/chain_anchors.jsonl" --pubkey "$KEYS_DIR/custody_public.pem" \
    > "$RUN_DIR/logs/custody_verify.json" 2>&1 \
    && echo "      custody verification: OK" || echo "      custody verification: PROBLEMS (see $RUN_DIR/logs/custody_verify.json)"
echo "Running threat-model probes (the private key in $KEYS_DIR is used only to simulate a key-compromise attacker)..."
"$PYTHON" "$SCRIPT_DIR/evaluation/sum_results/scripts/threat_model_probes.py" --run-dir "$RUN_DIR" \
    --pubkey "$KEYS_DIR/custody_public.pem" --signing-key "$KEYS_DIR/custody_signing_key.pem" \
    --out "$RUN_DIR/logs/threat_model_probes.json" >/dev/null 2>&1 \
    && echo "      probes behaved as designed" || echo "      probes: unexpected outcome (see $RUN_DIR/logs/threat_model_probes.json)"

# Computes every Chapter 4 table from the archived data
"$PYTHON" "$SCRIPT_DIR/evaluation/sum_results/scripts/evaluate_run.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/logs/evaluate.log"
echo "Results: $RUN_DIR"