# REAF-5G: Real-time Edge Adaptive Forensic IDS for 5G IoT

A two-tier (Heavy/Lite) machine-learning intrusion detection system that classifies live traffic at a simulated 5G edge node (UPF), trained on CICIoT2023, with forensic evidence preservation and a live results dashboard.

---

## What this project actually does

1. A simulated 5G core (Open5GS + UERANSIM) tunnels traffic from a simulated UE through a UPF network namespace.
2. A traffic generator replays labeled attack/benign pcap samples through that tunnel.
3. An edge-node agent captures the decapsulated packets in real time, extracts CICIoT2023-style features, and classifies each window using a 3-stage model pipeline, attack/benign gate to 6-way family classifier to DDoS-vs-DoS split, in either a Heavy (XGBoost) or Lite (Decision Tree) configuration depending on available resources.
4. Every detection triggers forensic evidence collection (packet capture, memory snapshot, process list) chained together with a tamper-evident hash log.
5. A dashboard (optional) shows live detections and computed accuracy; a standalone script can also do this without the dashboard.

---

## Architecture

```
reaf-5g-prototype/
 - docker-compose.yml
 - start.sh                     # brings up the whole stack in order, brings up core to gNB to UE to edge node to traffic generator
 - end.sh                        # tears everything down in reverse order
 - 5g-edge-node/                 # MAIN CONTRIBUTION, capture, features, classification, forensics
   - Dockerfile
   - capture.py                # packet capture, dual-window building, routing to classify_flow()
   - flow_builder.py           # fixed-size (10 / 100 packet) window accumulation
   - feature_extraction.py     # aggregates a completed window into the CICIoT2023 feature vector
   - model_engine.py           # 3-stage ONNX inference + dual-window reconciliation
   - chain_of_custody.py       # SHA-256 hash-chained evidence log
   - memory_snapshot.py        # RAM/process capture at detection time
   - fragment_reassembly.py    # reassembles IP fragments PRESENT IN THE ORIGINAL traffic
   - resource_monitor.py       # CPU/RAM polling, decides Heavy vs Lite tier
   - evidence_collect.py       # saves pcap/process/memory/syslog evidence bundle
   - trigger.py                # confidence/rate thresholds for evidence acquisition
   - report_generator.py       # ISO/IEC 27043-structured PDF report per incident
   - detection_models/         # trained ONNX artifacts, copied in from model-training/
 - iot-traffic-generator/       # replays labeled CICIoT2023 pcaps through the tunnel
   - Dockerfile
│  - requirements.txt
│  - traffic_generator.py      # orchestrator: synthetic / replay / mixed / random / manifest modes
│  - replay_engine.py          # reads a pcap, encapsulates + sends each packet
│  - build_envelope.py               # wire format for encapsulated packets (see "Encapsulation" below)
│  - pcaps/, pre-selected/     # sampled replay pre-selected/ + manifest_contigenous.json (see evaluation/flow_sampler.py)
 - 5g-core-sim/                  # minimal destination server traffic is forwarded to
 - open5gs/, ueransim/           # simulated 5G core + radio/UE (outsourced, see below)
 - evidence/                     # AUTO-GENERATED at runtime — packets/, memory/, processes/, chain_of_custody.log
 - model-training/               # offline: trains all 6 ONNX models from CICIoT2023 CSVs
   - training_models.py
 - evaluation/                   # AUTO-GENERATED at runtime + manual scoring
   - items_log.jsonl           # live, one line per classified window (the real accuracy source)
   - truth_log.jsonl           # live, ground truth per replay job
   - debug_predictions.jsonl   # live, optional, raw stage probabilities (DEBUG_FEATURES=1)
   - true_results.py           # manual join step to joined_results.csv (optional if using the dashboard)
   - results_engine.py         # the join + accuracy logic, shared by true_results.py and the dashboard
   - forensic_validation.py     # independently re-verifies the chain-of-custody hash chain
 - dashboard/                     # OPTIONAL, live web view of the same data
    - Dockerfile
    - app.py
    - results_engine.py          # live join, computed on every request — no manual step needed
    - monitor_tier.py           # CPU/RAM to the live dashboard, show live CPU/RAM
    - templates/index.html
```

`open5gs/` is cloned directly from [docker_open5gs](https://github.com/herlesupreeth/docker_open5gs); `ueransim/` is built from the same repo's UERANSIM images. Both are configured, not authored, by this project.

---

## The dataset and models, briefly

- Trained on **CICIoT2023** (39-feature official schema see `docs/` for why the 46-feature alternative was tested and rejected: its extraction code doesn't reliably reproduce a live pipeline's requirements, even though it scores higher offline).
- Official source: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **3-stage architecture**: Stage 1 (Benign vs Attack) to Stage 2 (6-way family: BruteForce, Flood, Mirai, Recon, Spoofing, Web) to Stage 3 (DDoS vs DoS, only on rows Stage 2 calls Flood).
- **Dual-window live classification**: the official dataset windows DDoS/DoS/Mirai traffic at 100 packets and everything else at 10, the live pipeline runs both window sizes in parallel and reconciles them (see `model_engine.py`). This was the single biggest fix to live-vs-training accuracy divergence during development.
- **Known limitation**: DDoS-vs-DoS specifically sits at a ~65-78% ceiling, traced to two features (`Srate`/`Drate`) missing from the CICIoT2023 schema and structurally unrecoverable from what's retained. This is documented, not an open bug.

---

## Setup

1. Clone the 5G core into this project:
   ```bash
   git clone https://github.com/herlesupreeth/docker_open5gs
   ```
   Edit its config files to set your PLMN values before first run.

2. Build the two services:
   ```bash
   docker compose build
   ```

3. Sample replay traffic from your raw CICIoT2023 pcaps (offline, runs on
   your host no containers needed):
   ```bash
   python3 iot-traffic-generator/flow_sampler.py
   ```
## Requirements

- Docker + Docker Compose
- Linux host with `NET_ADMIN`/`SYS_ADMIN` capability available (the edge node needs elevated permissions to capture in the UPF namespace)
- `sudo` access (for `sysctl -w net.ipv4.ip_forward=1`)
- Trained model artifacts already present in `5g-edge-node/detection_models/` (see **Training the models**, below, if you don't have these yet)

---

## Training the models (do this once, before first run)

```bash
cd model-training/
python3 training_models.py
```

This trains all 6 ONNX artifacts (Heavy + Lite, each for Stage 1/2/3) directly from the CICIoT2023 CSVs under `model-training/data/ciciot2023/`. Copy the output into the edge node before building it:

```bash
cp outputs/*/*.onnx outputs/*/*.pkl outputs/*/feature_lists.json ../5g-edge-node/detection_models/
```

---

## Running a live test

```bash
sudo ./start.sh
```

This brings up, in order: IP forwarding to Open5GS core to UERANSIM gNB to UERANSIM UE (with routing fixed so all UE traffic goes through the tunnel) to the edge node, traffic generator, and dashboard together to waits for the dashboard to actually respond to opens it in your browser automatically at `http://localhost:8080`.


Watch it working directly via on another terminal but same directory of the prototype:
```bash
docker logs -f 5g-edge-node
docker logs -f reaf-traffic
docker logs -f dashboard
```
Watch `docker logs -f reaf-traffic` for `"All N replay jobs complete"`,
then `docker compose stop iot-traffic-generator` — it has `restart:
unless-stopped`, so left alone it will start the whole manifest over again.

```bash
sudo ./end.sh
```
Tears down the traffic generator, edge node, UE, gNB, and core, in that
order.
---

### Automatic post-run evaluation
 
`start.sh` doesn't exit once the dashboard opens it stays running and calls `docker wait reaf-traffic`, which blocks until the traffic generator's replay job actually finishes, this can take anywhere from seconds to a long time depending on manifest size; the dashboard is already live and usable during that wait. Once the replay exits, it automatically runs, in order:
 
```bash
python3 evaluation/true_results.py               # joins items_log.jsonl + truth_log.jsonl -> joined_results.csv
python3 evaluation/forensic_validation.py       # verifies the evidence hash chain
```
 
So a full `./start.sh` run ends with both a fresh `joined_results.csv` and a chain-of-custody verification already done, with no separate manual step needed see **Getting results without the dashboard** below for running either of these on demand instead.
 
---

### Running WITHOUT the dashboard

The dashboard is a presentation layer only, every piece of actual research evidence (`evidence/`, `evaluation/items_log.jsonl`, `evaluation/truth_log.jsonl`, the chain-of-custody log) is written by the edge node and traffic generator regardless of whether the dashboard container exists at all.


#### Getting results without the dashboard

Two options, both read the same live files the dashboard would:

**Option A: one-off snapshot**, matching the original workflow this project started with:
```bash
cd evaluation/
python3 true_results.py
```
This joins `items_log.jsonl` against `truth_log.jsonl`, writes `joined_results.csv` and `flow_recall_summary.csv`, and prints a four-level accuracy breakdown (strict, attack-vs-benign, 6-family, DDoS-vs-DoS) straight to the terminal. Re-run it any time, during a replay or after to get an updated snapshot.

Both approaches are correct and give the same numbers, the dashboard just automates on a 10-second refresh instead of you running it by hand.

### A note on file naming

The pipeline writes **two different files** that can look similar at a glance:
- `evaluation/items_log.jsonl`: the real per-window classification log (`flow_id`, `predicted_label`, `confidence`, `model_used`, etc.). **This is the one everything accuracy-related reads.**
- `evaluation/predictions_log.jsonl`: a separate raw per-window model-output log (`flow_start_time`, `attack_type`, `proto`, no `flow_id`). It cannot be joined to `truth_log.jsonl` at all (no `flow_id` to match on), so it isn't used for scoring, treat it as a supplementary raw-output log if you need it for something else.

---

## Forensic evidence and reports

Every triggered detection writes to `evidence/`:
- `evidence/packets/*.pcap`: the captured traffic for that incident
- `evidence/memory/*.json`, `evidence/processes/*.json`: system state at detection time
- `evidence/chain_of_custody.log`: append-only, SHA-256 hash-chained log linking every entry to the previous one

---

## Evaluating chain-of-custody integrity

```bash
cd evaluation/
python3 forensic_validation.py
```
Verifies every hash in `chain_of_custody.log` correctly links to the one before it, and reports any break.

---
