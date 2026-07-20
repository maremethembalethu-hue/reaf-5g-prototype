#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stopping REAF-5G Prototype..."

echo "1/4 Stopping Edge Node + Traffic Generator..."
cd "$SCRIPT_DIR"
docker compose down || true

echo "2/4 Stopping the UE..."
cd "$SCRIPT_DIR/docker_open5gs"
docker compose -f nr-ue.yaml down || true
sleep 5

echo "3/4 Stopping the gNB..."
docker compose -f nr-gnb.yaml down || true
sleep 5

echo "4/4 Stopping the 5G Core..."
docker compose -f sa-deploy.yaml down || true

echo "Completed: All stopped."