#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


echo " REAF-5G Prototype — Startup"

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
echo "      Waiting 25 seconds for UE registration..."
sleep 20

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
sleep 10

echo " Samples a fixed number of *complete* flows from a source PCAP... "

python3 iot-traffic-generator/flow_sampler.py

# Step 5: Start edge node inside UPF network namespace
echo "5/6 Building and starting Edge Node and Traffic Generator..."
cd "$SCRIPT_DIR"
export TARGET_IP="$UPF_GW"
docker compose up -d --build 5g-edge-node iot-traffic-generator

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
echo ""
echo " docker logs -f 5g-edge-node"
echo " docker logs -f reaf-traffic"
