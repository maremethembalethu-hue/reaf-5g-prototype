#!/bin/bash

cd docker_open5gs

docker compose -f sa-deploy.yaml up -d

sleep 30

docker compose -f nr-gnb.yaml up -d

sleep 15 

docker compose -f nr-ue.yaml up -d

echo "Waiting for UE..."
sleep 20

cd ..

docker compose up --build -d 5g-edge-node


docker build -t reaf-5g-prototype-iot-traffic-generator .

docker run --build -d \
    --name traffic-generator \
    --network container:nr_ue \
    reaf-5g-prototype-iot-traffic-generator