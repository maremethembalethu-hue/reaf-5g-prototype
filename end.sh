
echo "First stop the project:"

cd ~/reaf-5g-prototype

docker compose down 

echo "Change directories ..."
sleep 15
cd docker_open5gs

echo "Stopping the UE ..."
docker compose -f nr-ue.yaml down

sleep 5

echo "Stopping the gNB..."
docker compose -f nr-gnb.yaml down

sleep 10

echo "Finally stopping the 5G Core..."
docker compose -f sa-deploy.yaml down

sleep 5

echo "Completed: All Stopped."