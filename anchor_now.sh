#!/bin/bash
# Anchors the custody chain of the running edge node (use after a Normal Run).
docker exec 5g-edge-node sh -c 'cd "$(dirname "$(find / -name chain_of_custody.py -not -path "*/site-packages/*" 2>/dev/null | head -1)")" && python3 chain_of_custody.py anchor'
