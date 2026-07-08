FROM python:3.11-slim

# iproute2    — ip command to detect uesimtun0 and get its IP
# iputils-ping — ping for tunnel verification
# net-tools   — ifconfig as fallback
RUN apt-get update && apt-get install -y \
    iproute2 \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# -u flag: unbuffered output so docker logs shows lines immediately
CMD ["python3", "-u", "traffic_generator.py"]