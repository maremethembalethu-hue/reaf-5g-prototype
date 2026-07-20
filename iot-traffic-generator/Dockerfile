FROM python:3.11-sli

# iproute2: ip command to detect uesimtun0 and get its IP
# iputils-ping: ping for tunnel verification
# net-tools: ifconfig as fallback
RUN apt-get update && apt-get install -y \
    iproute2 \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# -u flag forces Python to run in unbuffered mode, this means log output appears in docker logs immediately
CMD ["python3", "-u", "traffic_generator.py"]