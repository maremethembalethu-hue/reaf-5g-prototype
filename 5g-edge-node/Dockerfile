FROM python:3.11-slim

# Install system packages needed for packet capture
# tcpdump     — raw packet capture tool (also used as fallback)
# iproute2    — ip command for interface detection
# net-tools   — ifconfig for legacy interface listing
# iputils-ping — ping for connectivity testing
# libpcap-dev — C library that Scapy uses for raw socket capture
RUN apt-get update && apt-get install -y \
    tcpdump \
    iproute2 \
    net-tools \
    iputils-ping \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy and install Python dependencies first
# Doing this before copying code means Docker caches this layer
# and does not reinstall packages every time you change capture.py
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files into the container
COPY . .

# -u flag forces Python to run in unbuffered mode
# This means log output appears in docker logs immediately
# without buffering — critical for real-time forensic monitoring
CMD ["python3", "-u", "capture.py"]