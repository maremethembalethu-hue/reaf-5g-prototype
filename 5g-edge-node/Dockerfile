FROM python:3.11-slim

# Install system packages needed for packet capture
RUN apt-get update && apt-get install -y \
    tcpdump \
    iproute2 \
    net-tools \
    iputils-ping \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python files into the container
COPY . .

# When the container starts, run capture.py
CMD ["python3", "-u", "capture.py"]