FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    tcpdump \
    iproute2 \
    net-tools \
    iputils-ping \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy and install Python dependencies first, Docker caches this layer and does not reinstall packages every time you change capture.py
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files into the container
COPY . .

# -u flag forces Python to run in unbuffered mode, this means log output appears in docker logs immediately
#CMD ["python3", "-u", "capture.py"]
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]