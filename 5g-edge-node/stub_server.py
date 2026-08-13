# stub_server.py: place in the same directory as capture.py, gets baked into
# the 5g-edge-node image (same Dockerfile COPY step that already copies
# capture.py, flow_builder.py, etc.)

import socket
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [STUB-SERVER] %(message)s")
log = logging.getLogger(__name__)

BIND_IP = "192.168.100.1"
PORTS = [443, 80, 8009]   # from original pcap's real dport distribution

def serve(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((BIND_IP, port))
    s.listen(50)
    log.info(f"Listening on {BIND_IP}:{port}")
    while True:
        try:
            conn, addr = s.accept()
            log.info(f"Accepted connection from {addr} on port {port}")
            conn.settimeout(2.0)
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
            except socket.timeout:
                pass
            conn.shutdown(socket.SHUT_WR)
            conn.close()
        except Exception as e:
            log.error(f"Error on port {port}: {e}")


def main():
    threads = []
    for port in PORTS:
        t = threading.Thread(target=serve, args=(port,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()