# Samples a fixed number of *complete* flows from a source PCAP, writing them
# to a new, much smaller PCAP that replay_engine.py can replay

import random
import argparse
from pathlib import Path

from scapy.utils import PcapReader, wrpcap
from scapy.all import IP, TCP, UDP

RANDOM_STATE = 42


def flow_key(pkt):
    ip = pkt[IP]
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        sport, dport = 0, 0
    a, b = (ip.src, sport), (ip.dst, dport)
    endpoints = tuple(sorted([a, b]))
    return (ip.proto, endpoints[0], endpoints[1])