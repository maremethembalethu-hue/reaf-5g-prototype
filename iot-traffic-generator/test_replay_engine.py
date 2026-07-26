import sys
import logging
from pathlib import Path
 
import replay_engine as re
 
log = logging.getLogger("TEST")
 
 
def step(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")