import logging
from collections import deque
 
class RecentLogHandler(logging.Handler):
    #Keeps the last `capacity` formatted log lines in memory.
    def __init__(self, capacity=200):
        super().__init__()
        self.buffer = deque(maxlen=capacity)
 
    def emit(self, record):
        self.buffer.append(self.format(record))
 
    def get_recent(self, n=100):
        return list(self.buffer)[-n:]
 
 
recent_log_handler = RecentLogHandler(capacity=200)
recent_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))