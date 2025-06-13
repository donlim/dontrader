# trading_bot/state/buffers.py

from collections import defaultdict, deque

MAX_BUFFER_SIZE = 500

buffers = defaultdict(lambda: deque(maxlen=MAX_BUFFER_SIZE))

def update_buffer(symbol, price, timestamp):
    buffers[symbol].append({"price": price, "timestamp": timestamp})

def get_buffer(symbol):
    return list(buffers[symbol])
