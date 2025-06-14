from collections import deque, defaultdict
from trading_bot.config.config import SYMBOLS

BUFFER_SIZE = 500

buffers = {symbol: deque(maxlen=BUFFER_SIZE) for symbol in SYMBOLS}

class OrderBookData:
    def __init__(self):
        self.mid_price = None
        self.imbalance = None

    def update(self, mid, imb):
        self.mid_price = mid
        self.imbalance = imb

orderbooks = defaultdict(OrderBookData)

def update_buffer(symbol, price, volume):
    buffers[symbol].append({'price': price, 'volume': volume})

def get_buffer(symbol):
    return buffers[symbol]