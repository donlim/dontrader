# trading_bot/state/buffers.py

from collections import deque, defaultdict
from trading_bot.config.config import SYMBOLS

BUFFER_SIZE = 500

buffers = {symbol: deque(maxlen=BUFFER_SIZE) for symbol in SYMBOLS}

class OrderBookData:
    def __init__(self):
        self.mid_price = None
        self.imbalance = None  # full book imbalance
        self.bids = []
        self.asks = []

    def update(self, mid, imbalance, bids, asks):
        self.mid_price = mid
        self.imbalance = imbalance
        self.bids = bids
        self.asks = asks

    def get_mid(self):
        return self.mid_price

    def get_imbalance(self):
        return self.imbalance

    def get_depth(self):
        return self.bids, self.asks

orderbooks = defaultdict(OrderBookData)

def update_buffer(symbol, price, volume):
    buffers[symbol].append({
        'price': price,
        'volume': volume
    })

def get_buffer(symbol):
    return buffers[symbol]