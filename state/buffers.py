# trading_bot/state/buffers.py

from collections import deque, defaultdict
from trading_bot.config.config import SYMBOLS
import time
from trading_bot.config import parameters


# === Constants ===
BUFFER_SIZE = 500

# === Rolling Price/Volume Buffers ===
buffers = {symbol: deque(maxlen=BUFFER_SIZE) for symbol in SYMBOLS}

def update_buffer(symbol, price, volume):
    """
    Append latest price/volume (with timestamp) to buffer.
    """
    buffers[symbol].append({
        'timestamp': time.time(),
        'price': price,
        'volume': volume or 0.0
    })

def get_buffer(symbol):
    """
    Safely fetch buffer for a symbol.
    """
    return buffers.get(symbol, deque())

def get_latest_price(symbol):
    """
    Return last known price from buffer (or None).
    """
    buf = get_buffer(symbol)
    return buf[-1]['price'] if buf else None

def reset_buffers():
    """
    Clear all buffers — useful for testing or reinitialization.
    """
    for symbol in SYMBOLS:
        buffers[symbol].clear()
        orderbooks[symbol] = OrderBookData()

# === Order Book State ===
class OrderBookData:
    def __init__(self):
        self.mid_price = None
        self.imbalance = None
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

multi_timeframe_buffers = defaultdict(lambda: {
    tf: deque(maxlen=parameters.FEATURE_WINDOW)
    for tf in parameters.TIMEFRAMES
})

def update_multi_timeframe_buffers(symbol, price, volume, timestamp=None):
    """
    Update buffers for all configured timeframes with new tick data.
    """
    if timestamp is None:
        timestamp = time.time()

    for tf_name, tf_sec in parameters.TIMEFRAMES.items():
        buffer = multi_timeframe_buffers[symbol][tf_name]

        if tf_sec is None:
            # Tick-level data – append directly
            buffer.append({
                "timestamp": timestamp,
                "price": price,
                "volume": volume
            })
        else:
            # Aggregated timeframe logic (placeholder for now)
            buffer.append({
                "timestamp": timestamp,
                "price": price,   # In 3.3 we will implement OHLC aggregation here
                "volume": volume
            })

# === Signal Smoothing Buffers ===
signal_smoothing_buffers = defaultdict(lambda: deque(maxlen=parameters.SIGNAL_SMOOTH_WINDOW))

