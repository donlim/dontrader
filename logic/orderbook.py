# trading_bot/logic/orderbook.py

from collections import deque

IMBALANCE_SMOOTHING = 0.2

class OrderbookTracker:
    def __init__(self):
        self.imbalance = 0.0

    def update(self, best_bid, bid_size, best_ask, ask_size):
        total = bid_size + ask_size
        if total == 0:
            raw_imbalance = 0
        else:
            raw_imbalance = (bid_size - ask_size) / total

        # Smooth the imbalance to reduce noise
        self.imbalance = (IMBALANCE_SMOOTHING * raw_imbalance
                          + (1 - IMBALANCE_SMOOTHING) * self.imbalance)

    def get_imbalance(self):
        return self.imbalance
