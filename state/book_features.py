# trading_bot/state/book_features.py

from collections import deque, defaultdict

FEATURE_WINDOW = 10  # how many recent snapshots to smooth over

class BookFeatureBuffer:
    def __init__(self):
        self.features = deque(maxlen=FEATURE_WINDOW)
        self.delta_features = deque(maxlen=FEATURE_WINDOW)

    # ✅ Update liquidity features (density, spread, gaps, etc.)
    def update(self, feature_dict):
        self.features.append(feature_dict)

    # ✅ Update delta flow (aggressive buyer/seller volume estimates)
    def update_delta(self, buy: float, sell: float):
        self.delta_features.append({'buy': buy, 'sell': sell})

    # ✅ Smooth all liquidity features over window
    def get_smoothed(self):
        if not self.features:
            return {}

        avg = {}
        keys = self.features[0].keys()

        for key in keys:
            total = sum(f[key] for f in self.features)
            avg[key] = total / len(self.features)
        return avg

    # ✅ Smooth delta flow
    def get_delta_flow(self):
        if not self.delta_features:
            return 0.0

        total_buy = sum(f['buy'] for f in self.delta_features)
        total_sell = sum(f['sell'] for f in self.delta_features)

        net_flow = total_buy - total_sell

        # Normalize by window size for stability
        normalized_flow = net_flow / len(self.delta_features)
        return normalized_flow

# ✅ One feature buffer per symbol
book_feature_buffers = defaultdict(BookFeatureBuffer)