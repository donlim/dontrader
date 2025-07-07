# trading_bot/tools/audit_indicators.py
import sys
import os
import numpy as np

# Ensure the 'trading_bot' parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from logic.indicators import compute_all_indicators
from logic.signals import generate_signal
from config import parameters
from collections import defaultdict

def audit_indicators():
    # Realistic dummy price data
    dummy_prices = np.linspace(100, 110, 250).tolist()
    dummy_volumes = [100] * len(dummy_prices)
    dummy_highs = [p + 0.5 for p in dummy_prices]
    dummy_lows = [p - 0.5 for p in dummy_prices]
    dummy_opens = dummy_prices
    dummy_closes = dummy_prices

    dummy_bids = [[100, 1]] * 10
    dummy_asks = [[101, 1]] * 10
    class DummyBookBuffer:
        def get_delta_flow(self):
            return 0.0  # or any dummy value

    # Create a dummy buffer dict with 'TEST' key
    dummy_book_buffers = defaultdict(DummyBookBuffer)
    dummy_book_buffers["TEST"] = DummyBookBuffer()

    # Compute indicators safely
    indicators_dict = compute_all_indicators(
        prices=dummy_prices,
        volumes=dummy_volumes,
        highs=dummy_highs,
        lows=dummy_lows,
        opens=dummy_opens,
        closes=dummy_closes,
        bids=[],
        asks=[],
        symbol="TEST",
        book_feature_buffers=dummy_book_buffers,  # ✅ use dummy
    )

    compute_keys = set(indicators_dict.keys())
    weight_keys = set(parameters.SIGNAL_WEIGHTS.keys())

    # Generate signal test
    signal, category_subscores, final_score, normalized_scores, meta_confidence, mode, top_indicators = generate_signal(indicators_dict)
    generate_keys = set(normalized_scores.keys())

    missing_in_generate = compute_keys - generate_keys
    missing_in_weights = compute_keys - weight_keys

    print(f"✅ Total computed indicators: {len(compute_keys)}")
    print(f"🔴 Missing in generate_signal: {missing_in_generate}")
    print(f"🔴 Missing in SIGNAL_WEIGHTS: {missing_in_weights}")

if __name__ == "__main__":
    audit_indicators()