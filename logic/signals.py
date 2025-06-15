# trading_bot/logic/signals.py

"""
Phase 3.5.2 Pro — Complete Modular Signal Engine with Sub-Score Outputs
"""

import numpy as np
from trading_bot.config import parameters

# =====================
# Individual Scoring Functions
# =====================

def delta_flow_signal(delta_flow):
    if delta_flow is None:
        return 0
    return np.tanh(delta_flow / 10.0)

def book_imbalance_signal(book_imb):
    if book_imb is None:
        return 0
    return np.tanh(book_imb * 3.0)

def book_pressure_signal(bid_density, ask_density):
    if bid_density is None or ask_density is None or ask_density == 0:
        return 0
    ratio = bid_density / ask_density
    return np.tanh((ratio - 1.0) * 2.0)

def slope_signal(bid_slope, ask_slope):
    if bid_slope is None or ask_slope is None:
        return 0
    slope_diff = bid_slope - ask_slope
    return np.tanh(slope_diff / 10.0)

def liquidity_gap_signal(min_bid_gap, min_ask_gap):
    if min_bid_gap is None or min_ask_gap is None:
        return 0
    gap_diff = min_ask_gap - min_bid_gap
    return np.tanh(gap_diff * 10.0)

def spread_signal(spread):
    if spread is None or spread <= 0:
        return 0
    return np.tanh(1.0 / spread)

def volatility_signal(stddev):
    if stddev is None or stddev <= 0:
        return 0
    return -np.tanh(stddev / 10.0)

# =====================
# Master Aggregation Logic
# =====================

def generate_signal(indicators):
    """
    Full signal aggregator blending sub-signals via weighted sum.
    Returns: signal, sub-scores dict, final_score
    """

    # Extract necessary features
    delta_flow = indicators.get("DELTA_FLOW")
    book_imb = indicators.get("BOOK_IMB")
    bid_density = indicators.get("BID_DENSITY")
    ask_density = indicators.get("ASK_DENSITY")
    bid_slope = indicators.get("BID_SLOPE")
    ask_slope = indicators.get("ASK_SLOPE")
    min_bid_gap = indicators.get("BID_GAP")
    min_ask_gap = indicators.get("ASK_GAP")
    spread = indicators.get("SPREAD")
    stddev = indicators.get("STDDEV")

    # Sub-signal scores
    scores = {
        "DELTA_FLOW": delta_flow_signal(delta_flow),
        "BOOK_IMB": book_imbalance_signal(book_imb),
        "PRESSURE": book_pressure_signal(bid_density, ask_density),
        "SLOPE": slope_signal(bid_slope, ask_slope),
        "LIQUIDITY_GAP": liquidity_gap_signal(min_bid_gap, min_ask_gap),
        "SPREAD": spread_signal(spread),
        "VOLATILITY": volatility_signal(stddev),
    }

    # Weighted sum aggregation
    total_score = 0
    total_weight = 0
    for key, score in scores.items():
        weight = parameters.SIGNAL_WEIGHTS.get(key, 0)
        total_score += score * weight
        total_weight += weight

    final_score = total_score / total_weight if total_weight > 0 else 0

    # Decision threshold
    if final_score > parameters.SIGNAL_THRESHOLD:
        signal = 'BUY'
    elif final_score < -parameters.SIGNAL_THRESHOLD:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return signal, scores, final_score