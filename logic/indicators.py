# trading_bot/logic/indicators.py

import math
from collections import defaultdict
from scipy.stats import skew, kurtosis

ema_store = defaultdict(dict)

def update_ema(symbol, window, price):
    if price is None: return
    state = ema_store[symbol].setdefault(window, {"value": None, "alpha": 2 / (window + 1)})
    state["value"] = price if state["value"] is None else state["alpha"] * price + (1 - state["alpha"]) * state["value"]

def get_ema(symbol, window):
    state = ema_store[symbol].get(window)
    return state["value"] if state else None

def compute_sma(prices):
    return sum(prices) / len(prices) if prices else None

def compute_rsi(prices, window):
    if len(prices) < window + 1: return None
    gains, losses = [], []
    for i in range(1, window + 1):
        change = prices[-i] - prices[-i-1]
        (gains if change > 0 else losses).append(abs(change))
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(symbol):
    ema12, ema26 = get_ema(symbol, 12), get_ema(symbol, 26)
    return (ema12 - ema26) if ema12 and ema26 else None

def compute_momentum(prices, window):
    if len(prices) < window + 1: return None
    return prices[-1] - prices[-1 - window]

def compute_bollinger(prices, window, k=2.0):
    if len(prices) < window: return None
    sma = compute_sma(prices[-window:])
    variance = sum((p - sma) ** 2 for p in prices[-window:]) / window
    stddev = math.sqrt(variance)
    return (sma, sma + k * stddev, sma - k * stddev)

def compute_atr(prices, window):
    if len(prices) < window + 1: return None
    trs = [abs(prices[-i] - prices[-i - 1]) for i in range(1, window + 1)]
    return sum(trs) / window

def compute_stddev(prices, window):
    if len(prices) < window: return None
    sma = compute_sma(prices[-window:])
    variance = sum((p - sma) ** 2 for p in prices[-window:]) / window
    return math.sqrt(variance)

def compute_skew(prices, window):
    if len(prices) < window: return None
    return skew(prices[-window:])

def compute_kurtosis(prices, window):
    if len(prices) < window: return None
    return kurtosis(prices[-window:])

def compute_stoch_rsi(prices, rsi_window, stoch_window):
    if len(prices) < rsi_window + stoch_window:
        return None

    rsi_values = []
    for i in range(stoch_window):
        window_prices = prices[-(rsi_window + i):-(i) if i != 0 else None]
        rsi = compute_rsi(window_prices, rsi_window)
        if rsi is not None:
            rsi_values.append(rsi)

    if not rsi_values or len(rsi_values) < stoch_window:
        return None

    current_rsi = rsi_values[-1]
    min_rsi = min(rsi_values)
    max_rsi = max(rsi_values)
    if max_rsi - min_rsi == 0:
        return 0

    stoch_rsi = (current_rsi - min_rsi) / (max_rsi - min_rsi)
    return stoch_rsi

def compute_obv(prices, volumes):
    if not prices or not volumes or len(prices) != len(volumes):
        return None

    obv = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv += volumes[i]
        elif prices[i] < prices[i-1]:
            obv -= volumes[i]
    return obv

def compute_vwap(prices, volumes):
    if not prices or not volumes or len(prices) != len(volumes):
        return None

    cumulative_price_volume = sum(p * v for p, v in zip(prices, volumes))
    cumulative_volume = sum(volumes)
    if cumulative_volume == 0:
        return None

    return cumulative_price_volume / cumulative_volume

def compute_acc_dist(prices, highs, lows, closes, volumes):
    if not (len(prices) == len(highs) == len(lows) == len(closes) == len(volumes)):
        return None

    adl = 0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        if h - l == 0:
            continue
        mfm = ((c - l) - (h - c)) / (h - l)
        adl += mfm * v
    return adl

def compute_accumulation_distribution(prices, volumes, window):
    if not prices or not volumes or len(prices) != len(volumes):
        return None

    if len(prices) < window:
        return None

    ad = 0
    for i in range(window-1, len(prices)):
        high = max(prices[i-window+1:i+1])
        low = min(prices[i-window+1:i+1])
        close = prices[i]
        volume = volumes[i]

        if high == low:
            mfm = 0
        else:
            mfm = ((close - low) - (high - close)) / (high - low)
        ad += mfm * volume

    return ad

# trading_bot/logic/indicators.py

def detect_support_resistance(prices, window=20, tolerance=0.001):
    """
    Identify support and resistance levels in the price series.

    Args:
        prices: list of recent prices.
        window: number of periods to look back.
        tolerance: minimum price difference to treat levels as distinct.

    Returns:
        (support_level, resistance_level)
    """
    if len(prices) < window:
        return None, None

    recent_prices = prices[-window:]
    high = max(recent_prices)
    low = min(recent_prices)

    # Add tolerance logic if you want to avoid very close levels being treated as separate
    support_level = low * (1 - tolerance)
    resistance_level = high * (1 + tolerance)

    return support_level, resistance_level

def compute_book_imbalance(bids, asks, depth=5):
    total_bid = sum(sz for px, sz in bids[:depth])
    total_ask = sum(sz for px, sz in asks[:depth])

    if total_bid + total_ask == 0:
        return 0

    imbalance = (total_bid - total_ask) / (total_bid + total_ask)
    return imbalance

def compute_full_book_imbalance(bids, asks):
    total_bid = sum(sz for px, sz in bids)
    total_ask = sum(sz for px, sz in asks)

    if total_bid + total_ask == 0:
        return 0

    imbalance = (total_bid - total_ask) / (total_bid + total_ask)
    return imbalance

def compute_book_density(bids, asks, depth=5):
    bid_depth = sum(sz for px, sz in bids[:depth])
    ask_depth = sum(sz for px, sz in asks[:depth])
    return bid_depth, ask_depth

def compute_liquidity_gap(bids, asks):
    bid_gaps = [bids[i][0] - bids[i+1][0] for i in range(len(bids)-1)]
    ask_gaps = [asks[i+1][0] - asks[i][0] for i in range(len(asks)-1)]
    
    min_bid_gap = min(bid_gaps) if bid_gaps else None
    min_ask_gap = min(ask_gaps) if ask_gaps else None
    return min_bid_gap, min_ask_gap

def compute_spread(bids, asks):
    if not bids or not asks:
        return None
    return asks[0][0] - bids[0][0]

def compute_top_volatility(bids, asks):
    best_bid_prices = [px for px, _ in bids[:3]]
    best_ask_prices = [px for px, _ in asks[:3]]
    bid_vol = max(best_bid_prices) - min(best_bid_prices) if len(best_bid_prices) >= 2 else 0
    ask_vol = max(best_ask_prices) - min(best_ask_prices) if len(best_ask_prices) >= 2 else 0
    return bid_vol, ask_vol