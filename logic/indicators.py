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

def compute_sma(prices): return sum(prices) / len(prices) if prices else None

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