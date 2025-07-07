# trading_bot/logic/indicators.py

import math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import skew, kurtosis
from trading_bot.config import parameters

ema_store = defaultdict(dict)

def update_ema(symbol, window, price):
    if price is None: return
    state = ema_store[symbol].setdefault(window, {"value": None, "alpha": 2 / (window + 1)})
    state["value"] = price if state["value"] is None else state["alpha"] * price + (1 - state["alpha"]) * state["value"]

def get_ema(symbol, window):
    state = ema_store[symbol].get(window)
    return state["value"] if state else None

def compute_ema(prices, window):
    if len(prices) < window:
        return None
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    ema = np.convolve(prices, weights, mode='valid')
    return float(ema[-1]) if len(ema) else None

def compute_rsi(prices, window):
    if len(prices) < window + 1:
        return None

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1] if not rsi.isna().all() else None

def compute_macd(symbol):
    ema12, ema26 = get_ema(symbol, 12), get_ema(symbol, 26)
    return (ema12 - ema26) if ema12 and ema26 else None

def compute_momentum(prices, window):
    if len(prices) <= window:
        return None
    return prices.iloc[-1] - prices.iloc[-1 - window]

def compute_bollinger(prices, window, k=2.0):
    if len(prices) < window:
        return None

    price_series = pd.Series(prices[-window:])
    sma = compute_sma(price_series, window)
    if sma is None:
        return None

    variance = sum((p - sma) ** 2 for p in price_series) / window
    stddev = math.sqrt(variance)

    return (sma, sma + k * stddev, sma - k * stddev)

def compute_atr(prices, window):
    if len(prices) < window + 1:
        return None
    trs = [abs(prices.iloc[-i] - prices.iloc[-i - 1]) for i in range(1, window + 1)]
    return sum(trs) / window

def compute_stddev(prices, window=20):
    if len(prices) < window:
        return None

    sma = compute_sma(pd.Series(prices[-window:]), window)
    if sma is None:
        return None

    variance = sum((p - sma) ** 2 for p in prices[-window:]) / window
    return math.sqrt(variance)
    
def compute_skew(prices, window):
    if len(prices) < window: return None
    return skew(prices[-window:])

def compute_kurtosis(prices, window):
    if len(prices) < window: return None
    return kurtosis(prices[-window:])

def compute_stoch_rsi(prices, rsi_window, stoch_window):
    prices = pd.Series(prices)
    if len(prices) < rsi_window + stoch_window:
        return None

    # 1. Compute full RSI series
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))

    # 2. Take last stoch_window of RSI
    recent_rsi = rsi_series[-stoch_window:]

    if recent_rsi.isna().any():
        return None

    min_rsi = recent_rsi.min()
    max_rsi = recent_rsi.max()

    if max_rsi - min_rsi == 0:
        return 0

    current_rsi = recent_rsi.iloc[-1]
    return (current_rsi - min_rsi) / (max_rsi - min_rsi)

def compute_obv(prices, volumes):
    if prices is None or volumes is None:
        return None
    if len(prices) == 0 or len(volumes) == 0 or len(prices) != len(volumes):
        return None

    obv = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv += volumes[i]
        elif prices[i] < prices[i - 1]:
            obv -= volumes[i]
    return obv

def compute_vwap(prices, volumes):
    if prices is None or volumes is None:
        return None
    if len(prices) != len(volumes) or prices.empty or volumes.empty:
        return None

    cumulative_price_volume = (prices * volumes).sum()
    cumulative_volume = volumes.sum()
    if cumulative_volume == 0:
        return None

    return cumulative_price_volume / cumulative_volume

def compute_adl(prices, highs, lows, closes, volumes):
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
    if prices is None or volumes is None:
        return None
    if len(prices) == 0 or len(volumes) == 0 or len(prices) != len(volumes):
        return None
    if len(prices) < window:
        return None

    ad = 0
    for i in range(window - 1, len(prices)):
        high = max(prices[i - window + 1:i + 1])
        low = min(prices[i - window + 1:i + 1])
        close = prices[i]
        volume = volumes[i]

        if high == low:
            mfm = 0
        else:
            mfm = ((close - low) - (high - close)) / (high - low)

        ad += mfm * volume

    return ad

# trading_bot/logic/indicators.py

def compute_support_resistance(prices, window=20, tolerance=0.001):
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

def compute_slope(prices, window=10):
    if len(prices) < window:
        return None
    y = prices[-window:]
    x = np.arange(window)
    slope, _ = np.polyfit(x, y, 1)
    return slope

def compute_book_density(bids, asks, depth=5):
    bid_depth = sum(sz for px, sz in bids[:depth])
    ask_depth = sum(sz for px, sz in asks[:depth])
    return (bid_depth + ask_depth) / 2  # or return bid_depth or ask_depth

def compute_bid_density(bids, depth=5):
    return sum(sz for px, sz in bids[:depth])

def compute_ask_density(asks, depth=5):
    return sum(sz for px, sz in asks[:depth])

def compute_spread(bids, asks):
    if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
        return None

    best_bid_price = max(bids, key=lambda x: x[0])[0]
    best_ask_price = min(asks, key=lambda x: x[0])[0]

    return best_ask_price - best_bid_price

def compute_book_pressure_ratio(bids, asks, depth=5):
    bid_vol = sum(sz for px, sz in bids[:depth])
    ask_vol = sum(sz for px, sz in asks[:depth])
    
    if ask_vol == 0:
        return None  # or float('inf') if you want to show max pressure
    return bid_vol / ask_vol

def compute_pressure(bid_density, ask_density):
    if bid_density is None or ask_density in (None, 0):
        return None
    return (bid_density / ask_density - 1.0) * 2.0  # raw version of your signal

def compute_bid_ask_vol(bids, asks, depth=5):
    bid_vol = sum(sz for px, sz in bids[:depth])
    ask_vol = sum(sz for px, sz in asks[:depth])
    return bid_vol, ask_vol

def compute_top_of_book_volatility(bids, asks):
    best_bid_prices = [px for px, _ in bids[:3]]
    best_ask_prices = [px for px, _ in asks[:3]]

    bid_vol = max(best_bid_prices) - min(best_bid_prices) if len(best_bid_prices) >= 2 else 0
    ask_vol = max(best_ask_prices) - min(best_ask_prices) if len(best_ask_prices) >= 2 else 0

    avg_vol = (bid_vol + ask_vol) / 2
    return avg_vol


def compute_rolling_vwap(prices, volumes, window):
    if prices is None or volumes is None:
        return None
    if prices.empty or volumes.empty or len(prices) != len(volumes):
        return None

    if len(prices) < window:
        return None

    price_window = prices[-window:]
    volume_window = volumes[-window:]

    total_pv = sum(p * v for p, v in zip(price_window, volume_window))
    total_vol = sum(volume_window)

    if total_vol == 0:
        return None

    return total_pv / total_vol

def compute_anchored_vwap(prices, volumes, anchor_index=0):
    """
    VWAP starting from anchor_index up to the latest price.
    anchor_index = 0 means full period.
    """
    if prices is None or volumes is None:
        return None
    if prices.empty or volumes.empty or len(prices) != len(volumes):
        return None

    if anchor_index >= len(prices):
        return None

    price_window = prices[anchor_index:]
    volume_window = volumes[anchor_index:]

    total_pv = sum(p * v for p, v in zip(price_window, volume_window))
    total_vol = sum(volume_window)

    if total_vol == 0:
        return None

    return total_pv / total_vol

def compute_bid_ask_gap(bids, asks):
    bid_gaps = [bids[i][0] - bids[i+1][0] for i in range(len(bids)-1)]
    ask_gaps = [asks[i+1][0] - asks[i][0] for i in range(len(asks)-1)]

    min_bid_gap = min(bid_gaps) if bid_gaps else None
    min_ask_gap = min(ask_gaps) if ask_gaps else None
    return min_bid_gap, min_ask_gap

def compute_book_slope(bids, asks, depth=5):
    """
    Estimate order book steepness using linear regression on price vs size.
    """
    # Dynamically adjust depth to available size
    depth = min(depth, len(bids), len(asks))

    if depth == 0:
        print(f"[DEBUG] Skipping slope calc: bids={len(bids)}, asks={len(asks)}")
        return None, None

    bid_px = np.array([px for px, sz in bids[:depth]])
    bid_sz = np.array([sz for px, sz in bids[:depth]])
    ask_px = np.array([px for px, sz in asks[:depth]])
    ask_sz = np.array([sz for px, sz in asks[:depth]])

    bid_slope = np.polyfit(bid_px, bid_sz, 1)[0]
    ask_slope = np.polyfit(ask_px, ask_sz, 1)[0]

    return bid_slope, ask_slope

def compute_adx(prices, highs, lows, window):
    if len(prices) < window + 1:
        return None

    plus_dm, minus_dm, tr = [], [], []
    for i in range(1, len(prices)):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
        tr.append(max(highs[i] - lows[i], abs(highs[i] - prices[i-1]), abs(lows[i] - prices[i-1])))

    tr_sum = pd.Series(tr).rolling(window).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(window).sum() / tr_sum
    minus_di = 100 * pd.Series(minus_dm).rolling(window).sum() / tr_sum
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx.iloc[-1] if not adx.empty else None

def compute_chaikin_oscillator(highs, lows, closes, volumes, short_period=3, long_period=10):
    if len(closes) < long_period:
        return None
    mf_multiplier = [(2 * c - h - l) / (h - l) if h - l != 0 else 0 for h, l, c in zip(highs, lows, closes)]
    mf_volume = [m * v for m, v in zip(mf_multiplier, volumes)]
    adl = np.cumsum(mf_volume)
    ema_short = pd.Series(adl).ewm(span=short_period).mean()
    ema_long = pd.Series(adl).ewm(span=long_period).mean()
    return (ema_short - ema_long).iloc[-1]

def compute_donchian_channels(highs, lows, window):
    if len(highs) < window or len(lows) < window:
        return None
    upper = max(highs[-window:])
    lower = min(lows[-window:])
    return upper, lower

def compute_parabolic_sar(highs, lows, step=0.02, max_step=0.2):
    """
    Compute Parabolic SAR for backtesting (simplified non-stateful version).
    Returns a pandas Series.
    """
    import numpy as np
    import pandas as pd

    highs = pd.Series(highs).reset_index(drop=True)
    lows = pd.Series(lows).reset_index(drop=True)

    if len(highs) < 2 or len(lows) < 2:
        return pd.Series([None] * len(highs))

    length = len(highs)
    sar = [None] * length

    # Initialization
    trend = 1  # 1 = up, -1 = down
    af = step
    ep = highs[0] if trend == 1 else lows[0]
    sar[0] = lows[0] if trend == 1 else highs[0]

    for i in range(1, length):
        prev_sar = sar[i - 1]
        if prev_sar is None:
            sar[i] = lows[i] if trend == 1 else highs[i]
            continue

        # SAR Calculation
        if trend == 1:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], lows[i - 1], lows[i])
            if highs[i] > ep:
                ep = highs[i]
                af = min(af + step, max_step)
            if lows[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = lows[i]
                af = step
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], highs[i - 1], highs[i])
            if lows[i] < ep:
                ep = lows[i]
                af = min(af + step, max_step)
            if highs[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = highs[i]
                af = step

    return pd.Series(sar)

def compute_trend_strength(prices, highs, lows, window=None):
    """
    Composite trend strength score using ADX, stddev, and slope.
    """
    if window is None:
        window = parameters.TREND_STRENGTH_STDDEV_WINDOW

    if len(prices) < window or len(highs) < window or len(lows) < window:
        return None

    adx = compute_adx(prices, highs, lows, parameters.ADX_WINDOW)
    stddev = compute_stddev(prices, parameters.STDDEV_WINDOW)

    y = np.array(prices[-window:])
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    slope_angle = math.degrees(math.atan(slope))
    slope_score = min(max(abs(slope_angle), 0), 45) / 45 * 100

    if adx is None or stddev is None:
        return None

    trend_score = (
        parameters.TREND_STRENGTH_ADX_WEIGHT * adx +
        parameters.TREND_STRENGTH_STDDEV_WEIGHT * stddev +
        parameters.TREND_STRENGTH_SLOPE_WEIGHT * slope_score
    )
    return trend_score


def compute_sma(series, window=200):
    series = pd.Series(series)
    if len(series) < window:
        return None

    sma = series.rolling(window=window).mean()
    last_valid = sma.dropna().iloc[-1] if not sma.dropna().empty else None
    return last_valid

def compute_roc(prices, period):
    if len(prices) < period + 1:
        return None

    series = pd.Series(prices)
    return ((series.iloc[-1] - series.iloc[-period - 1]) / series.iloc[-period - 1]) * 100

def compute_cci(highs, lows, closes, period):
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)

    tp = (highs + lows + closes) / 3
    tp_series = pd.Series(tp)

    sma = tp_series.rolling(window=period).mean()
    mad = tp_series.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)

    cci = (tp_series - sma) / (0.015 * mad)

    return cci.iloc[-1] if not cci.isna().all() else None

def compute_cmf(highs, lows, closes, volumes, window):
    # Convert to NumPy arrays
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)
    volumes = np.array(volumes)

    mfv = ((closes - lows) - (highs - closes)) / (highs - lows + 1e-9) * volumes
    mfr = np.convolve(mfv, np.ones(window), mode='valid')
    vol_sum = np.convolve(volumes, np.ones(window), mode='valid')

    cmf = mfr / (vol_sum + 1e-9)
    return cmf[-1] if len(cmf) > 0 else None

def compute_heikin_ashi_ratio(opens, highs, lows, closes):
    if not all(len(arr) > 0 for arr in [opens, highs, lows, closes]):
        return None

    # Convert to NumPy arrays for vectorized math
    opens = np.array(opens)
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)

    ha_close = (opens + highs + lows + closes) / 4

    # TODO: Replace this with actual logic for HA ratio
    return ha_close[-1]  # Placeholder — return last ha_close as example

def compute_donchian_width(highs, lows, period=20):
    if len(highs) < period or len(lows) < period:
        return None

    highs_series = pd.Series(highs)
    lows_series = pd.Series(lows)

    upper = highs_series.rolling(window=period).max().iloc[-1]
    lower = lows_series.rolling(window=period).min().iloc[-1]

    if upper is None or lower is None or lower == 0:
        return 0

    return (upper - lower) / lower

def compute_donchian(prices, highs, lows, window):
    if len(prices) < window or len(highs) < window or len(lows) < window:
        return None

    price = prices.iloc[-1] if not prices.empty else None
    upper, lower = compute_donchian_channels(highs, lows, window)

    if upper is None or lower is None or price is None or upper == lower:
        return None

    return (price - lower) / (upper - lower)  # normalized value between 0 and 1

def compute_tsi(prices, fast_period, slow_period):
    if len(prices) < slow_period + 1:
        return None

    series = pd.Series(prices)
    momentum = series.diff()
    abs_momentum = momentum.abs()

    ema1 = momentum.ewm(span=fast_period).mean()
    ema2 = ema1.ewm(span=slow_period).mean()

    abs_ema1 = abs_momentum.ewm(span=fast_period).mean()
    abs_ema2 = abs_ema1.ewm(span=slow_period).mean()

    tsi = 100 * (ema2 / abs_ema2)
    return tsi.iloc[-1] if not tsi.empty else None

def compute_kvo(closes, volumes, fast=34, slow=55):
    if len(closes) < slow + 1:
        return None
    hlc = (closes + closes.shift(1)) / 2
    trend = np.where(hlc > hlc.shift(1), 1, -1)
    volume_force = volumes * trend
    kvo = volume_force.ewm(span=fast).mean() - volume_force.ewm(span=slow).mean()
    return kvo.iloc[-1]

def compute_williams_r(highs, lows, closes, period=14):
    if len(highs) < period:
        return None
    highest_high = highs.rolling(window=period).max()
    lowest_low = lows.rolling(window=period).min()
    r = (highest_high - closes) / (highest_high - lowest_low + 1e-9) * -100
    return r.iloc[-1]

def compute_vortex(highs, lows, closes, window):
    if len(highs) < window + 1 or len(lows) < window + 1:
        return None, None
    vm_plus = [abs(highs[i] - lows[i-1]) for i in range(1, len(highs))]
    vm_minus = [abs(lows[i] - highs[i-1]) for i in range(1, len(lows))]
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(highs))]
    sum_vm_plus = pd.Series(vm_plus).rolling(window).sum()
    sum_vm_minus = pd.Series(vm_minus).rolling(window).sum()
    sum_tr = pd.Series(tr).rolling(window).sum()
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    return vi_plus.iloc[-1], vi_minus.iloc[-1]

def update_ema(symbol, window, price):
    if price is None: return
    state = ema_store[symbol].setdefault(window, {"value": None, "alpha": 2 / (window + 1)})
    state["value"] = price if state["value"] is None else state["alpha"] * price + (1 - state["alpha"]) * state["value"]

def compute_dema(prices, window):
    if len(prices) < 2 * window:
        return None
    ema1 = pd.Series(prices).ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    dema = 2 * ema1 - ema2
    return dema.iloc[-1]

def compute_tema(prices, window):
    if len(prices) < 3 * window:
        return None
    ema1 = pd.Series(prices).ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema.iloc[-1]

def compute_trix(prices, window):
    if len(prices) < 3 * window:
        return None
    ema1 = pd.Series(prices).ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    trix = ema3.pct_change() * 100
    return trix.iloc[-1]

def compute_cmo(prices, window):
    if len(prices) < window:
        return None
    diff = np.diff(prices)
    up = np.sum(diff[-window:][diff[-window:] > 0])
    down = np.sum(np.abs(diff[-window:][diff[-window:] < 0]))
    if up + down == 0:
        return 0
    return 100 * (up - down) / (up + down)

def compute_vortex(highs, lows, closes, window):
    if len(highs) < window + 1 or len(lows) < window + 1:
        return None, None
    vm_plus = [abs(highs[i] - lows[i-1]) for i in range(1, len(highs))]
    vm_minus = [abs(lows[i] - highs[i-1]) for i in range(1, len(lows))]
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(highs))]
    sum_vm_plus = pd.Series(vm_plus).rolling(window).sum()
    sum_vm_minus = pd.Series(vm_minus).rolling(window).sum()
    sum_tr = pd.Series(tr).rolling(window).sum()
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    return vi_plus.iloc[-1], vi_minus.iloc[-1]

# ✅ Cycle / Smoothing

def compute_fisher(prices, window):
    if len(prices) < window:
        return None
    price_series = pd.Series(prices[-window:])
    min_val = price_series.min()
    max_val = price_series.max()
    value = 2 * ((price_series.iloc[-1] - min_val) / (max_val - min_val) - 0.5)
    value = max(min(value, 0.999), -0.999)
    return 0.5 * math.log((1 + value) / (1 - value))

def compute_supertrend(highs, lows, closes, period=10, multiplier=3):
    if len(highs) < period:
        return None
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    hl2 = (df["high"] + df["low"]) / 2
    tr = df["high"] - df["low"]
    atr = tr.rolling(period).mean()
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    in_uptrend = True
    if df["close"].iloc[-1] > upperband.iloc[-2]:
        in_uptrend = True
    elif df["close"].iloc[-1] < lowerband.iloc[-2]:
        in_uptrend = False
    return in_uptrend

# Hilbert Transform placeholder (complex DSP, usually library-based)
def compute_hilbert_dcp(prices):
    return None  # For future DSP modeling, not common in basic Python

# ✅ Volume / Accumulation

def compute_eom(highs, lows, closes, volumes, window=14):
    if highs is None or lows is None or closes is None or volumes is None:
        return None
    if len(highs) == 0 or len(lows) == 0 or len(closes) == 0 or len(volumes) == 0:
        return None
    if not (len(highs) == len(lows) == len(closes) == len(volumes)):
        return None
    if len(highs) < window:
        return None

    eom_values = []
    for i in range(1, len(highs)):
        distance_moved = ((highs[i] + lows[i]) / 2) - ((highs[i - 1] + lows[i - 1]) / 2)
        box_ratio = volumes[i] / (highs[i] - lows[i]) if (highs[i] - lows[i]) != 0 else 0
        eom = distance_moved / box_ratio if box_ratio != 0 else 0
        eom_values.append(eom)

    return sum(eom_values[-window:]) / window if len(eom_values) >= window else None

def compute_mfi(highs, lows, closes, volumes, window):
    if len(closes) < window:
        return None
    tp = (highs + lows + closes) / 3
    raw_mf = tp * volumes
    pos_mf = raw_mf.where(tp > tp.shift(1), 0)
    neg_mf = raw_mf.where(tp < tp.shift(1), 0)
    pos_sum = pos_mf.rolling(window).sum()
    neg_sum = neg_mf.rolling(window).sum()
    mfi = 100 - (100 / (1 + (pos_sum / (neg_sum + 1e-9))))
    return mfi.iloc[-1]

def compute_force_index(prices, volumes, window=13):
    if len(prices) < window + 1:
        return None
    force = pd.Series(prices).diff() * volumes
    return force.rolling(window).mean().iloc[-1]

def compute_volume_oscillator(volumes, short_window=14, long_window=28):
    if len(volumes) < long_window:
        return None
    short_ema = pd.Series(volumes).ewm(span=short_window, adjust=False).mean()
    long_ema = pd.Series(volumes).ewm(span=long_window, adjust=False).mean()
    vo = 100 * (short_ema - long_ema) / long_ema
    return vo.iloc[-1]

def compute_keltner_channels(prices, highs, lows, window=20, multiplier=2):
    if len(prices) < window or len(highs) < window or len(lows) < window:
        return None
    typical_price = [(h + l + c) / 3 for h, l, c in zip(highs, lows, prices)]
    tp_series = pd.Series(typical_price)
    ema = tp_series.ewm(span=window, adjust=False).mean()
    atr = pd.Series([h - l for h, l in zip(highs, lows)]).rolling(window).mean()
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    return upper.iloc[-1], lower.iloc[-1]

def compute_fractal_bands(prices, window=2):
    if len(prices) < window * 2 + 1:
        return None
    high_band = max(prices[-(window+1):-window])
    low_band = min(prices[-(window+1):-window])
    return high_band, low_band

def compute_zscore(prices, window=20):
    if len(prices) < window:
        return None
    mean = np.mean(prices[-window:])
    std = np.std(prices[-window:])
    return (prices.iloc[-1] - mean) / std if std != 0 else 0

def compute_candle_body_ratio(opens, closes, highs, lows):
    if any(x is None or not isinstance(x, pd.Series) or x.empty for x in [opens, closes, highs, lows]):
        return None
    body = abs(closes.iloc[-1] - opens.iloc[-1])
    range_total = highs.iloc[-1] - lows.iloc[-1]
    return body / range_total if range_total != 0 else 0

def compute_wick_percent(opens, closes, highs, lows):
    import pandas as pd

    if any(x is None or not isinstance(x, pd.Series) or x.empty for x in [opens, closes, highs, lows]):
        return None, None

    body_high = max(opens.iloc[-1], closes.iloc[-1])
    body_low = min(opens.iloc[-1], closes.iloc[-1])
    total_range = highs.iloc[-1] - lows.iloc[-1]
    upper_wick = highs.iloc[-1] - body_high
    lower_wick = body_low - lows.iloc[-1]

    return (
        upper_wick / total_range if total_range != 0 else 0,
        lower_wick / total_range if total_range != 0 else 0
    )

def compute_three_bar_reversal(closes):
    if len(closes) < 3:
        return None
    return (
        closes.iloc[-3] > closes.iloc[-2] < closes.iloc[-1] or
        closes.iloc[-3] < closes.iloc[-2] > closes.iloc[-1]
    )

def compute_engulfing_candle(opens, closes):
    if len(opens) < 2 or len(closes) < 2:
        return None

    prev_body = closes.iloc[-2] - opens.iloc[-2]
    curr_body = closes.iloc[-1] - opens.iloc[-1]

    return (
        (prev_body < 0 and curr_body > 0 and opens.iloc[-1] < closes.iloc[-2] and closes.iloc[-1] > opens.iloc[-2]) or
        (prev_body > 0 and curr_body < 0 and opens.iloc[-1] > closes.iloc[-2] and closes.iloc[-1] < opens.iloc[-2])
    )

def compute_max_drawdown(prices, window=100):
    if len(prices) < window:
        return None
    rolling_max = pd.Series(prices[-window:]).cummax()
    drawdowns = pd.Series(prices[-window:]) / rolling_max - 1
    return drawdowns.min()

def compute_sharpe(prices, window=100, risk_free_rate=0.0):
    if len(prices) < window + 1:
        return None
    returns = np.diff(prices[-(window + 1):]) / prices[-(window + 1):-1]
    excess_returns = returns - risk_free_rate / 252
    avg_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    return (avg_return / std_return) * np.sqrt(252) if std_return != 0 else None

def compute_ulcer_index(prices, window=100):
    window_prices = pd.Series(prices[-window:]).dropna()
    if len(window_prices) < window:
        return 0

    rolling_max = window_prices.cummax()
    percent_drawdown = 100 * (window_prices - rolling_max) / rolling_max
    squared_drawdown = percent_drawdown ** 2
    result = np.sqrt(np.mean(squared_drawdown))

    print(f"[DEBUG] Ulcer Index result = {result}, type = {type(result)}")

    return result

import numpy as np

def compute_avg_holding_time(entry_exit_pairs):
    """
    entry_exit_pairs: List or Series of (entry_time, exit_time) datetime tuples
    """
    if entry_exit_pairs is None or len(entry_exit_pairs) == 0:
        return None

    # Ensure valid tuples before proceeding
    valid_pairs = [
        (entry, exit) for entry_exit in entry_exit_pairs
        if isinstance(entry_exit, (list, tuple)) and len(entry_exit) == 2
        for entry, exit in [entry_exit]
    ]

    if not valid_pairs:
        return None

    holding_durations = [
        (exit_time - entry_time).total_seconds()
        for entry_time, exit_time in valid_pairs
    ]

    return np.mean(holding_durations) if holding_durations else None

def compute_all_indicators(prices, volumes, highs, lows, opens, closes, bids, asks, symbol, book_feature_buffers, trade_pairs=None):
    prices = pd.Series(prices)
    volumes = pd.Series(volumes)
    highs = pd.Series(highs)
    lows = pd.Series(lows)
    opens = pd.Series(opens)
    closes = pd.Series(closes)
    bids = pd.Series(bids)
    asks = pd.Series(asks)
    indicators_dict = {}
    print(f"[DEBUG] prices len = {len(prices)}, SMA50 = {compute_sma(prices, 50)}")

    # === Trend
    indicators_dict.update({
        "PRICE": prices.iloc[-1] if not prices.empty else None,
        "EMA10": get_ema(symbol, 10) or 0.0,
        "EMA50": get_ema(symbol, 50) or 0.0,
        "EMA100": get_ema(symbol, 100) or 0.0,
        "EMA200": get_ema(symbol, 200) or 0.0,
        "EMA_DIFF": (get_ema(symbol, 10) or 0.0) - (get_ema(symbol, 50) or 0.0),
        "SMA50": compute_sma(prices, 50),
        "SMA200": compute_sma(prices, 200),
        "SMA": compute_sma(prices, 50),  # alias, optional
        "SMA_DIFF": compute_sma(prices, 50) - compute_sma(prices, 200),
        "MACD": compute_macd(symbol),
        "ADX": compute_adx(highs, lows, prices, parameters.ADX_WINDOW),
        "PARABOLIC_SAR": float(compute_parabolic_sar(highs, lows, parameters.PARABOLIC_SAR_STEP, parameters.PARABOLIC_SAR_MAX_STEP).iloc[-1]) if not highs.empty and not lows.empty else None,
        "TREND_STRENGTH": compute_trend_strength(prices, highs, lows),
        "DONCHIAN": compute_donchian(prices, highs, lows, parameters.DONCHIAN_WINDOW),
        "DONCHIAN_UPPER": compute_donchian_channels(highs, lows, parameters.DONCHIAN_WINDOW)[0],
        "DONCHIAN_LOWER": compute_donchian_channels(highs, lows, parameters.DONCHIAN_WINDOW)[1],
        "DONCHIAN_WIDTH": compute_donchian_width(highs, lows, parameters.DONCHIAN_WINDOW),
        "HEIKIN_RATIO": compute_heikin_ashi_ratio(opens, highs, lows, closes),
        "CMF": compute_cmf(highs, lows, closes, volumes, parameters.CMF_WINDOW),
        "TEMA": compute_tema(prices, parameters.TEMA_WINDOW),
        "DEMA": compute_dema(prices, parameters.DEMA_WINDOW),
    })

    # === Momentum
    indicators_dict.update({
        "RSI": compute_rsi(prices, parameters.RSI_WINDOW),
        "STOCH_RSI": compute_stoch_rsi(prices, parameters.RSI_WINDOW, parameters.STOCH_WINDOW),
        "MOMENTUM": compute_momentum(prices, parameters.MOMENTUM_WINDOW),
        "CCI": compute_cci(highs, lows, closes, parameters.CCI_WINDOW),
        "ROC": compute_roc(prices, parameters.ROC_WINDOW),
        "TSI": compute_tsi(prices, parameters.TSI_FAST, parameters.TSI_SLOW),
        "KVO": compute_kvo(closes, volumes, parameters.KVO_FAST, parameters.KVO_SLOW),
        "WILLIAMS_R": compute_williams_r(highs, lows, closes, parameters.WILLIAMS_R_WINDOW),
        "CMO": compute_cmo(prices, parameters.CMO_WINDOW),
        "TRIX": compute_trix(prices, parameters.TRIX_WINDOW),
        "VI_PLUS": compute_vortex(highs, lows, closes, parameters.VI_WINDOW)[0],
        "VI_MINUS": compute_vortex(highs, lows, closes, parameters.VI_WINDOW)[1],
    })

    # === Volatility
    sma, upper, lower = compute_bollinger(prices, parameters.BOLLINGER_WINDOW, parameters.BOLLINGER_K)
    price = prices.iloc[-1] if not prices.empty else None
    if upper is not None and lower is not None and price is not None and upper != lower:
        bollinger_pctb = (price - lower) / (upper - lower)
    else:
        bollinger_pctb = None
    indicators_dict.update({
        "BOLLINGER_components": {
            "middle": float(sma),
            "upper": float(upper),
            "lower": float(lower),
        },
        "BOLLINGER": bollinger_pctb,
        "ATR": float(compute_atr(prices, parameters.ATR_WINDOW)),
        "STDDEV": float(compute_stddev(prices, parameters.STDDEV_WINDOW)),
        "SKEW": float(compute_skew(prices, parameters.SKEW_WINDOW)),
        "KURTOSIS": float(compute_kurtosis(prices, parameters.KURTOSIS_WINDOW)),
        "ZSCORE_PRICE": float(compute_zscore(prices, parameters.ZSCORE_WINDOW)),
    })

    # === Volume / Accumulation
    indicators_dict.update({
        "VWAP": compute_vwap(prices, volumes),
        "OBV": compute_obv(prices, volumes),
        "AD": compute_accumulation_distribution(prices, volumes, parameters.AD_WINDOW),
        "EOM": compute_eom(highs, lows, closes, volumes),
        "MFI": compute_mfi(highs, lows, closes, volumes, parameters.MFI_WINDOW),
        "FORCE_INDEX": compute_force_index(prices, volumes, parameters.FORCE_WINDOW),
        "VOLUME_OSC": compute_volume_oscillator(volumes, parameters.VOLUME_OSC_FAST, parameters.VOLUME_OSC_SLOW),
        "ADL": compute_adl(prices, highs, lows, closes, volumes),
        "ANCHOR_VWAP": compute_anchored_vwap(prices, volumes, anchor_index=0),  # you can adjust anchor_index logic later
        "ROLLING_VWAP": compute_rolling_vwap(prices, volumes, parameters.ROLLING_VWAP_WINDOW),
    })

    # === Cycle / Smoothing
    indicators_dict.update({
        "FISHER": compute_fisher(prices, parameters.FISHER_WINDOW),
        "SUPERTREND": compute_supertrend(highs, lows, closes, parameters.SUPERTREND_WINDOW, parameters.SUPERTREND_MULT),
        "HILBERT_CYCLE": compute_hilbert_dcp(prices),
    })

    # === Mean Reversion / Band
    indicators_dict.update({
        "KELTNER_UPPER": compute_keltner_channels(highs, lows, closes, parameters.KELTNER_EMA, parameters.KELTNER_MULT_UPPER)[0],
        "KELTNER_LOWER": compute_keltner_channels(highs, lows, closes, parameters.KELTNER_EMA, parameters.KELTNER_MULT_LOWER)[1],
        "FRACTAL_UPPER": compute_fractal_bands(prices)[0],
        "FRACTAL_LOWER": compute_fractal_bands(prices)[1],
    })

    # === Price Action / Patterns
    support, resistance = compute_support_resistance(prices, window=20, tolerance=0.001)
    val = compute_candle_body_ratio(opens, closes, highs, lows)
    print(f"[DEBUG] CANDLE_BODY_RATIO = {val}")
    indicators_dict.update({"CANDLE_BODY_RATIO": val})
    indicators_dict.update({
        "WICK_UP": compute_wick_percent(opens, closes, highs, lows)[0],
        "WICK_DOWN": compute_wick_percent(opens, closes, highs, lows)[1],
        "ENGULFING": compute_engulfing_candle(opens, closes),
        "THREE_BAR_REV": compute_three_bar_reversal(prices),
        "SUPPORT": support,
        "RESISTANCE": resistance,
    })
    indicators_dict.update({
        "DRAWDOWN": compute_max_drawdown(prices),
        "SHARPE": compute_sharpe(prices),
        "ULCER_INDEX": compute_ulcer_index(prices),
    })

    # === Orderbook / Microstructure
    indicators_dict.update({
        "FULL_BOOK_IMB": compute_full_book_imbalance(bids, asks),
        "BOOK_IMB": compute_book_imbalance(bids, asks),
        "BOOK_PRESSURE": compute_book_pressure_ratio(bids, asks),
        "PRESSURE": compute_pressure(compute_bid_density(bids), compute_ask_density(asks)),
        "BOOK_DENSITY": compute_book_density(bids, asks),
        "SPREAD": compute_spread(bids, asks),
        "TOP_BOOK_VOL": compute_top_of_book_volatility(bids, asks),
        "DELTA_FLOW": book_feature_buffers[symbol].get_delta_flow(),
        "BID_DENSITY": compute_bid_density(bids),
        "ASK_DENSITY": compute_ask_density(asks),
        "CHAIKIN_OSC": compute_chaikin_oscillator(highs, lows, closes, volumes, parameters.CHAIKIN_FAST, parameters.CHAIKIN_SLOW),
    })

    bid_slope, ask_slope = compute_book_slope(bids, asks)
    indicators_dict.update({
        "BID_SLOPE": bid_slope,
        "ASK_SLOPE": ask_slope,
        "SLOPE" : compute_slope(prices, parameters.SLOPE_WINDOW)
    })

    bid_gap, ask_gap = compute_bid_ask_gap(bids, asks)
    indicators_dict.update({
        "BID_GAP": bid_gap,
        "ASK_GAP": ask_gap,
        "LIQUIDITY_GAP" : bid_gap - ask_gap if bid_gap is not None and ask_gap is not None else None
    })

    bid_vol, ask_vol = compute_bid_ask_vol(bids, asks)
    indicators_dict.update({
        "BID_VOL": bid_vol,
        "ASK_VOL": ask_vol,
    })
    print("[DEBUG] indicators_dict keys right after BOLLINGER_components update:", indicators_dict.keys())
    return indicators_dict