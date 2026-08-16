# trading_bot/logic/signals.py

"""
Phase 3.5.2 Pro — Complete Modular Signal Engine with Sub-Score Outputs + Trend Strength + Advanced Indicators
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from trading_bot.config import parameters
from trading_bot.config.parameters import INDICATOR_NAMES  # single source of truth
from trading_bot.logic.subscores import compute_category_subscores, compute_meta_confidence
from trading_bot.config.indicator_categories import INDICATOR_CATEGORIES
from trading_bot.logic.indicators import (
    # === Trend Indicators ===
    compute_ema, compute_sma, compute_adx, compute_parabolic_sar,
    compute_trend_strength, compute_dema, compute_tema,
    compute_vortex, compute_trix, compute_supertrend,

    # === Momentum Indicators ===
    compute_rsi, compute_macd, compute_momentum, compute_stoch_rsi,
    compute_cci, compute_roc, compute_tsi, compute_williams_r,
    compute_heikin_ashi_ratio, compute_cmo, compute_fisher,
    compute_zscore, compute_hilbert_dcp,  # placeholder

    # === Volatility Indicators ===
    compute_bollinger, compute_atr, compute_stddev,
    compute_skew, compute_kurtosis, compute_keltner_channels, compute_fractal_bands,

    # === Volume / Accumulation Indicators ===
    compute_obv, compute_vwap, compute_adl, compute_accumulation_distribution,
    compute_kvo, compute_cmf, compute_anchored_vwap,
    compute_rolling_vwap, compute_eom, compute_mfi,
    compute_force_index, compute_volume_oscillator,

    # === Support / Resistance Indicators ===
    compute_support_resistance, compute_donchian_channels,
    compute_donchian_width,

    # === Orderbook / Microstructure Indicators ===
    compute_book_imbalance, compute_full_book_imbalance,
    compute_book_density, compute_bid_ask_gap, compute_spread,
    compute_top_of_book_volatility, compute_book_slope,
    compute_book_pressure_ratio, compute_chaikin_oscillator,
    compute_bid_density, compute_ask_density, compute_bid_ask_gap, compute_bid_ask_vol,


    # === Price Action / Candlestick Indicators ===
    compute_candle_body_ratio, compute_wick_percent,
    compute_three_bar_reversal, compute_engulfing_candle,

    # === Risk / Return Metrics ===
    compute_max_drawdown, compute_sharpe, compute_ulcer_index,
    compute_avg_holding_time,
)

# =====================
# Individual Scoring Functions
# =====================
# NOTE: All signal functions normalize raw indicator values to [-1, +1] range
# using tanh() with appropriate scaling factors based on observed data ranges.
# Price-relative calculations are used where applicable to be asset-agnostic.

def to_scalar_safe(x):
    if isinstance(x, pd.Series):
        return x.iloc[-1]
    elif isinstance(x, np.ndarray):
        return x[-1]
    return x

# === Orderbook / Microstructure Signals ===

def delta_flow_signal(delta_flow):
    # Delta flow is typically small (-0.1 to 0.1), scale up
    return np.tanh(delta_flow * 50.0) if delta_flow is not None else 0

def book_imbalance_signal(book_imb):
    # Book imbalance is -1 to 1, amplify slightly
    return np.tanh(book_imb * 2.0) if book_imb is not None else 0

def book_pressure_signal(bid_density, ask_density):
    if bid_density is None or ask_density in (None, 0): return 0
    ratio = bid_density / ask_density
    return np.tanh((ratio - 1.0) * 2.0)

def slope_signal(bid_slope, ask_slope):
    if bid_slope is None or ask_slope is None: return 0
    return np.tanh((bid_slope - ask_slope) * 5.0)

def liquidity_gap_signal(min_bid_gap, min_ask_gap):
    if min_bid_gap is None or min_ask_gap is None: return 0
    return np.tanh((min_ask_gap - min_bid_gap) * 50.0)

def spread_signal(spread):
    # Spread as percentage - lower is better (tighter spread)
    if spread is None or spread <= 0: return 0
    # Invert so tighter spread = higher signal
    return np.tanh(0.001 / spread) if spread > 0 else 0

def full_book_imbalance_signal(value):
    # Full book imbalance is -1 to 1
    return np.tanh(value * 2.0) if value is not None else 0

def book_density_signal(value):
    # Book density typically 0-1000+, normalize
    return np.tanh(value / 500.0) if value is not None else 0

def top_of_book_volatility_signal(value):
    # Top of book vol is small, scale up
    return np.tanh(value * 100.0) if value is not None else 0

def book_pressure_ratio_signal(value):
    # Ratio centered at 1.0
    return np.tanh((value - 1.0) * 2.0) if value is not None else 0

def chaikin_oscillator_signal(value):
    # Chaikin oscillator - use relative scaling
    return np.tanh(value * 10.0) if value is not None else 0

def bid_gap_signal(value):
    # Bid gap as fraction, scale up
    return np.tanh(value * 100.0) if value is not None else 0

def ask_gap_signal(value):
    # Ask gap as fraction, scale up (negative = bearish)
    return -np.tanh(value * 100.0) if value is not None else 0

def bid_vol_signal(value):
    # Bid volume ratio
    return np.tanh(value * 2.0) if value is not None else 0

def ask_vol_signal(value):
    # Ask volume ratio
    return np.tanh(value * 2.0) if value is not None else 0

# === Volatility Signals ===

def volatility_signal(stddev, price=None):
    # Volatility as percentage of price if price provided
    if stddev is None or stddev <= 0: return 0
    # Higher volatility = negative signal (risk)
    return -np.tanh(stddev * 10.0)

def atr_signal_relative(atr, price):
    # ATR as percentage of price - price relative!
    if atr is None or price is None or price <= 0: return 0
    atr_pct = (atr / price) * 100  # Convert to percentage
    # ATR 0.5% = neutral, higher = more volatile
    return np.tanh((atr_pct - 0.5) * 2.0)

def atr_signal(value):
    # Fallback if no price - assume it's already normalized
    if value is None: return 0
    return np.tanh(value * 2.0)

def keltner_signal(value):
    # Keltner width as ratio
    return np.tanh(value * 2.0) if value is not None else 0

def skew_signal(value):
    # Skewness typically -3 to 3
    return np.tanh(value / 2.0) if value is not None else 0

def kurtosis_signal(value):
    # Kurtosis typically -2 to 10+
    return np.tanh(value / 5.0) if value is not None else 0

def bollinger_signal(bollinger_dict, current_price):
    if bollinger_dict is None or current_price is None:
        return 0.0
    try:
        upper = float(bollinger_dict.get("upper"))
        lower = float(bollinger_dict.get("lower"))
        current_price = float(current_price)
        if upper == lower:
            return 0.0
        # Position within bands: 0=lower, 1=upper, 0.5=middle
        normalized = (2 * (current_price - lower) / (upper - lower)) - 1
        return float(np.tanh(normalized * 1.5))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0

def bollinger_pctb_signal(bollinger_pctb):
    if bollinger_pctb is None:
        return 0.0
    try:
        # %B: 0=lower band, 1=upper band, 0.5=middle
        normalized = float(bollinger_pctb) - 0.5
        return float(np.tanh(normalized * 3.0))
    except (TypeError, ValueError):
        return 0.0

# === Trend Signals ===

def adx_signal(adx):
    # ADX 0-100, trend strength. Use smooth gradient, not binary.
    if adx is None: return 0
    # ADX > 25 = strong trend, < 20 = weak/no trend
    # Map to [-1, 1]: 20->0, 40->0.76, 60->0.95
    return np.tanh((adx - 25) / 15.0)

def trend_strength_signal(ts):
    # Trend strength typically 0-100
    if ts is None: return 0
    return np.tanh((ts - 50) / 25.0)

def ema_signal(ema_diff, price=None):
    # EMA difference - should be price-relative
    if ema_diff is None: return 0
    if price and price > 0:
        # Convert to percentage of price
        pct_diff = (ema_diff / price) * 100
        return np.tanh(pct_diff * 5.0)
    # Fallback: assume small values
    return np.tanh(ema_diff / 5.0)

def sma_signal(value, price=None):
    if value is None: return 0
    if price and price > 0:
        pct_diff = (value / price) * 100
        return np.tanh(pct_diff * 5.0)
    return np.tanh(value / 50.0)

def supertrend_signal(value, price=None):
    if value is None: return 0
    if price and price > 0:
        pct_diff = (value / price) * 100
        return np.tanh(pct_diff * 5.0)
    return np.tanh(value / 50.0)

def trix_signal(value):
    # TRIX is typically -0.5 to 0.5
    return np.tanh(value * 5.0) if value is not None else 0

def vortex_plus_signal(value):
    # Vortex VI+ typically 0.5 to 1.5, centered at 1
    return np.tanh((value - 1.0) * 3.0) if value is not None else 0

def vortex_minus_signal(value):
    # Vortex VI- typically 0.5 to 1.5, centered at 1 (invert for signal)
    return -np.tanh((value - 1.0) * 3.0) if value is not None else 0

def dema_signal(value, price=None):
    if value is None: return 0
    if price and price > 0:
        pct_diff = (value / price) * 100
        return np.tanh(pct_diff * 5.0)
    return np.tanh(value / 50.0)

def tema_signal(value, price=None):
    if value is None: return 0
    if price and price > 0:
        pct_diff = (value / price) * 100
        return np.tanh(pct_diff * 5.0)
    return np.tanh(value / 50.0)

def parabolic_sar_signal(sar, price):
    if sar is None or price is None or price <= 0:
        return 0
    # Price above SAR = bullish, below = bearish
    diff_ratio = (price - sar) / price
    return np.tanh(diff_ratio * 100.0)

# === Momentum Signals ===

def rsi_signal(rsi):
    # RSI 0-100, centered at 50
    if rsi is None: return 0
    return np.tanh((rsi - 50) / 15.0)

def macd_signal(macd, price=None):
    # MACD should be relative to price
    if macd is None: return 0
    if price and price > 0:
        pct = (macd / price) * 1000  # Scale to meaningful range
        return np.tanh(pct * 2.0)
    return np.tanh(macd / 10.0)

def momentum_signal(momentum):
    # Momentum as price change
    if momentum is None: return 0
    return np.tanh(momentum / 5.0)

def cci_signal(cci):
    # CCI typically -200 to +200
    return np.tanh(cci / 150.0) if cci is not None else 0

def roc_signal(roc):
    # ROC is percentage change, typically -5% to +5%
    if roc is None: return 0
    # ROC is already a percentage (0.01 = 1%)
    return np.tanh(roc * 100.0)  # 1% change -> tanh(1) = 0.76

def tsi_signal(tsi):
    # TSI typically -50 to +50
    return np.tanh(tsi / 30.0) if tsi is not None else 0

def stoch_rsi_signal(value):
    # Stochastic RSI 0-100
    if value is None: return 0
    return np.tanh((value - 50) / 25.0)

def williams_r_signal(value):
    # Williams %R is -100 to 0
    if value is None: return 0
    # -100 = oversold (bullish), 0 = overbought (bearish)
    return np.tanh((value + 50) / 25.0)

def fisher_signal(value):
    # Fisher transform typically -2 to +2
    return np.tanh(value / 1.5) if value is not None else 0

def cmo_signal(value):
    # Chande Momentum Oscillator -100 to +100
    return np.tanh(value / 50.0) if value is not None else 0

def heikin_ashi_ratio_signal(value):
    # HA ratio 0 to 1
    if value is None: return 0
    return np.tanh((value - 0.5) * 3.0)

def zscore_price_signal(value):
    # Z-score typically -3 to +3
    return np.tanh(value / 2.0) if value is not None else 0

def hilbert_dcp_signal(value):
    # Placeholder
    return np.tanh(value / 2.0) if value is not None else 0

# === Volume / Accumulation Signals ===

def vwap_signal(price, vwap):
    # Price relative to VWAP - percentage deviation
    if price is None or vwap is None or vwap <= 0: return 0
    pct_diff = (price - vwap) / vwap * 100
    return np.tanh(pct_diff * 2.0)

def obv_signal(obv, obv_prev=None):
    # OBV trend - use rate of change instead of absolute
    if obv is None: return 0
    if obv_prev is not None and obv_prev != 0:
        roc = (obv - obv_prev) / abs(obv_prev)
        return np.tanh(roc * 10.0)
    # Fallback: sign of OBV
    return np.tanh(obv / (abs(obv) + 1e-10)) * 0.5

def accdist_signal(ad, ad_prev=None):
    # A/D Line trend
    if ad is None: return 0
    if ad_prev is not None and ad_prev != 0:
        roc = (ad - ad_prev) / abs(ad_prev)
        return np.tanh(roc * 10.0)
    return np.tanh(ad / (abs(ad) + 1e-10)) * 0.5

def mfi_signal(value):
    # Money Flow Index 0-100
    if value is None: return 0
    return np.tanh((value - 50) / 25.0)

def eom_signal(value):
    # Ease of Movement - small values
    return np.tanh(value * 100.0) if value is not None else 0

def cmf_signal(value):
    # Chaikin Money Flow -1 to +1
    return np.tanh(value * 3.0) if value is not None else 0

def force_index_signal(value, volume_avg=None):
    # Force Index - normalize by typical volume
    if value is None: return 0
    if volume_avg and volume_avg > 0:
        normalized = value / volume_avg
        return np.tanh(normalized * 2.0)
    return np.tanh(value * 0.001)  # Fallback

def rolling_vwap_signal(price, rolling_vwap):
    if price is None or rolling_vwap is None or rolling_vwap <= 0: return 0
    pct_diff = (price - rolling_vwap) / rolling_vwap * 100
    return np.tanh(pct_diff * 2.0)

def anchored_vwap_signal(price, anchored_vwap):
    if price is None or anchored_vwap is None or anchored_vwap <= 0: return 0
    pct_diff = (price - anchored_vwap) / anchored_vwap * 100
    return np.tanh(pct_diff * 2.0)

def volume_oscillator_signal(value):
    # Volume oscillator as percentage
    return np.tanh(value / 50.0) if value is not None else 0

def adl_signal(value, prev_value=None):
    # ADL trend
    if value is None: return 0
    if prev_value is not None and prev_value != 0:
        roc = (value - prev_value) / abs(prev_value)
        return np.tanh(roc * 10.0)
    return np.tanh(value / (abs(value) + 1e-10)) * 0.5

def kvo_signal(kvo):
    # Klinger Volume Oscillator
    return np.tanh(kvo / 1000.0) if kvo is not None else 0

# === Support / Resistance Signals ===

def donchian_signal(price, upper, lower):
    if None in (price, upper, lower) or upper == lower: return 0
    # Position within channel: 0=lower, 1=upper
    position = (price - lower) / (upper - lower)
    return np.tanh((position - 0.5) * 3.0)

def donchian_width_signal(value, price=None):
    # Donchian width as percentage of price
    if value is None: return 0
    if price and price > 0:
        pct = (value / price) * 100
        return np.tanh(pct * 2.0)
    return np.tanh(value * 0.01)

def support_resistance_signal(support, resistance, price=None):
    if support is None or resistance is None: return 0
    if support == 0: return 0
    # Price position relative to S/R
    if price:
        sr_range = resistance - support
        if sr_range <= 0: return 0
        position = (price - support) / sr_range
        return np.tanh((position - 0.5) * 3.0)
    # Fallback: S/R spread
    spread_pct = (resistance - support) / support
    return np.tanh(spread_pct * 10.0)

# === Price Action / Candlestick Signals ===

def candle_body_ratio_signal(value):
    # Body ratio 0 to 1
    if value is None: return 0
    return np.tanh((value - 0.5) * 3.0)

def wick_up_signal(value):
    # Upper wick ratio 0 to 1
    return np.tanh(value * 3.0) if value is not None else 0

def wick_down_signal(value):
    # Lower wick ratio 0 to 1
    return np.tanh(value * 3.0) if value is not None else 0

def three_bar_reversal_signal(is_reversal):
    return 1.0 if is_reversal else 0.0

def engulfing_candle_signal(is_engulfing):
    return 1.0 if is_engulfing else 0.0

# === Risk / Return Signals ===

def sharpe_signal(value):
    # Sharpe ratio typically -2 to +4
    return np.tanh(value / 2.0) if value is not None else 0

def max_drawdown_signal(value):
    # Drawdown as decimal (0.1 = 10%)
    if value is None: return 0
    # Larger drawdown = more negative signal
    return -np.tanh(abs(value) * 10.0)

def ulcer_index_signal(value):
    # Ulcer index typically 0 to 0.2
    if value is None: return 0
    return -np.tanh(value * 20.0)

def average_holding_time_signal(value):
    # Holding time in minutes
    return np.tanh(value / 30.0) if value is not None else 0

# === Fractal Signals ===

def fractal_upper_signal(value, price=None):
    if value is None: return 0
    if price and price > 0:
        pct = (value / price) * 100
        return np.tanh(pct * 2.0)
    return np.tanh(value * 0.01)

def fractal_lower_signal(value, price=None):
    if value is None: return 0
    if price and price > 0:
        pct = (value / price) * 100
        return -np.tanh(pct * 2.0)
    return -np.tanh(value * 0.01)

def ad_signal(value, prev_value=None):
    # Accumulation/Distribution - same as adl_signal
    if value is None: return 0
    if prev_value is not None and prev_value != 0:
        roc = (value - prev_value) / abs(prev_value)
        return np.tanh(roc * 10.0)
    return np.tanh(value / (abs(value) + 1e-10)) * 0.5

# =====================
# Master Aggregation Logic
# =====================

def _num(x) -> float:
    """Coerce any score-like value to a finite float (fallback 0.0)."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        import numpy as _np, pandas as _pd  # local to avoid global assumptions
        if isinstance(x, _pd.Series):
            return float(x.iloc[-1]) if len(x) else 0.0
        if isinstance(x, _np.ndarray):
            return float(x[-1]) if x.size else 0.0
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def _safe_weights(w: dict | None) -> dict:
    """Restrict to known features and coerce to floats."""
    w = w or {}
    out = {}
    for k in INDICATOR_NAMES:
        try:
            out[k] = float(w.get(k, 0.0) or 0.0)
        except Exception:
            out[k] = 0.0
    return out

def generate_signal(indicators: dict, weights: dict | None = None):
    # 1) weights: prefer passed-in, else fallback to static defaults
    weights = _safe_weights(weights if weights is not None else parameters.SIGNAL_WEIGHTS)

    # Get current price for price-relative calculations
    price = indicators.get("PRICE")

    # 2) Raw scores with price-relative normalization where applicable
    raw_scores = {
        # --- Orderbook / Microstructure ---
        "DELTA_FLOW":       delta_flow_signal(indicators.get("DELTA_FLOW")),
        "BOOK_IMB":         book_imbalance_signal(indicators.get("BOOK_IMB")),
        "FULL_BOOK_IMB":    full_book_imbalance_signal(indicators.get("FULL_BOOK_IMB")),
        "BOOK_PRESSURE":    book_pressure_ratio_signal(indicators.get("BOOK_PRESSURE")),
        "PRESSURE":         book_pressure_signal(indicators.get("BID_DENSITY"), indicators.get("ASK_DENSITY")),
        "SLOPE":            slope_signal(indicators.get("BID_SLOPE"), indicators.get("ASK_SLOPE")),
        "BOOK_DENSITY":     book_density_signal(indicators.get("BOOK_DENSITY")),
        "TOP_BOOK_VOL":     top_of_book_volatility_signal(indicators.get("TOP_BOOK_VOL")),
        "LIQUIDITY_GAP":    liquidity_gap_signal(indicators.get("BID_GAP"), indicators.get("ASK_GAP")),
        "SPREAD":           spread_signal(indicators.get("SPREAD")),
        "CHAIKIN_OSC":      chaikin_oscillator_signal(indicators.get("CHAIKIN_OSC")),
        "BID_VOL":          bid_vol_signal(indicators.get("BID_VOL")),
        "ASK_VOL":          ask_vol_signal(indicators.get("ASK_VOL")),
        "BID_GAP":          bid_gap_signal(indicators.get("BID_GAP")),
        "ASK_GAP":          ask_gap_signal(indicators.get("ASK_GAP")),
        "ASK_DENSITY":      book_density_signal(indicators.get("ASK_DENSITY")),
        "BID_DENSITY":      book_density_signal(indicators.get("BID_DENSITY")),

        # --- Volatility (price-relative where applicable) ---
        "VOLATILITY":           volatility_signal(indicators.get("STDDEV"), price),
        "BOLLINGER_components": bollinger_signal(indicators.get("BOLLINGER_components"), price),
        "BOLLINGER":            bollinger_pctb_signal(indicators.get("BOLLINGER")),
        "ATR":                  atr_signal_relative(indicators.get("ATR"), price),
        "KELTNER_CHANNELS":     keltner_signal(indicators.get("KC_WIDTH")),
        "FRACTAL_UPPER":        fractal_upper_signal(indicators.get("FRACTAL_UPPER"), price),
        "FRACTAL_LOWER":        fractal_lower_signal(indicators.get("FRACTAL_LOWER"), price),
        "SKEW":                 skew_signal(indicators.get("SKEW")),
        "KURTOSIS":             kurtosis_signal(indicators.get("KURTOSIS")),

        # --- Trend (price-relative where applicable) ---
        "ADX":              adx_signal(indicators.get("ADX")),
        "TREND_STRENGTH":   trend_strength_signal(indicators.get("TREND_STRENGTH")),
        "EMA_DIFF":         ema_signal(indicators.get("EMA_DIFF"), price),
        "SMA50":            sma_signal(indicators.get("SMA50"), price),
        "SMA200":           sma_signal(indicators.get("SMA200"), price),
        "SMA_DIFF":         sma_signal(indicators.get("SMA_DIFF"), price),
        "SUPERTREND":       supertrend_signal(indicators.get("SUPERTREND"), price),
        "TRIX":             trix_signal(indicators.get("TRIX")),
        "VI_PLUS":          vortex_plus_signal(indicators.get("VI_PLUS")),
        "VI_MINUS":         vortex_minus_signal(indicators.get("VI_MINUS")),
        "DEMA":             dema_signal(indicators.get("DEMA_DIFF"), price),
        "TEMA":             tema_signal(indicators.get("TEMA_DIFF"), price),
        "EMA10":            ema_signal(indicators.get("EMA10"), price),
        "EMA50":            ema_signal(indicators.get("EMA50"), price),
        "EMA100":           ema_signal(indicators.get("EMA100"), price),
        "EMA200":           ema_signal(indicators.get("EMA200"), price),
        "SMA":              sma_signal(indicators.get("SMA"), price),
        "AD":               ad_signal(indicators.get("AD")),

        # --- Momentum ---
        "RSI":          rsi_signal(indicators.get("RSI")),
        "MACD":         macd_signal(indicators.get("MACD"), price),
        "MOMENTUM":     momentum_signal(indicators.get("MOMENTUM")),
        "CCI":          cci_signal(indicators.get("CCI")),
        "ROC":          roc_signal(indicators.get("ROC")),
        "TSI":          tsi_signal(indicators.get("TSI")),
        "STOCH_RSI":    stoch_rsi_signal(indicators.get("STOCH_RSI")),
        "WILLIAMS_R":   williams_r_signal(indicators.get("WILLIAMS_R")),
        "FISHER":       fisher_signal(indicators.get("FISHER")),
        "CMO":          cmo_signal(indicators.get("CMO")),
        "HEIKIN_RATIO": heikin_ashi_ratio_signal(indicators.get("HEIKIN_RATIO")),
        "ZSCORE_PRICE": zscore_price_signal(indicators.get("ZSCORE_PRICE")),
        "HILBERT_CYCLE": hilbert_dcp_signal(indicators.get("HILBERT_CYCLE")),

        # --- Volume / Accumulation ---
        "VWAP":          vwap_signal(price, indicators.get("VWAP")),
        "OBV":           obv_signal(indicators.get("OBV")),
        "ACC_DIST":      accdist_signal(indicators.get("AD")),
        "MFI":           mfi_signal(indicators.get("MFI")),
        "EOM":           eom_signal(indicators.get("EOM")),
        "CMF":           cmf_signal(indicators.get("CMF")),
        "FORCE_INDEX":   force_index_signal(indicators.get("FORCE_INDEX")),
        "ROLLING_VWAP":  rolling_vwap_signal(price, indicators.get("ROLLING_VWAP")),
        "ANCHOR_VWAP":   anchored_vwap_signal(price, indicators.get("ANCHOR_VWAP")),
        "VOLUME_OSC":    volume_oscillator_signal(indicators.get("VOLUME_OSC")),
        "ADL":           adl_signal(indicators.get("ADL")),
        "KVO":           kvo_signal(indicators.get("KVO")),

        # --- Support / Resistance (price-relative) ---
        "DONCHIAN":       donchian_signal(price,
                                          indicators.get("DONCHIAN_UPPER"),
                                          indicators.get("DONCHIAN_LOWER")),
        "DONCHIAN_WIDTH": donchian_width_signal(indicators.get("DONCHIAN_WIDTH"), price),
        "SUPPORT_RESISTANCE": support_resistance_signal(indicators.get("SUPPORT_LEVEL"),
                                                        indicators.get("RESISTANCE_LEVEL"), price),

        # --- Price Action ---
        "CANDLE_BODY_RATIO": candle_body_ratio_signal(indicators.get("CANDLE_BODY_RATIO")),
        "WICK_UP":           wick_up_signal(indicators.get("WICK_UP")),
        "WICK_DOWN":         wick_down_signal(indicators.get("WICK_DOWN")),
        "THREE_BAR_REV":     three_bar_reversal_signal(indicators.get("THREE_BAR_REV")),
        "ENGULFING":         engulfing_candle_signal(indicators.get("ENGULFING")),

        # --- Risk / Return ---
        "SHARPE":       sharpe_signal(indicators.get("SHARPE")),
        "DRAWDOWN":     max_drawdown_signal(indicators.get("DRAWDOWN")),
        "ULCER_INDEX":  ulcer_index_signal(indicators.get("ULCER_INDEX")),

        # --- Other ---
        "PARABOLIC_SAR": parabolic_sar_signal(indicators.get("PARABOLIC_SAR"), price),
    }

    # 3) Normalize scores with tanh after coercion to float
    normalized_scores = {k: float(np.tanh(_num(v))) for k, v in raw_scores.items()}

    # 4) Weighted blend (use only weights keys; others effectively 0)
    total_w = sum(weights.values()) or 1.0
    final_score = sum(normalized_scores.get(k, 0.0) * w for k, w in weights.items()) / total_w

    # 5) Decision using static thresholds (live trader applies hysteresis/overrides)
    mode = parameters.CURRENT_MODE
    buy_threshold  = parameters.BOT_MODES.get(mode, {}).get("buy_threshold",  parameters.SIGNAL_THRESHOLD)
    sell_threshold = parameters.BOT_MODES.get(mode, {}).get("sell_threshold", -parameters.SIGNAL_THRESHOLD)

    if final_score > buy_threshold:
        signal = "BUY"
    elif final_score < sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    # 6) Confidence & top indicators
    category_subscores = compute_category_subscores(normalized_scores)
    meta_confidence = compute_meta_confidence(category_subscores)
    top_indicators = sorted(
        normalized_scores.items(),
        key=lambda kv: abs(_num(kv[1]) * weights.get(kv[0], 0.0)),
        reverse=True
    )[:5]

    # 7) Return exactly what live_trader expects
    return signal, category_subscores, float(final_score), normalized_scores, float(meta_confidence), mode, top_indicators

def compute_smoothed_score(buffer):
    """
    Compute smoothed score from a rolling buffer.
    Uses simple mean by default. Extend with weighted mean if needed later.
    """
    if not buffer:
        return None
    return sum(buffer) / len(buffer)
