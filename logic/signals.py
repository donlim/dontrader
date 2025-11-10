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
def to_scalar_safe(x):
    if isinstance(x, pd.Series):
        return x.iloc[-1]
    elif isinstance(x, np.ndarray):
        return x[-1]
    return x
    
def delta_flow_signal(delta_flow):
    return np.tanh(delta_flow / 10.0) if delta_flow is not None else 0

def book_imbalance_signal(book_imb):
    return np.tanh(book_imb * 3.0) if book_imb is not None else 0

def book_pressure_signal(bid_density, ask_density):
    if bid_density is None or ask_density in (None, 0): return 0
    return np.tanh((bid_density / ask_density - 1.0) * 2.0)

def slope_signal(bid_slope, ask_slope):
    if bid_slope is None or ask_slope is None: return 0
    return np.tanh((bid_slope - ask_slope) / 10.0)

def liquidity_gap_signal(min_bid_gap, min_ask_gap):
    if min_bid_gap is None or min_ask_gap is None: return 0
    return np.tanh((min_ask_gap - min_bid_gap) * 10.0)

def spread_signal(spread):
    return np.tanh(1.0 / spread) if spread and spread > 0 else 0

def volatility_signal(stddev):
    return -np.tanh(stddev / 5.0) if stddev and stddev > 0 else 0

def adx_signal(adx):
    if adx is None:
        return 0
    return 1.0 if adx > 25 else -1.0

def cci_signal(cci):
    return np.tanh(cci / 100.0) if cci is not None else 0

def roc_signal(roc):
    return np.tanh(roc / 5.0) if roc is not None else 0

def trend_strength_signal(ts):
    return np.tanh(ts / 100.0) if ts is not None else 0

def tsi_signal(tsi):
    return np.tanh(tsi / 50.0) if tsi is not None else 0

def kvo_signal(kvo):
    return np.tanh(kvo / 5000.0) if kvo is not None else 0

def rsi_signal(rsi):
    return np.tanh((rsi - 50) / 10.0) if rsi is not None else 0

def macd_signal(macd):
    return np.tanh(macd / 5.0) if macd is not None else 0

def ema_signal(ema_diff):
    return np.tanh(ema_diff / 10.0) if ema_diff is not None else 0

def momentum_signal(momentum):
    return np.tanh(momentum / 10.0) if momentum is not None else 0

def vwap_signal(price, vwap):
    if price is None or vwap is None: return 0
    return np.tanh((price - vwap) / price * 5.0)

def obv_signal(obv):
    return np.tanh(obv / 1000000.0) if obv is not None else 0

def accdist_signal(ad):
    return np.tanh(ad / 1000000.0) if ad is not None else 0

def donchian_signal(price, upper, lower):
    if None in (price, upper, lower) or upper == lower: return 0
    return np.tanh(((price - lower) / (upper - lower) - 0.5) * 5.0)

def parabolic_sar_signal(sar, price):
    if sar is None or price is None:
        return 0
    diff_ratio = (price - sar) / price
    return np.tanh(diff_ratio * 50.0)  # amplifies subtle differences

def sma_signal(value):
    # Compare short vs long SMA if available, else use deviation
    return np.tanh(value / 100.0) if value is not None else 0

def supertrend_signal(value):
    # Use supertrend direction or distance from price
    return np.tanh(value / 100.0) if value is not None else 0

def trix_signal(value):
    # TRIX is a momentum oscillator of EMA triples
    return np.tanh(value / 100.0) if value is not None else 0

def vortex_plus_signal(value):
    # Vortex VI+ component
    return np.tanh(value / 100.0) if value is not None else 0

def vortex_minus_signal(value):
    # Vortex VI- component
    return np.tanh(value / 100.0) if value is not None else 0

def dema_signal(value):
    # Double EMA difference vs price
    return np.tanh(value / 100.0) if value is not None else 0

def tema_signal(value):
    # Triple EMA difference vs price
    return np.tanh(value / 100.0) if value is not None else 0

def stoch_rsi_signal(value):
    # Stochastic RSI normalized between 0–1
    return np.tanh(value / 100.0) if value is not None else 0

def williams_r_signal(value):
    # Williams %R, inverted and scaled from -100 to 0
    return np.tanh(value / 100.0) if value is not None else 0

def fisher_signal(value):
    # Fisher Transform typically normalized
    return np.tanh(value / 100.0) if value is not None else 0

def cmo_signal(value):
    # Chande Momentum Oscillator
    return np.tanh(value / 100.0) if value is not None else 0

def heikin_ashi_ratio_signal(value):
    # Ratio of HA candle body vs total candle range
    return np.tanh(value / 100.0) if value is not None else 0

def zscore_price_signal(value):
    # Z-score of price from mean
    return np.tanh(value / 100.0) if value is not None else 0

def hilbert_dcp_signal(value):
    # Placeholder: Hilbert Dominant Cycle Phase
    return np.tanh(value / 100.0) if value is not None else 0

def bollinger_signal(bollinger_dict, current_price):
    if bollinger_dict is None or current_price is None:
        return 0.0

    try:
        upper = float(bollinger_dict.get("upper"))
        lower = float(bollinger_dict.get("lower"))
        current_price = float(current_price)

        if upper == lower:
            return 0.0

        normalized = (2 * (current_price - lower) / (upper - lower)) - 1
        return float(np.tanh(normalized * 2.0))

    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0

def bollinger_pctb_signal(bollinger_pctb):
    if bollinger_pctb is None:
        return 0.0
    try:
        normalized = float(bollinger_pctb) - 0.5  # center at mid band
        return float(np.tanh(normalized * 4.0))  # scale to emphasize deviation
    except (TypeError, ValueError):
        return 0.0

def atr_signal(value):
    # Average True Range scaled
    return np.tanh(value / 100.0) if value is not None else 0

def keltner_signal(value):
    # Keltner Channel width or distance from center
    return np.tanh(value / 2.0) if value is not None else 0

def skew_signal(value):
    # Skewness: positive = right tail
    return np.tanh(value / 5.0) if value is not None else 0

def kurtosis_signal(value):
    # Kurtosis: tail heaviness
    return np.tanh(value / 10.0) if value is not None else 0

def mfi_signal(value):
    # Money Flow Index
    return np.tanh((value - 50) / 25.0) if value is not None else 0

def eom_signal(value):
    # Ease of Movement
    return np.tanh(value / 1_000_000.0) if value is not None else 0

def cmf_signal(value):
    # Chaikin Money Flow
    return np.tanh(value / 0.1) if value is not None else 0

def force_index_signal(value):
    # Force Index = price change * volume
    return np.tanh(value / 1_000_000.0) if value is not None else 0

def rolling_vwap_signal(price, rolling_vwap):
    if price is None or rolling_vwap is None: return 0
    return np.tanh((price - rolling_vwap) / price * 5.0)

def anchored_vwap_signal(price, anchored_vwap):
    if price is None or anchored_vwap is None: return 0
    return np.tanh((price - anchored_vwap) / price * 5.0)

def volume_oscillator_signal(value):
    # Volume Oscillator (fast - slow volume)
    return np.tanh(value / 1_000_000.0) if value is not None else 0

def adl_signal(value):
    # Accumulation Distribution Line
    return np.tanh(value / 1_000_000.0) if value is not None else 0

def donchian_width_signal(value):
    # Width between Donchian upper/lower bands
    return np.tanh(value / 100.0) if value is not None else 0

def support_resistance_signal(support, resistance):
    if support is None or resistance is None:
        return 0
    return np.tanh((resistance - support) / support)  # or divide by 100.0 for smoother scaling

def full_book_imbalance_signal(value):
    return np.tanh(value * 3.0) if value is not None else 0

def chaikin_oscillator_signal(value):
    return np.tanh(value / 1_000_000.0) if value is not None else 0

def book_density_signal(value):
    return np.tanh(value / 100.0) if value is not None else 0

def top_of_book_volatility_signal(value):
    return np.tanh(value / 500.0) if value is not None else 0

def book_pressure_ratio_signal(value):
    return np.tanh((value - 1.0) * 3.0) if value is not None else 0

def candle_body_ratio_signal(value):
    return np.tanh(value * 5.0) if value is not None else 0

def wick_up_signal(value):
    return np.tanh(value * 5.0) if value is not None else 0

def wick_down_signal(value):
    return np.tanh(value * 5.0) if value is not None else 0

def three_bar_reversal_signal(is_reversal):
    return 1.0 if is_reversal else 0.0

def engulfing_candle_signal(is_engulfing):
    return 1.0 if is_engulfing else 0.0

def sharpe_signal(value):
    return np.tanh(value / 2.0) if value is not None else 0

def max_drawdown_signal(value):
    return -np.tanh(value / 0.1) if value is not None else 0  # penalize deeper drawdowns

def ulcer_index_signal(value):
    return -np.tanh(value / 0.1) if value is not None else 0  # penalize higher ulcer index

def average_holding_time_signal(value):
    return np.tanh(value / 60.0) if value is not None else 0  # longer avg hold = higher score

def fractal_upper_signal(value):
    return np.tanh(value / 100.0) if value is not None else 0

def fractal_lower_signal(value):
    return -np.tanh(value / 100.0) if value is not None else 0

def bid_gap_signal(value):
    return np.tanh(value * 10.0) if value is not None else 0

def ask_gap_signal(value):
    return -np.tanh(value * 10.0) if value is not None else 0

def bid_vol_signal(value):
    return np.tanh(value / 100.0) if value is not None else 0

def ask_vol_signal(value):
    return np.tanh(value / 100.0) if value is not None else 0

def ad_signal(value):
    # Accumulation/Distribution index
    return np.tanh(value / 1_000_000.0) if value is not None else 0

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

    # 2) Raw scores (keep your calls as-is; just wrap with _num later)
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

        # --- Volatility ---
        "VOLATILITY":           volatility_signal(indicators.get("STDDEV")),
        "BOLLINGER_components": bollinger_signal(indicators.get("BOLLINGER_components"), indicators.get("PRICE")),
        "BOLLINGER":            bollinger_pctb_signal(indicators.get("BOLLINGER")),
        "ATR":                  atr_signal(indicators.get("ATR")),
        "KELTNER_CHANNELS":     keltner_signal(indicators.get("KC_WIDTH")),
        "FRACTAL_UPPER":        fractal_upper_signal(indicators.get("FRACTAL_UPPER")),
        "FRACTAL_LOWER":        fractal_lower_signal(indicators.get("FRACTAL_LOWER")),
        "SKEW":                 skew_signal(indicators.get("SKEW")),
        "KURTOSIS":             kurtosis_signal(indicators.get("KURTOSIS")),

        # --- Trend ---
        "ADX":              adx_signal(indicators.get("ADX")),
        "TREND_STRENGTH":   trend_strength_signal(indicators.get("TREND_STRENGTH")),
        "EMA_DIFF":         ema_signal(indicators.get("EMA_DIFF")),
        "SMA50":            sma_signal(indicators.get("SMA50")),
        "SMA200":           sma_signal(indicators.get("SMA200")),
        "SMA_DIFF":         sma_signal(indicators.get("SMA_DIFF")),
        "SUPERTREND":       supertrend_signal(indicators.get("SUPERTREND")),
        "TRIX":             trix_signal(indicators.get("TRIX")),
        "VI_PLUS":          vortex_plus_signal(indicators.get("VI_PLUS")),
        "VI_MINUS":         vortex_minus_signal(indicators.get("VI_MINUS")),
        "DEMA":             dema_signal(indicators.get("DEMA_DIFF")),
        "TEMA":             tema_signal(indicators.get("TEMA_DIFF")),
        "EMA10":            ema_signal(indicators.get("EMA10")),
        "EMA50":            ema_signal(indicators.get("EMA50")),
        "EMA100":           ema_signal(indicators.get("EMA100")),
        "EMA200":           ema_signal(indicators.get("EMA200")),
        "SMA":              sma_signal(indicators.get("SMA")),
        "AD":               ad_signal(indicators.get("AD")),

        # --- Momentum ---
        "RSI":          rsi_signal(indicators.get("RSI")),
        "MACD":         macd_signal(indicators.get("MACD")),
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
        "VWAP":          vwap_signal(indicators.get("PRICE"), indicators.get("VWAP")),
        "OBV":           obv_signal(indicators.get("OBV")),
        "ACC_DIST":      accdist_signal(indicators.get("AD")),
        "MFI":           mfi_signal(indicators.get("MFI")),
        "EOM":           eom_signal(indicators.get("EOM")),
        "CMF":           cmf_signal(indicators.get("CMF")),
        "FORCE_INDEX":   force_index_signal(indicators.get("FORCE_INDEX")),
        "ROLLING_VWAP":  rolling_vwap_signal(indicators.get("PRICE"), indicators.get("ROLLING_VWAP")),
        "ANCHOR_VWAP":   anchored_vwap_signal(indicators.get("PRICE"), indicators.get("ANCHOR_VWAP")),
        "VOLUME_OSC":    volume_oscillator_signal(indicators.get("VOLUME_OSC")),
        "ADL":           adl_signal(indicators.get("ADL")),

        # --- Support / Resistance ---
        "DONCHIAN":       donchian_signal(indicators.get("PRICE"),
                                          indicators.get("DONCHIAN_UPPER"),
                                          indicators.get("DONCHIAN_LOWER")),
        "DONCHIAN_WIDTH": donchian_width_signal(indicators.get("DONCHIAN_WIDTH")),
        "SUPPORT_RESISTANCE": support_resistance_signal(indicators.get("SUPPORT_LEVEL"),
                                                        indicators.get("RESISTANCE_LEVEL")),

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
        "PARABOLIC_SAR": parabolic_sar_signal(indicators.get("PARABOLIC_SAR"),
                                              indicators.get("PRICE")),
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
