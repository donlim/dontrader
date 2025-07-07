# trading_bot/config/indicator_categories.py
CATEGORY_WEIGHTS = {
    "orderbook": 1.5,
    "trend": 1.2,
    "momentum": 1.0,
    "volatility": 1.0,
    "volume": 1.0,
    "support_resistance": 0.9,
    "price_action": 0.6,
    "risk": 1.2
}


INDICATOR_CATEGORIES = {
    # === Trend Indicators (🔵) ===
    "EMA10": "trend", "EMA50": "trend", "EMA100": "trend", "EMA200": "trend",
    "EMA_DIFF": "trend", "SMA50": "trend", "SMA200": "trend",
    "ADX": "trend", "PARABOLIC_SAR": "trend", "TREND_STRENGTH": "trend",
    "DEMA": "trend", "TEMA": "trend", "VI_PLUS": "trend", "VI_MINUS": "trend",
    "TRIX": "trend", "SUPERTREND": "trend",

    # === Momentum Indicators (🟠) ===
    "RSI": "momentum", "MACD": "momentum", "MOMENTUM": "momentum",
    "STOCH_RSI": "momentum", "CCI": "momentum", "ROC": "momentum", "TSI": "momentum",
    "WILLIAMS_R": "momentum", "HEIKIN_RATIO": "momentum", "CMO": "momentum",
    "FISHER": "momentum", "ZSCORE": "momentum", "HILBERT_CYCLE": "momentum",

    # === Volatility Indicators (🟡) ===
    "BOLLINGER": "volatility", "ATR": "volatility", "STDDEV": "volatility",
    "SKEW": "volatility", "KURTOSIS": "volatility",
    "KELTNER_UPPER": "volatility", "KELTNER_LOWER": "volatility",
    "FRACTAL_UPPER": "volatility", "FRACTAL_LOWER": "volatility",

    # === Volume / Accumulation Indicators (🟢) ===
    "OBV": "volume", "VWAP": "volume", "ADL": "volume",
    "ACCUM_DIST": "volume", "KVO": "volume", "CMF": "volume",
    "ANCHOR_VWAP": "volume", "ROLLING_VWAP": "volume",
    "EOM": "volume", "MFI": "volume", "FORCE_INDEX": "volume",
    "VOLUME_OSC": "volume", "AD": "volume",

    # === Support / Resistance (🟫) ===
    "SUPPORT": "support", "RESISTANCE": "support",
    "DONCHIAN_UPPER": "support", "DONCHIAN_LOWER": "support",
    "DONCHIAN_WIDTH": "support", "DONCHIAN": "support",

    # === Orderbook / Microstructure (🔴) ===
    "BOOK_IMB": "orderbook", "FULL_BOOK_IMB": "orderbook",
    "BID_DENSITY": "orderbook", "ASK_DENSITY": "orderbook",
    "BID_GAP": "orderbook", "ASK_GAP": "orderbook", "SPREAD": "orderbook",
    "BID_VOL": "orderbook", "ASK_VOL": "orderbook",
    "DELTA_FLOW": "orderbook", "BOOK_SLOPE_BID": "orderbook", "BOOK_SLOPE_ASK": "orderbook",
    "BOOK_PRESSURE": "orderbook", "CHAIKIN_OSC": "orderbook", "TOP_BOOK_VOL": "orderbook",

    # === Price Action / Candlestick (🟣) ===
    "CANDLE_RATIO": "price_action", "WICK_UP": "price_action",
    "WICK_DOWN": "price_action", "THREE_BAR_REV": "price_action",
    "ENGULFING": "price_action",

    # === Risk / Return Metrics (⚫) ===
    "DRAWDOWN": "risk", "SHARPE": "risk",
    "ULCER": "risk", "AVG_HOLD_TIME": "risk"
}

# Optional list of category names
CATEGORIES = list(set(INDICATOR_CATEGORIES.values()))