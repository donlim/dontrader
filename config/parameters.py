# trading_bot/config/parameters.py
from .indicator_categories import INDICATOR_CATEGORIES
# === Classic Technical Indicators ===


EMA_WINDOWS = [5, 10, 12, 26, 50, 100, 200]
SMA_WINDOWS = [5, 10, 50, 100, 200]
RSI_WINDOW = 14
STOCH_WINDOW = 14
MOMENTUM_WINDOW = 10
BOLLINGER_WINDOW = 20
BOLLINGER_K = 2.0
ATR_WINDOW = 14
SWING_LOOKBACK = 20
SWING_TOLERANCE = 0.005
STDDEV_WINDOW = 20
SKEW_WINDOW = 20
KURTOSIS_WINDOW = 20
AD_WINDOW = 20
SUPPORT_RESISTANCE_WINDOW = 50
SUPPORT_RESISTANCE_TOLERANCE = 0.003

# === Advanced Technical Indicators ===

ADX_WINDOW = 14
CCI_WINDOW = 20
ROC_WINDOW = 12
TSI_FAST = 25
TSI_SLOW = 13
KVO_FAST = 34
KVO_SLOW = 55
WILLIAMS_R_WINDOW = 14
DONCHIAN_WINDOW = 20
PARABOLIC_SAR_STEP = 0.02
PARABOLIC_SAR_MAX_STEP = 0.2
DEMA_WINDOW = 20
TEMA_WINDOW = 20
VI_WINDOW = 14
TRIX_WINDOW = 15
SUPERTREND_WINDOW = 10
SUPERTREND_MULT = 3

# === Composite / Trend Strength Indicators ===

TREND_STRENGTH_STDDEV_WINDOW = 20
TREND_STRENGTH_WEIGHT = 1.0
TREND_STRENGTH_ADX_WEIGHT = 0.5
TREND_STRENGTH_STDDEV_WEIGHT = 0.3
TREND_STRENGTH_SLOPE_WEIGHT = 0.2

# === Order Book Parameters ===

ORDERBOOK_DEPTH = 5
DELTA_FLOW_WINDOW = 5
BOOK_IMB_LOOKBACK = 5
PRESSURE_LOOKBACK = 3
SLOPE_WINDOW = 10

# === Raw Thresholds ===

DELTA_FLOW_THRESHOLD = 3.0
BOOK_IMB_THRESHOLD = 0.5
PRESSURE_RATIO_THRESHOLD = 1.5
BOOK_SLOPE_THRESHOLD = 0.05
LIQUIDITY_GAP_THRESHOLD = 0.05
SPREAD_THRESHOLD = 1.0
VOLATILITY_THRESHOLD = 1.0

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
CCI_THRESHOLD = 100
ROC_THRESHOLD = 2.5
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80

SMA200_WINDOW = 200
SMA50_WINDOW = 50

HEIKIN_ASHI_WINDOW = 14
CMF_WINDOW = 20
DONCHIAN_WIDTH_WINDOW = 20
FISHER_WINDOW = 10
ZSCORE_WINDOW = 20
EOM_WINDOW = 14
MFI_WINDOW = 14
FORCE_WINDOW = 13
VOLUME_OSC_FAST = 12
VOLUME_OSC_SLOW = 26
ANCHOR_VWAP_WINDOW = 20
ROLLING_VWAP_WINDOW = 20
CHAIKIN_FAST = 3
CHAIKIN_SLOW = 10
CMO_WINDOW = 14  # Or whatever window size you want
KELTNER_EMA = 20
KELTNER_MULT_UPPER = 2
KELTNER_MULT_LOWER = 2

# === Modular Signal Weights (Phase 3.6+) ===

SIGNAL_WEIGHTS = {
    "DELTA_FLOW": 0.25,
    "BOOK_IMB": 0.20,
    "SLOPE": 0.05,
    "LIQUIDITY_GAP": 0.05,
    "SPREAD": 0.025,
    "VOLATILITY": 0.025,
    "RSI": 0.05,
    "MOMENTUM": 0.025,
    "BOLLINGER": 0.025,
    "BOLLINGER_components": 0.025,
    "ATR": 0.025,
    "EMA_DIFF": 0.025,
    "ADX": 0.025,
    "CCI": 0.025,
    "ROC": 0.025,
    "TSI": 0.025,
    "KVO": 0.025,
    "WILLIAMS_R": 0.025,
    "DONCHIAN": 0.025,
    "DONCHIAN_WIDTH": 0.025,
    "SUPERTREND": 0.025,
    "PARABOLIC_SAR": 0.025,
    "TREND_STRENGTH": 0.025,
    "SMA50": 0.025,
    "SMA200": 0.025,
    "SMA_DIFF": 0.025,  # only if you keep using it
    "MACD": 0.025,
    "TRIX": 0.025,
    "VI_PLUS": 0.025,
    "VI_MINUS": 0.025,
    "DEMA": 0.025,
    "TEMA": 0.025,
    "CMO": 0.025,
    "HEIKIN_RATIO": 0.025,
    "ZSCORE_PRICE": 0.025,
    "CHAIKIN_OSC": 0.025,
    "SKEW": 0.025,
    "KURTOSIS": 0.025,
    "CMF": 0.025,
    "FISHER": 0.025,
    "EOM": 0.025,
    "MFI": 0.025,
    "FORCE_INDEX": 0.025,
    "ROLLING_VWAP": 0.025,
    "ANCHOR_VWAP": 0.025,
    "VOLUME_OSC": 0.025,
    "FULL_BOOK_IMB": 0.025,
    "BOOK_DENSITY": 0.025,
    "TOP_BOOK_VOL": 0.025,
    "BOOK_PRESSURE": 0.025,
    "PRESSURE" : 0.025,
    "CANDLE_BODY_RATIO": 0.025,
    "WICK_UP": 0.025,
    "WICK_DOWN": 0.025,
    "THREE_BAR_REV": 0.025,
    "ENGULFING": 0.025,
    "DRAWDOWN": 0.025,
    "ULCER_INDEX": 0.025,
    "AVG_HOLD_TIME": 0.025,
    "HILBERT_CYCLE": 0.025,  # Placeholder or dummy value
    "STOCH_RSI": 0.025,
    "SUPPORT": 0.01,
    "RESISTANCE": 0.01,
    "AD": 0.01,
    "OBV": 0.01,
    "VWAP": 0.01,
    "SHARPE": 0.01,
    "STDDEV": 0.01,
    "ASK_DENSITY": 0.01,
    # Optional orderbook microstructure additions:
    "BID_SLOPE": 0.01,
    "ASK_SLOPE": 0.01,
}

INDICATOR_NAMES = list(SIGNAL_WEIGHTS.keys())

# === Master Decision Threshold ===

SIGNAL_THRESHOLD = 0.25

# === Indicator Blending (Optional) ===

INDICATOR_WEIGHTS = {
    "TREND": 0.5,
    "MEAN_REVERT": 0.3,
    "VOLUME": 0.2,
}

# === Risk Management ===

MAX_POSITION_SIZE = 1
DAILY_MAX_TRADES = 100
MAX_LOSS_PER_DAY = 0.05
MAX_DRAWDOWN = 0.10

# === Portfolio Risk (Phase 3.8) ===

MAX_POSITION_PCT = 0.15
MAX_TOTAL_EXPOSURE_PCT = 0.50
DAILY_LOSS_LIMIT_PCT = 0.03

# === Paper Trading Config ===

SYMBOLS = ['BTC', 'ETH', 'HYPE']
STARTING_BALANCE = 100_000
RISK_PER_TRADE = 500
FEES = 0.0005
FEE_RATE = FEES
POSITION_SCALING_POWER = 2.0
MAX_POSITION_NOTIONAL = 3000

# === Exchange Parameters ===

EXCHANGE_FEE_RATE = FEES

# === Execution Cost Model ===

SLIPPAGE_BPS = 5
IMPACT_BPS = 10
EXECUTION_LATENCY_MS = 200
ORDER_BOOK_DECAY_MS = 1000

# === Live Price Bootstrapping ===
# NOTE: These are only fallback values used before first live price is received.
# Actual prices are fetched from Hyperliquid WebSocket in real-time.
# Set to 1.0 to ensure relative calculations work correctly as placeholders.

SYMBOL_STARTING_PRICES = {
    'BTC': 1.0,  # Will be updated from Hyperliquid
    'ETH': 1.0,  # Will be updated from Hyperliquid
    'HYPE': 1.0  # Will be updated from Hyperliquid
}

FEATURE_WINDOW = 10

BOT_MODES = {
    "default": {"buy_threshold": 0.2, "sell_threshold": -0.2},
    "aggressive": {"buy_threshold": 0.1, "sell_threshold": -0.1},
    "conservative": {"buy_threshold": 0.3, "sell_threshold": -0.3},
    "mean_reversion": {"buy_threshold": 0.25, "sell_threshold": -0.25},
    "trend_following": {"buy_threshold": 0.15, "sell_threshold": -0.15},
}
CURRENT_MODE = "default"

DEFAULT_BUY_THRESHOLD = 0.2
DEFAULT_SELL_THRESHOLD = -0.2
AGGRESSIVE_BUY_THRESHOLD = 0.1
AGGRESSIVE_SELL_THRESHOLD = -0.1
CONSERVATIVE_BUY_THRESHOLD = 0.3
CONSERVATIVE_SELL_THRESHOLD = -0.3

# === Strategy Parameters ======
CONFIDENCE_THRESHOLD = 0.05
CONFIDENCE_SCALING_POWER = 2.0

# === Multi-Timeframe Config ===
TIMEFRAMES = {
    "tick": None,
    "1s": 1,
    "5s": 5,
    "15s": 15,
    "1m": 60
}

SIGNAL_SMOOTH_WINDOW = 5  # adjust based on desired smoothing strength