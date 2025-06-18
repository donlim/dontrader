# trading_bot/config/parameters.py

# === Classic Technical Indicators ===

EMA_WINDOWS = [10, 12, 26, 50, 100]
SMA_WINDOWS = [10, 50, 100]
RSI_WINDOW = 14
STOCH_WINDOW = 14
MOMENTUM_WINDOW = 10
BOLLINGER_WINDOW = 20
BOLLINGER_K = 2.0
ATR_WINDOW = 14
SWING_LOOKBACK = 20
SWING_TOLERANCE = 0.005  # ✅ updated tighter sensitivity
STDDEV_WINDOW = 20
SKEW_WINDOW = 20
KURTOSIS_WINDOW = 20
AD_WINDOW = 20
SUPPORT_RESISTANCE_WINDOW = 50  # ✅ updated deeper support zone
SUPPORT_RESISTANCE_TOLERANCE = 0.003  # ✅ updated stricter tolerance

# === Order Book Parameters ===

ORDERBOOK_DEPTH = 5

# === Raw Thresholds ===

DELTA_FLOW_THRESHOLD = 3.0
BOOK_IMB_THRESHOLD = 0.5
PRESSURE_RATIO_THRESHOLD = 1.5
BOOK_SLOPE_THRESHOLD = 0.05
LIQUIDITY_GAP_THRESHOLD = 0.05
SPREAD_THRESHOLD = 1.0
VOLATILITY_THRESHOLD = 1.0

# === Phase 3 Pro Signal Engine Weights ===

SIGNAL_WEIGHTS = {
    "DELTA_FLOW": 0.4,
    "BOOK_IMB": 0.3,
    "PRESSURE": 0.15,
    "SLOPE": 0.05,
    "LIQUIDITY_GAP": 0.05,
    "SPREAD": 0.025,
    "VOLATILITY": 0.025
}

# === Master Decision Threshold ===

SIGNAL_THRESHOLD = 0.25

# === Risk Parameters ===

MAX_POSITION_SIZE = 1
DAILY_MAX_TRADES = 100
MAX_LOSS_PER_DAY = 0.05

# === Portfolio Risk Limits (Phase 3.8) ===

MAX_POSITION_PCT = 0.15
MAX_TOTAL_EXPOSURE_PCT = 0.50
DAILY_LOSS_LIMIT_PCT = 0.03

# === Paper Trading Parameters (USD Allocation Mode) ===

SYMBOLS = ['BTC', 'ETH', 'HYPE']
STARTING_BALANCE = 100_000
RISK_PER_TRADE = 500  # ✅ updated lower risk per trade
FEES = 0.0005
POSITION_SCALING_POWER = 2.0

MAX_POSITION_NOTIONAL = 3000  # ✅ updated larger scaling cap

# === Exchange Parameters ===

EXCHANGE_FEE_RATE = FEES

# === Execution Cost Model (NEW V4)

SLIPPAGE_BPS = 5  # slippage in basis points
IMPACT_BPS = 10   # market impact in basis points

# === Live Price Initialization (for risk_manager live_prices state)

SYMBOL_STARTING_PRICES = {
    'BTC': 105000,
    'ETH': 2500,
    'HYPE': 40
}