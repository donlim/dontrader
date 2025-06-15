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
SWING_TOLERANCE = 0.003
STDDEV_WINDOW = 20
SKEW_WINDOW = 20
KURTOSIS_WINDOW = 20
AD_WINDOW = 20
SUPPORT_RESISTANCE_WINDOW = 20
SUPPORT_RESISTANCE_TOLERANCE = 0.001

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

MAX_POSITION_SIZE = 1        # max coin units per symbol (safety cap)
DAILY_MAX_TRADES = 100
MAX_LOSS_PER_DAY = 0.05

# === Portfolio Risk Limits (Phase 3.8) ===

MAX_POSITION_PCT = 0.15         # Max 15% of equity per symbol
MAX_TOTAL_EXPOSURE_PCT = 0.50   # Max 50% total allocation
DAILY_LOSS_LIMIT_PCT = 0.03     # Max 3% daily drawdown

# === Paper Trading Parameters (USD Allocation Mode) ===

SYMBOLS = ['BTC', 'ETH', 'HYPE']
STARTING_BALANCE = 100_000
RISK_PER_TRADE = 1000
FEES = 0.0005
POSITION_SCALING_POWER = 2.0

MAX_POSITION_NOTIONAL = 1500  # or whatever exact USD size you want

# === Exchange Parameters ===

EXCHANGE_FEE_RATE = FEES

# === 🔥 Live Price Initialization (for risk_manager live_prices state) ===

SYMBOL_STARTING_PRICES = {
    'BTC': 105000,
    'ETH': 2500,
    'HYPE': 40
}