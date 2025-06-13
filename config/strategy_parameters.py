# EMA cross logic
EMA_WEIGHT = 1.0
EMA_CROSS_THRESHOLD = 0.1
EMA_NORM_SCALE = 0.05   # Normalization scale

# MACD logic
MACD_WEIGHT = 1.2
MACD_ZERO_THRESHOLD = 0.0
MACD_NORM_SCALE = 0.05

# RSI logic
RSI_WEIGHT = 0.8
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_NORM_SCALE = 20  # (full range ~0-100)

# Momentum logic
MOMENTUM_WEIGHT = 0.7
MOMENTUM_THRESHOLD = 0.0
MOMENTUM_NORM_SCALE = 5

# Bollinger breakout logic
BOLLINGER_WEIGHT = 1.0

# ATR (volatility filter)
ATR_WEIGHT = 0.5
ATR_NORM_SCALE = 50

# Orderbook imbalance logic
ORDERBOOK_WEIGHT = 1.3
ORDERBOOK_IMBALANCE_THRESHOLD = 0.2

# Master threshold for trade scoring
MASTER_THRESHOLD = 3.5