# trading_bot/config/strategy_loader.py

import json
import os

DEFAULT_PROFILE = {
    "EMA_WEIGHT": 1.0,
    "EMA_CROSS_THRESHOLD": 0.1,
    "MACD_WEIGHT": 1.2,
    "MACD_ZERO_THRESHOLD": 0.0,
    "RSI_WEIGHT": 0.8,
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "MOMENTUM_WEIGHT": 0.7,
    "MOMENTUM_THRESHOLD": 0.0,
    "BOLLINGER_WEIGHT": 1.0,
    "ATR_WEIGHT": 0.5,
    "ORDERBOOK_WEIGHT": 1.3,
    "ORDERBOOK_IMBALANCE_THRESHOLD": 0.2,
    "MASTER_THRESHOLD": 1.0
}

# Load from JSON file
path = os.path.join(os.path.dirname(__file__), 'strategy_profile.json')
with open(path, 'r') as f:
    STRATEGY_PROFILES = json.load(f)

def get_profile(symbol):
    return STRATEGY_PROFILES.get(symbol, DEFAULT_PROFILE)