# trading_bot/logic/risk_manager.py

"""
3.9.9+ - Fully Synced Professional Risk Manager (Unified Paper Account State)
"""

import math
from trading_bot.config import parameters, exchange_config
from trading_bot.execution import paper_engine
from trading_bot.config.config import SYMBOLS

# === Risk Manager State ===
risk_state = {
    'daily_loss': 0.0,
    'starting_equity': parameters.STARTING_BALANCE,
    'live_prices': {symbol: parameters.SYMBOL_STARTING_PRICES.get(symbol, 1) for symbol in SYMBOLS}
}

def initialize_risk_manager():
    risk_state['daily_loss'] = 0.0
    risk_state['starting_equity'] = parameters.STARTING_BALANCE
    print("✅ Risk Manager initialized.")

# === Live Price Injection ===

def update_live_prices(price_store):
    risk_state['live_prices'] = price_store

# === Position Sizing (ATR + Confidence Scaling) ===

def compute_position_size(symbol, price, atr, score):
    min_atr = max(atr, price * 0.001)
    stop_distance = min_atr * 2
    dollar_risk = parameters.RISK_PER_TRADE

    base_size = dollar_risk / stop_distance

    confidence = max(0, score)
    scaling = confidence ** parameters.POSITION_SCALING_POWER
    adjusted_size = base_size * scaling

    allocated_capital = min(adjusted_size * price, get_current_balance())
    allocated_capital = min(allocated_capital, parameters.MAX_POSITION_NOTIONAL)

    return allocated_capital / price

# === Portfolio-Level Limits ===

def check_portfolio_limits(symbol, price, size):
    prices = get_latest_prices()
    prices[symbol] = price  # always use latest price

    total_equity = compute_total_equity(prices)

    max_position_value = total_equity * parameters.MAX_POSITION_PCT
    if price * size > max_position_value:
        size = max_position_value / price

    total_positions_value = compute_total_positions_value(prices)
    total_limit = total_equity * parameters.MAX_TOTAL_EXPOSURE_PCT
    if total_positions_value + price * size > total_limit:
        allowed_capital = total_limit - total_positions_value
        size = allowed_capital / price

    return max(size, 0)

# === Daily Loss Protection ===

def check_daily_loss_limit():
    total_equity = compute_total_equity(get_latest_prices())
    risk_state['daily_loss'] = total_equity - risk_state['starting_equity']

    max_loss = parameters.STARTING_BALANCE * parameters.DAILY_LOSS_LIMIT_PCT
    if risk_state['daily_loss'] < -max_loss:
        print("🚨 Daily loss limit hit. Trading halted.")
        return False
    return True

# === Account State Helpers (now fully unified on paper_engine) ===

def get_current_balance():
    return paper_engine.get_balance()

def get_current_positions():
    return paper_engine.get_positions()

def get_latest_prices():
    return risk_state['live_prices']

def compute_total_equity(prices):
    return get_current_balance() + compute_total_positions_value(prices)

def compute_total_positions_value(prices):
    total = 0
    positions = get_current_positions()
    for symbol, position in positions.items():
        price = prices.get(symbol, 0)
        total += position * price
    return total