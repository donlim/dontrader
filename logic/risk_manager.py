# trading_bot/logic/risk_manager.py

"""
Unified Risk Manager — Position tracking, sizing, and portfolio limits
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

# === Position State (from risk.py) ===
positions = {}

def initialize_positions(symbols):
    """Initialize position tracking for given symbols."""
    global positions
    positions = {symbol: 0 for symbol in symbols}

def evaluate_position(symbol, smoothed_score):
    """
    Evaluate whether to BUY/SELL/HOLD based on score and current position.
    Uses thresholds from parameters.BOT_MODES.
    """
    mode = parameters.CURRENT_MODE
    buy_threshold = parameters.BOT_MODES[mode]["buy_threshold"]
    sell_threshold = parameters.BOT_MODES[mode]["sell_threshold"]

    if smoothed_score > buy_threshold:
        signal = 'BUY'
    elif smoothed_score < sell_threshold:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    current = positions.get(symbol, 0)
    if signal == 'BUY' and current < 1:
        positions[symbol] = positions.get(symbol, 0) + 1
        return 'BUY'
    elif signal == 'SELL' and current > -1:
        positions[symbol] = positions.get(symbol, 0) - 1
        return 'SELL'
    else:
        return 'HOLD'

def initialize_risk_manager():
    risk_state['daily_loss'] = 0.0
    risk_state['starting_equity'] = parameters.STARTING_BALANCE
    print("✅ Risk Manager initialized.")

# === Live Price Injection ===

def update_live_prices(price_store):
    risk_state['live_prices'] = price_store

# === Position Sizing (ATR + Confidence Scaling + Equity Normalization) ===

def compute_position_size(symbol, price, atr, score, meta_confidence=None):
    """
    Industry-level position sizing:
    - Risk-based sizing with ATR stop distance
    - Confidence scaling (score)
    - Meta-confidence scaling (new)
    - Capital constraints per symbol
    """

    # === ATR stop calculation ===
    min_atr = max(atr, price * 0.001) if atr else price * 0.001
    stop_distance = min_atr * 2

    # === Current equity normalization ===
    current_equity = compute_total_equity(get_latest_prices())
    normalized_risk_per_trade = min(parameters.RISK_PER_TRADE, 0.02 * current_equity)

    # === Base position size based on risk per trade and stop distance ===
    base_size = normalized_risk_per_trade / stop_distance

    # === Score-based confidence scaling ===
    confidence = max(0, score)
    score_scaling = confidence ** parameters.POSITION_SCALING_POWER

    # === Meta-confidence scaling ===
    if meta_confidence is None:
        meta_confidence = 0.5  # fallback neutral confidence
    meta_scaling = meta_confidence ** parameters.CONFIDENCE_SCALING_POWER

    # === Adjusted position size ===
    adjusted_size = base_size * score_scaling * meta_scaling

    # === Capital constraints ===
    allocated_capital = min(adjusted_size * price, get_current_balance())

    symbol_notional_limits = {
        "BTC": 1500,
        "ETH": 1000,
        "HYPE": 500
    }
    max_notional = symbol_notional_limits.get(symbol, parameters.MAX_POSITION_NOTIONAL)
    allocated_capital = min(allocated_capital, max_notional)

    final_size = allocated_capital / price

    print(f"[{symbol}] Size calculation – ATR: {min_atr:.4f}, Confidence: {confidence:.3f}, Meta: {meta_confidence:.3f}, Final Size: {final_size:.4f}")

    return final_size

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

# === Account State Helpers (fully unified on paper_engine) ===

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

# === (NEW) Expose total equity for optimizer engines ===

def get_total_equity():
    prices = get_latest_prices()
    return compute_total_equity(prices)