# trading_bot/logic/risk.py

"""
Risk Layer — Position Manager v1
"""

from trading_bot.config import parameters

# Track current position state per symbol
positions = {}

def initialize_positions(symbols):
    global positions
    positions = {symbol: 0 for symbol in symbols}

def evaluate_position(symbol, smoothed_score):
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
        positions[symbol] += 1
        return 'BUY'
    elif signal == 'SELL' and current > -1:
        positions[symbol] -= 1
        return 'SELL'
    else:
        return 'HOLD'