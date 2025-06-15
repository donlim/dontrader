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

def evaluate_position(symbol, signal):
    """
    Basic position management:
    - Only 1 unit per position.
    - Avoids flipping rapidly.
    """
    current = positions.get(symbol, 0)

    if signal == 'BUY' and current < 1:
        positions[symbol] += 1
        return 'BUY'
    elif signal == 'SELL' and current > -1:
        positions[symbol] -= 1
        return 'SELL'
    else:
        return 'HOLD'
    
def get_position(symbol):
    return positions.get(symbol, 0)