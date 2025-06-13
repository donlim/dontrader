# trading_bot/state/orderbook_state.py

orderbook_state = {}

def update_orderbook(symbol, imbalance):
    orderbook_state[symbol] = imbalance

def get_orderbook(symbol):
    return orderbook_state.get(symbol, 0.0)
