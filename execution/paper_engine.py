# trading_bot/execution/paper_engine.py

"""
Final Paper Trading Execution Engine
- Uses config from parameters.py
- Supports fixed-weight simulation
- Tracks positions, equity, PnL
"""

import time
from trading_bot.config import parameters
from trading_bot.logic.signals import generate_signal
from trading_bot.logic.risk import evaluate_position
from trading_bot.config.config import SYMBOLS
# === Paper Trading State ===
paper_account = {
    'balance': parameters.STARTING_BALANCE,
    'positions': {symbol: 0 for symbol in parameters.SYMBOLS},
    'pnl': 0.0,
    'trades': []
}

def initialize_paper_account(symbols=parameters.SYMBOLS):
    paper_account['balance'] = parameters.STARTING_BALANCE
    paper_account['positions'] = {symbol: 0 for symbol in symbols}
    paper_account['pnl'] = 0.0
    paper_account['trades'] = []
    print("✅ Paper account initialized.")

def execute_paper_trade(symbol, price, decision, score):
    if decision not in ['BUY', 'SELL']:
        return

    confidence = max(0, score)
    scaling_factor = confidence ** parameters.POSITION_SCALING_POWER
    allocated_capital = scaling_factor * parameters.RISK_PER_TRADE
    allocated_capital = min(allocated_capital, paper_account['balance'])

    size = allocated_capital / price
    fee = price * size * parameters.FEES

    if decision == 'BUY':
        paper_account['balance'] -= (price * size + fee)
        paper_account['positions'][symbol] += size
    elif decision == 'SELL':
        paper_account['balance'] += (price * size - fee)
        paper_account['positions'][symbol] -= size

    log_trade(symbol, price, size, decision, fee)

def log_trade(symbol, price, size, side, fee):
    trade = {
        'timestamp': time.time(),
        'symbol': symbol,
        'price': price,
        'size': size,
        'side': side,
        'fee': fee
    }
    paper_account['trades'].append(trade)
    print(f"💰 PAPER TRADE: {side} {size:.4f} {symbol} @ {price:.2f} (Fee: {fee:.4f})")
    print(f"Balance: {paper_account['balance']:.2f} | Positions: {paper_account['positions']}")

def simulate_portfolio_with_execution(df, weights):
    initialize_paper_account()

    for _, row in df.iterrows():
        symbol = row['symbol']
        price = row['price']
        indicators = {k: row.get(k, 0) for k in weights}

        signal, _, score = generate_signal(indicators, weights)
        decision = evaluate_position(symbol, signal)

        execute_paper_trade(symbol, price, decision, score)

    return {
        symbol: paper_account['positions'][symbol] * df[df['symbol'] == symbol]['price'].iloc[-1]
        for symbol in parameters.SYMBOLS
    }

def summarize_account(latest_prices):
    summary = {
        'cash': paper_account['balance'],
        'total_equity': paper_account['balance'],
        'positions': {},
        'total_unrealized_pnl': 0.0,
        'trades': len(paper_account['trades'])
    }

    for symbol, size in paper_account['positions'].items():
        if size == 0:
            continue

        price = latest_prices.get(symbol, 0)

        buys = [t for t in paper_account['trades'] if t['symbol'] == symbol and t['side'] == 'BUY']
        total_qty = sum(t['size'] for t in buys)
        total_cost = sum(t['price'] * t['size'] for t in buys)
        avg_entry = total_cost / total_qty if total_qty else 0

        market_value = size * price
        unrealized_pnl = market_value - (size * avg_entry)

        summary['positions'][symbol] = {
            'units': size,
            'current_price': price,
            'market_value': market_value,
            'avg_entry': avg_entry,
            'unrealized_pnl': unrealized_pnl
        }

        summary['total_equity'] += market_value
        summary['total_unrealized_pnl'] += unrealized_pnl

    return summary

def get_balance():
    return paper_account['balance']

def get_positions():
    return paper_account['positions']

# trading_bot/execution/paper_engine.py

def update_total_pnl(price_store, positions):
    for symbol in SYMBOLS:
        pos = positions.get(symbol)
        if pos and pos['entry_price'] is not None:
            current_price = price_store.get(symbol)
            if current_price:
                size = pos['size']
                entry_price = pos['entry_price']
                pnl = (current_price - entry_price) * size
                account_pnl[symbol] = pnl
                print(f"[PNL] {symbol}: {pnl:.2f}")