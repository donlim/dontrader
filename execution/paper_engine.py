# trading_bot/execution/paper_engine.py

"""
Final Paper Trading Execution Engine (Fully Synced to Live Pricing + Nonlinear Sizing + Equity Accounting)
"""

import time
from trading_bot.config import parameters

# === Paper Trading Account State ===
paper_account = {
    'balance': parameters.STARTING_BALANCE,
    'positions': {symbol: 0 for symbol in parameters.SYMBOLS},
    'pnl': 0.0,
    'trades': []
}

def initialize_paper_account(symbols):
    """
    Initialize/reset paper account state.
    """
    paper_account['balance'] = parameters.STARTING_BALANCE
    paper_account['positions'] = {symbol: 0 for symbol in symbols}
    paper_account['pnl'] = 0.0
    paper_account['trades'] = []
    print("✅ Paper account initialized.")

def execute_paper_trade(symbol, price, decision, final_score):
    """
    Simulate trade execution for BUY/SELL decisions using nonlinear dynamic sizing.
    """
    if decision not in ['BUY', 'SELL']:
        return

    # Compute size based on confidence score
    confidence = max(0, final_score)
    scaling_factor = confidence ** parameters.POSITION_SCALING_POWER
    allocated_capital = scaling_factor * parameters.RISK_PER_TRADE

    # Prevent oversize allocation
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
    """
    Log trade and display updated balances.
    """
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

def update_total_pnl(latest_prices):
    """
    Recalculate unrealized PnL based on latest live prices.
    """
    total_market_value = sum(
        position * latest_prices.get(symbol, 0)
        for symbol, position in paper_account['positions'].items()
    )
    total_equity = paper_account['balance'] + total_market_value
    paper_account['pnl'] = total_equity

    print(f"[PAPER ACCOUNT] Cash: {paper_account['balance']:.2f} | Total Equity: {total_equity:.2f}")

def get_balance():
    return paper_account['balance']

def get_positions():
    return paper_account['positions']