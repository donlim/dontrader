# trading_bot/execution/paper_engine.py

"""
Final Paper Trading Execution Engine
- Uses config from parameters.py
- Supports fixed-weight simulation
- Tracks positions, equity, PnL
"""

from __future__ import annotations

import time

from trading_bot.config import parameters
from trading_bot.logic.signals import generate_signal

# === Paper Trading State ===
paper_account = {
    'balance': parameters.STARTING_BALANCE,
    'positions': {symbol: 0.0 for symbol in parameters.SYMBOLS},
    'pnl': 0.0,
    'trades': []
}

def initialize_paper_account(symbols=parameters.SYMBOLS):
    paper_account['balance'] = parameters.STARTING_BALANCE
    paper_account['positions'] = {symbol: 0.0 for symbol in symbols}
    paper_account['pnl'] = 0.0
    paper_account['trades'] = []
    print("✅ Paper account initialized.")

def _infer_position_size(score: float, price: float) -> float:
    if price <= 0:
        return 0.0
    confidence = max(0.0, float(score))
    scaling_factor = confidence ** parameters.POSITION_SCALING_POWER
    allocated_capital = scaling_factor * parameters.RISK_PER_TRADE
    allocated_capital = min(allocated_capital, paper_account['balance'])
    return allocated_capital / price if price > 0 else 0.0

def execute_paper_trade(symbol, price, decision, score, *, size: float | None = None):
    if decision not in ['BUY', 'SELL'] or price is None or price <= 0:
        return

    paper_account['positions'].setdefault(symbol, 0.0)

    trade_size = float(size) if size is not None else _infer_position_size(score, price)
    if trade_size <= 0:
        return

    if decision == 'BUY':
        max_affordable = paper_account['balance'] / (price * (1 + parameters.FEES))
        trade_size = min(trade_size, max_affordable)
        if trade_size <= 0:
            return
        fee = price * trade_size * parameters.FEES
        paper_account['balance'] -= (price * trade_size + fee)
        paper_account['positions'][symbol] += trade_size
    else:  # SELL
        held = paper_account['positions'].get(symbol, 0.0)
        trade_size = min(trade_size, held)
        if trade_size <= 0:
            return
        fee = price * trade_size * parameters.FEES
        paper_account['balance'] += (price * trade_size - fee)
        paper_account['positions'][symbol] -= trade_size

    log_trade(symbol, price, trade_size, decision, fee)

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
    # Lazy import to avoid circular dependency
    from trading_bot.logic.risk_manager import evaluate_position

    initialize_paper_account()

    for _, row in df.iterrows():
        symbol = row['symbol']
        price = row['price']
        indicators = {k: row.get(k, 0) for k in weights}

        (
            signal,
            _category_subscores,
            final_score,
            _normalized_scores,
            _meta_confidence,
            _mode_used,
            _top_inds,
        ) = generate_signal(indicators, weights)
        decision = evaluate_position(symbol, final_score)

        execute_paper_trade(symbol, price, decision, final_score)

    results = {}
    for symbol in parameters.SYMBOLS:
        symbol_rows = df[df['symbol'] == symbol]
        if symbol_rows.empty:
            continue
        last_price = symbol_rows['price'].iloc[-1]
        size = paper_account['positions'].get(symbol, 0.0)
        results[symbol] = size * last_price
    return results

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

def update_total_pnl(price_store, positions=None):
    """
    Lightweight mark-to-market summary for live loops.
    """
    positions = positions or paper_account['positions']
    if positions is None:
        return {}

    per_symbol = {}
    total_equity = paper_account['balance']
    for symbol, qty in positions.items():
        current_price = price_store.get(symbol)
        if current_price is None:
            continue
        mtm = qty * current_price
        per_symbol[symbol] = mtm
        total_equity += mtm

    print(f"[PNL] cash={paper_account['balance']:.2f} | equity={total_equity:.2f}")
    return {'cash': paper_account['balance'], 'per_symbol': per_symbol, 'equity': total_equity}
