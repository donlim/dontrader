# trading_bot/execution/execution_engine.py

"""
4.0 - Unified Execution Engine (Clean Architecture)
"""

from trading_bot.config import exchange_config
from trading_bot.execution import paper_engine
from trading_bot.execution import exchange_adapter
from trading_bot.logic import risk_manager

# === Init Phase ===

risk_manager.initialize_risk_manager()

# === Master trade interface ===

def execute_trade(symbol, price, decision, score, atr):
    """
    Main unified execution function called by asset_manager.
    Handles position sizing, risk limits, and routing to correct engine.
    """

    if decision == 'HOLD':
        print(f"[{symbol}] No trade triggered.")
        return

    # Step 1️⃣: Position sizing using risk manager logic
    size = risk_manager.compute_position_size(symbol, price, atr, score)

    # Step 2️⃣: Apply portfolio constraints (per-symbol exposure)
    size = risk_manager.check_portfolio_limits(symbol, price, size)

    if size <= 0:
        print(f"[{symbol}] Trade blocked by risk manager. No position opened.")
        return

    # Step 3️⃣: Route order based on trading mode
    trading_mode = exchange_config.TRADING_MODE

    if trading_mode == 'paper':
        paper_engine.execute_paper_trade(symbol, price, decision, score)

    elif trading_mode == 'live':
        exchange_adapter.submit_order(symbol, decision, size, price)

    else:
        raise ValueError(f"Unknown trading mode: {trading_mode}")