# trading_bot/logic/trade_executor.py

"""
3.5 – Confidence Filtering + Scaling Hybrid Executor
This module acts as a middleware between signal generation and execution_engine.
"""

from trading_bot.config import parameters
from trading_bot.execution import execution_engine

def process_trade_decision(symbol, price, decision, score, atr, meta_confidence):
    """
    Filters trades below confidence threshold and scales position size based on meta_confidence.
    
    Args:
        symbol (str): Trading symbol, e.g. 'BTC'
        price (float): Current market price
        decision (str): BUY / SELL / HOLD decision from signal logic
        score (float): Final aggregated signal score
        atr (float): Average True Range value (for volatility-aware sizing)
        meta_confidence (float): Confidence value from signal normalization
    
    Returns:
        None
    """

    # === 🛑 1. Filter trades below confidence threshold ===
    if meta_confidence < parameters.CONFIDENCE_THRESHOLD:
        print(f"[{symbol}] Confidence {meta_confidence:.3f} below threshold {parameters.CONFIDENCE_THRESHOLD}. Trade skipped.")
        return

    # === ⚖️ 2. Compute size multiplier using scaling power ===
    size_multiplier = meta_confidence ** parameters.CONFIDENCE_SCALING_POWER

    print(f"[{symbol}] Executing with confidence {meta_confidence:.3f} -> size multiplier {size_multiplier:.3f}")

    # === 🚀 3. Call execute_trade with size_multiplier ===
    execution_engine.execute_trade(
        symbol=symbol,
        price=price,
        decision=decision,
        score=score,
        atr=atr,
        meta_confidence=meta_confidence,  # ✅ pass through directly
        size_multiplier=size_multiplier
    )