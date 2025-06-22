import numpy as np
import random
from config import parameters

# === Trade Gating Logic Based on Indicator Filters (V4) ===

def is_trade_valid(row):
    try:
        if row.get("KURTOSIS", 1.0) < getattr(parameters, "KURTOSIS_THRESHOLD", 0.0):
            return False
        if row.get("STDDEV", 1.0) < getattr(parameters, "STDDEV_THRESHOLD", 0.0):
            return False
        if row.get("SPREAD", 0.0) > getattr(parameters, "SPREAD_MAX", 1.0):
            return False
        if row.get("VOLATILITY", 0.0) > getattr(parameters, "VOLATILITY_MAX", 1.0):
            return False
        return True
    except Exception:
        return True  # fallback to safe trade if anything is missing

# === Core Fill Simulation ===

def simulate_trade_execution(price, side, notional):
    """
    Simulates realistic execution fill:
        - Applies random slippage (from parameters)
        - Applies market impact (from parameters)
        - Applies exchange fee (from parameters)
    """
    slippage_pct = parameters.SLIPPAGE_BPS / 10000.0
    impact_pct = parameters.IMPACT_BPS / 10000.0
    fee_rate = parameters.EXCHANGE_FEE_RATE

    if side == "BUY":
        executed_price = price * (1 + np.random.uniform(0, slippage_pct) + impact_pct)
    elif side == "SELL":
        executed_price = price * (1 - np.random.uniform(0, slippage_pct) - impact_pct)
    else:
        return 0, 0

    quantity = notional / executed_price
    fees = executed_price * quantity * fee_rate

    return quantity, fees

# === Portfolio Simulator with Execution Model ===

def simulate_portfolio_with_execution(df, weights):
    """
    Full backtest portfolio simulation with execution costs + gating logic.
    """
    starting_balance = parameters.STARTING_BALANCE
    threshold = parameters.SIGNAL_THRESHOLD
    fee_rate = parameters.EXCHANGE_FEE_RATE

    symbols = df['symbol'].unique()
    balances = {symbol: starting_balance / len(symbols) for symbol in symbols}
    positions = {symbol: 0 for symbol in symbols}

    for _, row in df.iterrows():
        symbol = row['symbol']
        price = row['price']

        # Score calculation (same as optimizer scoring function)
        score = sum(weights.get(k, 0) * row.get(k, 0) for k in weights)

        # === Trade only if score is strong and indicator filter passes ===
        if score > threshold and is_trade_valid(row):
            notional = min(parameters.RISK_PER_TRADE, balances[symbol])
            qty, fees = simulate_trade_execution(price, "BUY", notional)
            balances[symbol] -= (qty * price + fees)
            positions[symbol] += qty

        elif score < -threshold and is_trade_valid(row):
            position_notional = positions[symbol] * price
            qty, fees = simulate_trade_execution(price, "SELL", position_notional)
            balances[symbol] += (qty * price - fees)
            positions[symbol] -= qty

    # Final equity calculation
    final_equity = sum(
        balances[symbol] + positions[symbol] * df[df['symbol'] == symbol].iloc[-1]['price']
        for symbol in balances
    )
    return final_equity

def apply_execution_costs(df, weights):
    return simulate_portfolio_with_execution(df, weights)