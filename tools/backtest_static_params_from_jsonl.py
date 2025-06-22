# tools/backtest_static_params_from_jsonl.py

import json
import os
from trading_bot.config import parameters

# === Paths ===
LOG_DIR = "logs"
LATEST_SESSION = sorted(os.listdir(LOG_DIR))[-1]  # assumes session_YYYYMMDD_HHMMSS format
LOG_PATH = os.path.join(LOG_DIR, LATEST_SESSION, "trade_logs.jsonl")

# === Params ===
STARTING_BALANCE = parameters.STARTING_BALANCE
THRESHOLD = parameters.SIGNAL_THRESHOLD
FEE_RATE = parameters.EXCHANGE_FEE_RATE

balances = {symbol: STARTING_BALANCE / len(parameters.SYMBOLS) for symbol in parameters.SYMBOLS}
positions = {symbol: 0 for symbol in parameters.SYMBOLS}
latest_prices = {symbol: None for symbol in parameters.SYMBOLS}


def simulate_trade(price, side, notional):
    slippage = parameters.SLIPPAGE_BPS / 10000.0
    impact = parameters.IMPACT_BPS / 10000.0
    if side == "BUY":
        executed_price = price * (1 + slippage + impact)
    else:
        executed_price = price * (1 - slippage - impact)

    qty = notional / executed_price
    fees = qty * executed_price * FEE_RATE
    return qty, fees, executed_price


# === Main Loop: Simulate Trades ===
with open(LOG_PATH, "r") as f:
    for line in f:
        row = json.loads(line)
        symbol = row["symbol"]
        price = row["price"]
        score = row["score"]
        decision = row["decision"]

        # Track most recent price per symbol
        latest_prices[symbol] = price

        if decision == "BUY" and score > THRESHOLD:
            notional = min(parameters.RISK_PER_TRADE, balances[symbol])
            qty, fees, executed_price = simulate_trade(price, "BUY", notional)
            balances[symbol] -= (qty * executed_price + fees)
            positions[symbol] += qty

        elif decision == "SELL" and score < -THRESHOLD:
            qty, fees, executed_price = simulate_trade(price, "SELL", positions[symbol] * price)
            balances[symbol] += (qty * executed_price - fees)
            positions[symbol] -= qty

# === Final Equity Calculation ===
final_equity = sum(
    balances[symbol] + positions[symbol] * latest_prices[symbol]
    for symbol in parameters.SYMBOLS
    if latest_prices[symbol] is not None
)

print(f"\n📈 Final simulated equity using current parameters.py: ${final_equity:,.2f}")