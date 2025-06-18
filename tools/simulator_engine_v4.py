# trading_bot/tools/simulator_engine_v4.py

import pandas as pd
import numpy as np
import json

# === Define core features ===

FEATURES = [
    "DELTA_FLOW", "BOOK_IMB", "PRESSURE", "SLOPE", "LIQUIDITY_GAP",
    "SPREAD", "VOLATILITY", "EMA10", "EMA50", "MACD", "RSI",
    "MOMENTUM", "ATR", "VWAP", "OBV", "AD", "STDDEV", "SKEW", "KURTOSIS"
]

# === Simulator Engine ===

class PortfolioSimulator:

    def __init__(self, starting_balance=100_000, fee_rate=0.0005):
        self.starting_balance = starting_balance
        self.cash = starting_balance
        self.positions = {}
        self.fee_rate = fee_rate

    def trade(self, symbol, price, signal, position_size=1000):
        size = position_size / price
        fee = price * size * self.fee_rate

        if signal == "BUY":
            self.cash -= (price * size + fee)
            self.positions[symbol] = self.positions.get(symbol, 0) + size

        elif signal == "SELL":
            self.cash += (price * size - fee)
            self.positions[symbol] = self.positions.get(symbol, 0) - size

    def compute_equity(self, price_map):
        equity = self.cash
        for symbol, position in self.positions.items():
            equity += position * price_map[symbol]
        return equity

# === Scoring function (linear weighted sum) ===

def compute_score(row, weights):
    score = 0.0
    for feat in FEATURES:
        w = weights.get(feat, 0)
        v = row.get(feat, 0)
        score += w * v
    return score

# === Full simulation ===

def run_simulation(df, weights, threshold=0.25):
    sim = PortfolioSimulator()
    price_map = {}

    for _, row in df.iterrows():
        symbol = row['symbol']
        price = row['price']
        price_map[symbol] = price

        score = compute_score(row, weights)

        if score > threshold:
            sim.trade(symbol, price, "BUY")
        elif score < -threshold:
            sim.trade(symbol, price, "SELL")

    final_equity = sim.compute_equity(price_map)
    return final_equity

# === Simple test (safe for manual runs only) ===

if __name__ == "__main__":
    # This block will only execute when you run this file directly (not on import)
    data = pd.read_csv("training_dataset_v4_full.csv")
    weights = {feat: 0 for feat in FEATURES}
    equity = run_simulation(data, weights)

    print("\n✅ Simulation complete")
    print(f"Final Equity: {equity:.2f}")