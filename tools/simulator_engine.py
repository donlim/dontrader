# trading_bot/tools/simulator_engine.py

import os
import json
import pandas as pd
from collections import namedtuple
from trading_bot.config import parameters

# ✅ Structured result type
SimulationResult = namedtuple("SimulationResult", ["final", "equity_curve", "trade_pairs"])

class PortfolioSimulator:
    """
    Industry-grade portfolio simulation engine for quant strategy evaluation.
    """
    def __init__(self, symbols):
        self.symbols = symbols
        self.balances = {sym: parameters.STARTING_BALANCE for sym in symbols}
        self.positions = {sym: 0.0 for sym in symbols}
        mode_config = parameters.BOT_MODES.get(parameters.CURRENT_MODE, {})
        self.buy_threshold = mode_config.get("buy_threshold", parameters.DEFAULT_BUY_THRESHOLD)
        self.sell_threshold = mode_config.get("sell_threshold", parameters.DEFAULT_SELL_THRESHOLD)
        self.fee_rate = parameters.FEE_RATE

        # Tracking
        self.equity_curve = []
        self.trade_pairs = {sym: [] for sym in symbols}
        self.last_entry_time = {sym: None for sym in symbols}
        self.in_position = {sym: False for sym in symbols}

    def simulate(self, df, weights):
        """
        Simulate portfolio execution based on weighted signal scores.
        """
        for _, row in df.iterrows():
            symbol = row['symbol']
            price = row['price']
            timestamp = row['timestamp']
            if symbol not in self.balances:
                continue

            # Weighted signal score
            score = sum(weights.get(k, 0) * row.get(k, 0) for k in weights)

            size = 1000 / price
            fee = price * size * self.fee_rate

            # Extract meta_confidence for logging
            confidence = row.get('meta_confidence', 0.0)

            # Determine decision
            if score > self.buy_threshold:
                decision = "BUY"
                self.balances[symbol] -= (price * size + fee)
                self.positions[symbol] += size

                if not self.in_position[symbol]:
                    self.last_entry_time[symbol] = timestamp
                    self.in_position[symbol] = True

            elif score < self.sell_threshold:
                decision = "SELL"
                self.balances[symbol] += (price * size - fee)
                self.positions[symbol] -= size

                if self.in_position[symbol]:
                    entry_time = self.last_entry_time[symbol]
                    if entry_time is not None:
                        self.trade_pairs[symbol].append((entry_time, timestamp))
                    self.last_entry_time[symbol] = None
                    self.in_position[symbol] = False

            else:
                decision = "HOLD"

            # Log equity with enriched metadata
            equity = self.balances[symbol] + self.positions[symbol] * price
            self.equity_curve.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "equity": equity,
                "score": score,
                "decision": decision,
                "meta_confidence": confidence
            })

        # Final equity snapshot
        final = {}
        for sym in self.symbols:
            sym_df = df[df['symbol'] == sym]
            if not sym_df.empty:
                latest_price = sym_df.iloc[-1]['price']
                final[sym] = self.balances[sym] + self.positions[sym] * latest_price

        return SimulationResult(final, pd.DataFrame(self.equity_curve), self.trade_pairs)

# === Compatibility Wrappers ===

def run_simulation(df, weights):
    """
    Runs a full simulation and returns SimulationResult.
    """
    symbols = df['symbol'].unique().tolist()
    sim = PortfolioSimulator(symbols)
    return sim.simulate(df, weights)

def simulate_portfolio_with_execution(df, weights):
    """
    Returns only final equity per symbol for quick evaluation.
    """
    result = run_simulation(df, weights)
    return result.final

# === Entry Test ===
if __name__ == "__main__":
    # Auto-load most recent outputs
    latest_csv = "training_dataset_latest.csv"
    latest_weights = "best_weights_latest.json"

    if not os.path.exists(latest_csv) or not os.path.exists(latest_weights):
        raise FileNotFoundError("Missing latest CSV or weights. Run optimizer_pipeline first.")

    df = pd.read_csv(latest_csv)
    with open(latest_weights) as f:
        weights = json.load(f)

    # Run simulation
    result = run_simulation(df, weights)

    # Save equity curve
    result.equity_curve.to_csv("equity_curve_latest.csv", index=False)

    # Display results
    print("\n✅ Final equity per symbol:")
    for sym, eq in result.final.items():
        print(f"{sym}: ${eq:,.2f}")

    print("\n📈 Equity curve preview:")
    print(result.equity_curve.tail())