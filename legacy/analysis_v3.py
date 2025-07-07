# analysis_v3.py

import pandas as pd
import matplotlib.pyplot as plt
import json
from simulator_v3_portfolio import PortfolioSimulator

# Load everything
df = pd.read_csv("training_dataset_20250614_211342.csv")
df.columns = [c.strip().lower() for c in df.columns]
symbols = df['symbol'].unique().tolist()

# Load weights
with open("best_weights_v3_portfolio.json", "r") as f:
    weights_dict = json.load(f)

# Simulate & plot
final_equity = {}
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()
    sim = PortfolioSimulator([symbol])
    equity = sim.simulate(symbol_df, weights_dict[symbol])
    final_equity[symbol] = equity[symbol]
    print(f"{symbol} Final Equity: {equity[symbol]:.2f}")

# Plot portfolio breakdown
plt.bar(final_equity.keys(), final_equity.values())
plt.title("Per-Symbol Final Equity")
plt.show()