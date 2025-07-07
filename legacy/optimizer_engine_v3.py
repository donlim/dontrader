# optimizer_engine_v3.py

import pandas as pd
import itertools
import json
from simulator_v3_portfolio import PortfolioSimulator

# Load dataset
df = pd.read_csv("training_dataset_20250614_211342.csv")
df.columns = [c.strip().lower() for c in df.columns]
symbols = df['symbol'].unique().tolist()

# Features + grid
features = ['delta_flow', 'book_imb', 'pressure', 'slope', 'liquidity_gap', 'spread', 'volatility']
weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Per-symbol optimizer
def optimize_symbol(symbol_df, symbol):
    total = len(weight_grid) ** len(features)
    print(f"\n🔎 Optimizing {symbol} — {total} combinations")

    best_equity = -1e9
    best_weights = None

    for idx, combo in enumerate(itertools.product(weight_grid, repeat=len(features))):
        weights = dict(zip(features, combo))
        sim = PortfolioSimulator([symbol])
        final = sim.simulate(symbol_df, weights)
        equity = final[symbol]

        if equity > best_equity:
            best_equity = equity
            best_weights = weights

        if idx % 5000 == 0:
            print(f"Checked {idx}/{total} | Best Equity: {best_equity:.2f}")

    print(f"✅ Done optimizing {symbol} | Final Best Equity: {best_equity:.2f}")
    print(json.dumps(best_weights, indent=4))
    return symbol, best_weights

# Full loop across all symbols
all_results = {}

for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()
    _, best_weights = optimize_symbol(symbol_df, symbol)
    all_results[symbol] = best_weights

# Save results
with open("best_weights_v3_portfolio.json", "w") as f:
    json.dump(all_results, f, indent=4)
print("\n✅ Saved optimizer results.")