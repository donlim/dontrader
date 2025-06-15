# trading_bot/tools/analysis.py

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# === Load one full session ===

def load_session(session_path):
    logs = []
    log_file = os.path.join(session_path, "trade_logs.jsonl")
    with open(log_file, "r") as f:
        for line in f:
            logs.append(json.loads(line))
    return pd.DataFrame(logs)

# === Simple PnL calculation ===

def compute_pnl(df, starting_balance=100000, fee_rate=0.0005):
    cash = starting_balance
    positions = {}

    pnl_records = []

    for _, row in df.iterrows():
        sym = row['symbol']
        price = row['price']
        side = row['decision']
        score = row['score']

        if sym not in positions:
            positions[sym] = 0.0

        # Use very simple size formula: 1000 USD per trade
        trade_usd = 1000 * (score ** 2 if score > 0 else 0.5)
        size = trade_usd / price

        fee = price * size * fee_rate

        if side == 'BUY':
            cash -= (price * size + fee)
            positions[sym] += size

        elif side == 'SELL':
            cash += (price * size - fee)
            positions[sym] -= size

        total_equity = cash + sum(positions[s] * row['price'] for s in positions)
        pnl_records.append({
            "timestamp": row['timestamp'],
            "cash": cash,
            "total_equity": total_equity
        })

    return pd.DataFrame(pnl_records)

# === Session Analyzer ===

def analyze_session(session_path):
    print(f"\n📊 Analyzing session: {session_path}")
    df = load_session(session_path)

    pnl_df = compute_pnl(df)
    print(pnl_df.tail())

    # Plot PnL curve
    plt.figure(figsize=(10,5))
    plt.plot(pnl_df['timestamp'], pnl_df['total_equity'])
    plt.title("Equity Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Total Equity")
    plt.show()

    # Signal Distribution
    print("\n✅ Signal Distribution:")
    print(df['decision'].value_counts())

# === Main Entry ===

if __name__ == "__main__":
    sessions = glob.glob("logs/session_*")
    sessions.sort()

    if not sessions:
        print("❌ No sessions found.")
    else:
        for session in sessions:
            analyze_session(session)