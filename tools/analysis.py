# trading_bot/tools/analysis.py

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from trading_bot.config import parameters

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

# === Correlation Heatmap for Sub-scores ===
def plot_subscore_correlations(df):
    sub_scores = df['sub_scores'].dropna().apply(pd.Series)

    if sub_scores.empty:
        print("⚠️ No sub_scores found.")
        return

    plt.figure(figsize=(14, 10))
    corr = sub_scores.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("📈 Sub-Score Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # Correlation with total score
    if 'score' in df.columns:
        merged = pd.concat([sub_scores, df['score']], axis=1)
        corrs = merged.corr()['score'].sort_values(ascending=False)
        print("\n🔍 Correlation with final score:")
        print(corrs)

# === Session Analyzer ===
def analyze_session(session_path):
    print(f"\n📊 Analyzing session: {session_path}")
    df = load_session(session_path)

    pnl_df = compute_pnl(df)
    print(pnl_df.tail())

    # Plot PnL curve
    plt.figure(figsize=(10, 5))
    plt.plot(pnl_df['timestamp'], pnl_df['total_equity'], label='Equity')
    plt.title("Equity Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Total Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Signal Distribution
    print("\n✅ Signal Distribution:")
    print(df['decision'].value_counts())

    # Sub-score correlation matrix
    plot_subscore_correlations(df)

# === Main Entry ===
if __name__ == "__main__":
    sessions = glob.glob("logs/session_*")
    sessions.sort()

    if not sessions:
        print("❌ No sessions found.")
    else:
        for session in sessions:
            analyze_session(session)