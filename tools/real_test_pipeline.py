import sys
import os
import json
import argparse
from glob import glob

# === Enable absolute import of 'trading_bot' ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trading_bot.execution import paper_engine
from trading_bot.execution.paper_engine import summarize_account
from trading_bot.config import parameters  # ✅ Now you can use parameters.SYMBOLS
from trading_bot.config.parameters import SIGNAL_WEIGHTS
from trading_bot.tools.feature_engineering import load_sessions, build_features

def parse_args():
    parser = argparse.ArgumentParser(description="🎯 Real Test Pipeline with Full Breakdown")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to JSONL log folder (e.g., logs/session_20250623_121912)")
    return parser.parse_args()

args = parse_args()
LOG_DIR = os.path.abspath(args.log_dir or os.path.join("..", "logs"))

# === Locate the .jsonl file ===
jsonl_files = sorted(glob(os.path.join(LOG_DIR, "*.jsonl")))
if not jsonl_files:
    raise ValueError(f"❌ No .jsonl files found in {LOG_DIR}")

jsonl_path = jsonl_files[0]
print(f"\n📂 Loading session logs from: {jsonl_path}")

# === Load and build features ===
df_raw = load_sessions(jsonl_path)
if df_raw.empty:
    raise ValueError("❌ No logs found!")

df_features = build_features(df_raw)
df_features = df_features[df_features['symbol'].isin(parameters.SYMBOLS)]
print(f"✅ Loaded {len(df_features)} rows with indicators")

# === Run simulation with fixed weights ===
print("\n🧠 Simulating fixed-weight run with SIGNAL_WEIGHTS...")
final_equity = paper_engine.simulate_portfolio_with_execution(df_features, SIGNAL_WEIGHTS)
print(f"📈 Final Equity by Symbol: {final_equity}")

# === Use most recent prices
latest_prices = {
    row['symbol']: row['price']
    for _, row in df_raw.groupby('symbol').tail(1).iterrows()
}

# === Show portfolio breakdown
summary = summarize_account(latest_prices)

print("\n📊 Portfolio Breakdown:")
print(f"- Cash: ${summary['cash']:,.2f}")
print(f"- Total Equity: ${summary['total_equity']:,.2f}")
print(f"- Total Unrealized PnL: ${summary['total_unrealized_pnl']:,.2f}")
print(f"- Total Trades: {summary['trades']}\n")

for symbol, info in summary['positions'].items():
    print(f"🔹 {symbol}")
    print(f"  • Units Held: {info['units']:.6f}")
    print(f"  • Entry Price: ${info['avg_entry']:.2f}")
    print(f"  • Current Price: ${info['current_price']:.2f}")
    print(f"  • Market Value: ${info['market_value']:.2f}")
    print(f"  • Unrealized PnL: ${info['unrealized_pnl']:.2f}\n")