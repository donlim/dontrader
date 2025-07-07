# trading_bot/tools/optimizer_pipeline_v4_debug.py

import pandas as pd
import os
import json

from tools.feature_engineering import load_sessions, build_features
from tools.validation_engine_v4 import train_test_split
from tools.optimizer_engine_v4 import optimize_weights
from tools.execution_model_v4 import simulate_portfolio_with_execution

# === 🔧 Safe Dynamic Paths ===

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))

# ✅ Fix the log directory to correctly point one level higher
LOG_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "logs"))

FEATURED_CSV = os.path.join(PROJECT_DIR, "training_dataset_v4_full.csv")
WEIGHTS_OUTPUT = os.path.join(PROJECT_DIR, "best_weights_v4.json")

# === 1️⃣ Load JSONL Trade Logs ===

print("\n🔎 Loading raw JSONL logs...")
raw_df = load_sessions(log_dir=LOG_DIR)
print(f"✅ Raw sessions loaded: {len(raw_df)} rows")

# Optional: sanity check first few rows
print(raw_df.head())

# === 2️⃣ Flatten & Build Features ===

features_df = build_features(raw_df)
print(f"✅ Features after flattening: {len(features_df)} rows")
print(f"✅ Columns detected: {list(features_df.columns)}")

# Additional debug: count nulls per column
null_counts = features_df.isnull().sum()
print("\n🔎 Null counts per column:")
print(null_counts[null_counts > 0])

# === 3️⃣ Fail fast if too little data ===

if len(features_df) < 500:
    raise ValueError("❌ Not enough usable feature records after flattening. Collect more live data.")

# === 4️⃣ Save engineered features ===

features_df.to_csv(FEATURED_CSV, index=False)
print(f"✅ Feature set saved: {FEATURED_CSV}")

# === 5️⃣ Validation Split ===

print("\n🧪 Splitting train/test sets...")
train_df, test_df = train_test_split(features_df, test_ratio=0.2)

# === 6️⃣ Optimizer Run ===

print("\n🚀 Starting optimization engine...")
best_weights = optimize_weights(train_df)

# === 7️⃣ Backtest with Execution Cost Model ===

print("\n💸 Running execution model backtest...")
final_equity = simulate_portfolio_with_execution(test_df, best_weights)

# === 8️⃣ Save Results ===

with open(WEIGHTS_OUTPUT, "w") as f:
    json.dump(best_weights, f, indent=4)

print("\n✅ Optimization complete!")
print(f"💰 Final test equity: ${final_equity:,.2f}")
print(f"📁 Weights saved to: {WEIGHTS_OUTPUT}")