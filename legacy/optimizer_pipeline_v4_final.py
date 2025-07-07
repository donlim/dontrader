# trading_bot/tools/optimizer_pipeline_v4_final.py

import pandas as pd
import os
import json
import argparse
import datetime
import numpy as np

from tools.feature_engineering import load_sessions, build_features
from tools.validation_engine_v4 import train_test_split
from tools.optimizer_engine_v4 import optimize_weights
from tools.execution_model_v4 import simulate_portfolio_with_execution

# === 0️⃣ Build Argument Parser (full CLI interface) ===

def parse_args():
    parser = argparse.ArgumentParser(description="V4 Optimizer Pipeline (Final Production Version)")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to logs directory (default auto)")
    parser.add_argument("--min_records", type=int, default=500, help="Minimum records required to run optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

args = parse_args()

# === 1️⃣ Safe dynamic project paths ===

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))

# If no log_dir provided, default one level up
LOG_DIR = os.path.abspath(args.log_dir) if args.log_dir else os.path.abspath(os.path.join(PROJECT_DIR, "..", "logs"))

OUTPUT_DIR = PROJECT_DIR
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

FEATURED_CSV = os.path.join(OUTPUT_DIR, f"training_dataset_v4_full_{TIMESTAMP}.csv")
WEIGHTS_OUTPUT = os.path.join(OUTPUT_DIR, f"best_weights_v4_{TIMESTAMP}.json")

# === 2️⃣ Load JSONL Trade Logs ===

print("\n🔎 Loading raw JSONL logs...")
raw_df = load_sessions(log_dir=LOG_DIR)
print(f"✅ Raw sessions loaded: {len(raw_df)} rows")

if raw_df.empty:
    raise ValueError("❌ No JSONL records found in provided logs directory!")

print("\n📝 Sample raw record:")
print(raw_df.head(3))

# === 3️⃣ Flatten & Build Features ===

features_df = build_features(raw_df)
print(f"\n✅ Features after flattening: {len(features_df)} rows")
print(f"✅ Columns detected: {list(features_df.columns)}")

# Null checks
null_counts = features_df.isnull().sum()
nulls_present = null_counts[null_counts > 0]

if not nulls_present.empty:
    print("\n⚠️ Null counts per column:")
    print(nulls_present)
else:
    print("\n✅ No nulls detected after flattening")

# Fail fast if too little data
if len(features_df) < args.min_records:
    raise ValueError(f"❌ Only {len(features_df)} usable records. Required: {args.min_records}")

# === 4️⃣ Save flattened feature set ===

features_df.to_csv(FEATURED_CSV, index=False)
print(f"\n✅ Feature set saved: {FEATURED_CSV}")

# === 5️⃣ Train-Test Split ===

print("\n🧪 Performing train/test split...")
np.random.seed(args.seed)
train_df, test_df = train_test_split(features_df, test_ratio=0.2)

print(f"✅ Train size: {len(train_df)} rows")
print(f"✅ Test size: {len(test_df)} rows")

# === 6️⃣ Optimizer Run ===

print("\n🚀 Running optimizer engine...")
best_weights = optimize_weights(train_df)

# === 7️⃣ Backtest with execution model ===

print("\n💸 Running backtest with execution model...")
final_equity = simulate_portfolio_with_execution(test_df, best_weights)

# === 8️⃣ Save optimizer output ===

with open(WEIGHTS_OUTPUT, "w") as f:
    json.dump(best_weights, f, indent=4)

print("\n🎯 Optimization complete!")
print(f"💰 Final test equity: ${final_equity:,.2f}")
print(f"📁 Weights saved to: {WEIGHTS_OUTPUT}")