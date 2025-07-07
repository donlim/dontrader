# trading_bot/tools/optimizer_pipeline.py

import os
import json
import argparse
import datetime
import numpy as np
import pandas as pd

from trading_bot.tools.feature_engineering import load_sessions, build_features
from trading_bot.tools.validation_engine import train_test_split, validate_model
from trading_bot.tools.optimizer_engine import optimize_weights
from trading_bot.tools.simulator_engine import run_simulation, simulate_portfolio_with_execution
from trading_bot.utils.data_utils import clean_nan

# === 0 – Argument Parser (CLI-Ready) ===
def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Optimizer Pipeline")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to JSONL logs")
    parser.add_argument("--min_records", type=int, default=500, help="Minimum rows required to proceed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

args = parse_args()

# === 1 – Path Setup ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
LOG_DIR = os.path.abspath(args.log_dir) if args.log_dir else os.path.join(PROJECT_DIR, "..", "logs")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
FEATURES_CSV = os.path.join(PROJECT_DIR, f"training_dataset_{TIMESTAMP}.csv")
WEIGHTS_JSON = os.path.join(PROJECT_DIR, f"best_weights_{TIMESTAMP}.json")

# === 2 – Load Raw Logs ===
print("\n🔄 Loading raw JSONL logs from:", LOG_DIR)
df_raw = load_sessions(LOG_DIR)
df_raw = clean_nan(df_raw)  # 🧼 Clean early
print(f"✅ Loaded {len(df_raw)} raw rows")

if df_raw.empty:
    raise ValueError("❌ No valid trade logs found in directory")

# === 3 – Feature Engineering ===
df_features = build_features(df_raw)
print(f"✅ Features generated: {len(df_features)} rows")
print(f"✅ Columns: {list(df_features.columns)}")

# === 3.5 – Restore required metadata columns ===
df_features["symbol"] = df_raw["symbol"].values
df_features["price"] = df_raw["price"].values
df_features["timestamp"] = df_raw["timestamp"].values

# === 4 – Null Check ===
nulls = df_features.isnull().sum()
nulls_present = nulls[nulls > 0]
if not nulls_present.empty:
    print("\n⚠️ Null counts per column:")
    print(nulls_present)

    # Drop fully-null columns
    fully_null_cols = nulls_present[nulls_present == len(df_features)].index.tolist()
    if fully_null_cols:
        print(f"\n🗑️ Dropping fully-null columns: {fully_null_cols}")
        df_features = df_features.drop(columns=fully_null_cols)

    # Fill partial nulls with 0
    df_features = df_features.fillna(0)

# === 5 – Data Sufficiency Check ===
if len(df_features) < args.min_records:
    raise ValueError(f"❌ Only {len(df_features)} records found. Minimum required: {args.min_records}")

# === 6 – Save Features to CSV ===
df_features.to_csv(FEATURES_CSV, index=False)
print(f"📁 Feature CSV saved to: {FEATURES_CSV}")

# === 7 – Train/Test Split ===
train_df, test_df = train_test_split(df_features, test_ratio=0.2, seed=args.seed)
print(f"\n📊 Split: {len(train_df)} train rows | {len(test_df)} test rows")

# === 8 – Optimize Weights ===
print("\n⚙️ Running optimizer engine...")
best_weights = optimize_weights(train_df)

# === 9 – Validate Results with run_simulation ===
print("\n🔍 Validating model with simulation engine...")
train_result = run_simulation(train_df, best_weights)
test_result = run_simulation(test_df, best_weights)

train_eq = train_result.final
test_eq = test_result.final

# === 🔟 Save Best Weights ===
with open(WEIGHTS_JSON, "w") as f:
    json.dump(best_weights, f, indent=4)

final_equity = sum(test_eq.values())

print("\n🏁 Optimization Complete")
print(f"💰 Final Test Equity: ${final_equity:,.2f}")
print(f"📁 Weights saved to: {WEIGHTS_JSON}")

# === 🔁 Real Scenario: Autotrader with default weights ===
from trading_bot.config.parameters import SIGNAL_WEIGHTS

print("\n📦 Simulating real autotrader scenario using default SIGNAL_WEIGHTS...")
real_eq_by_symbol = simulate_portfolio_with_execution(test_df, SIGNAL_WEIGHTS)
real_equity = sum(real_eq_by_symbol.values())

print(f"🧠 Real Autotrader Final Equity (No Optimization): ${real_equity:,.2f}")

# === 🔄 Comparison ===
diff = final_equity - real_equity
print(f"\n📊 Comparison:\n"
      f"Optimized Equity:   ${final_equity:,.2f}\n"
      f"Autotrader Equity: ${real_equity:,.2f}\n"
      f"📉 Difference:       ${diff:,.2f}")