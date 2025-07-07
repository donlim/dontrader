# trading_bot/tools/validation_engine_v4.py

import numpy as np
from tools.simulator_engine_v4 import run_simulation

# === Train/Test Split Utility ===

def train_test_split(df, test_ratio=0.2, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle once for randomness
    train_size = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df

# === Validation Logic ===

def validate_model(train_df, test_df, best_weights):
    train_equity, _ = run_simulation(train_df, best_weights, execution_costs=True)
    test_equity, _ = run_simulation(test_df, best_weights, execution_costs=True)

    print("\n========== VALIDATION RESULT ==========")
    print(f"Train Equity: {train_equity:,.2f}")
    print(f"Test Equity:  {test_equity:,.2f}")

    # Simple overfit check
    if test_equity < train_equity * 0.7:
        print("\n🚩 Potential Overfitting Warning: Test set underperforming heavily!")
    else:
        print("\n✅ Model generalizes reasonably well.")

    return train_equity, test_equity