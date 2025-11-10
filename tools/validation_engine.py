# trading_bot/tools/validation_engine.py

import numpy as np
from trading_bot.tools.simulator_engine import run_simulation

# === 1️⃣ Train/Test Split Utility ===

def train_test_split(df, test_ratio=0.2, seed=42):
    """
    Shuffle and split the input DataFrame into training and test sets.

    Args:
        df (pd.DataFrame): Full feature DataFrame.
        test_ratio (float): Fraction of data to reserve for testing.
        seed (int): Random seed for reproducibility.ok

    Returns:
        (train_df, test_df): Tuple of training and testing DataFrames.
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

# === 2️⃣ Validation Logic ===

def _total_equity(sim_result) -> float:
    """
    Convert SimulationResult (or scalar fallback) into a float equity figure.
    """
    if sim_result is None:
        return 0.0
    if hasattr(sim_result, "final"):
        try:
            return float(sum(sim_result.final.values()))
        except Exception:
            pass
    try:
        return float(sim_result)
    except Exception:
        return 0.0

def validate_model(train_df, test_df, weights, verbose=True):
    """
    Run backtest simulation on both train and test sets using the same weights.

    Args:
        train_df (pd.DataFrame): Training feature set.
        test_df (pd.DataFrame): Testing feature set.
        weights (dict): Feature weights from optimizer.
        verbose (bool): If True, prints evaluation report.

    Returns:
        (train_equity, test_equity): Final equity for both sets.
    """
    train_res = run_simulation(train_df, weights)
    test_res = run_simulation(test_df, weights)
    train_equity = _total_equity(train_res)
    test_equity = _total_equity(test_res)

    if verbose:
        print("\n========== 📊 VALIDATION REPORT ==========")
        print(f"Train Equity: ${train_equity:,.2f}")
        print(f"Test Equity:  ${test_equity:,.2f}")

        if test_equity < 0.7 * train_equity:
            print("🚩 Overfitting warning: Test performance significantly worse.")
        elif test_equity > train_equity:
            print("🎉 Generalizing well: Test > Train (may indicate underfit or conservative training set).")
        else:
            print("✅ Reasonable generalization.")

    return train_equity, test_equity
