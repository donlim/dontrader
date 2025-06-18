# trading_bot/tools/data_loader.py

import os
import json
import glob
import pandas as pd

def load_sessions(log_dir="logs"):
    """
    Load all JSONL sessions from logs directory.
    """
    sessions = glob.glob(os.path.join(log_dir, "session_*"))
    records = []

    for session_path in sessions:
        log_file = os.path.join(session_path, "trade_logs.jsonl")
        if not os.path.exists(log_file):
            continue

        with open(log_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue

    return pd.DataFrame(records)

def flatten_sessions(df):
    """
    Flatten the sub_scores into top-level columns.
    """
    data = []
    for _, row in df.iterrows():
        entry = {
            "symbol": row['symbol'],
            "price": row['price'],
            "decision": row['decision'],
            "score": row['score'],
        }
        entry.update(row['sub_scores'])
        data.append(entry)

    return pd.DataFrame(data)

if __name__ == "__main__":
    raw_df = load_sessions()
    features_df = flatten_sessions(raw_df)

    print(f"\n✅ Loaded {len(features_df)} records")
    print(features_df.head())

    # Save output for V4 optimizer
    features_df.to_csv("training_dataset_v4.csv", index=False)
    print("\n✅ Saved: training_dataset_v4.csv")