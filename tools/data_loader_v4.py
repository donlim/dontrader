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
    Flatten sub_scores + indicators into one row.
    """
    data = []
    for _, row in df.iterrows():
        entry = {
            "symbol": row['symbol'],
            "price": row['price'],
            "decision": row['decision'],
            "score": row['score'],
        }

        # Flatten sub_scores
        if 'sub_scores' in row and isinstance(row['sub_scores'], dict):
            entry.update(row['sub_scores'])

        # Flatten indicators (✅ Remove unnecessary prefix)
        if 'indicators' in row and isinstance(row['indicators'], dict):
            for k, v in row['indicators'].items():
                entry[k] = v   # ✅ directly use k without prefix

        data.append(entry)

    return pd.DataFrame(data)

if __name__ == "__main__":
    raw_df = load_sessions()
    features_df = flatten_sessions(raw_df)

    print(f"\n✅ Loaded {len(features_df)} records")
    print(features_df.head())

    # Save output for optimizer v4 work
    features_df.to_csv("training_dataset_v4.csv", index=False)
    print("\n✅ Saved: training_dataset_v4.csv")