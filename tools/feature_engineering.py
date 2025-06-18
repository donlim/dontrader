import os
import json
import glob
import pandas as pd
import numpy as np

# Load all sessions from JSONL logs
def load_sessions(log_dir="logs"):
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

# Flatten the nested indicators safely
def flatten_record(row):
    base = {
        "symbol": row.get('symbol'),
        "price": row.get('price'),
        "decision": row.get('decision'),
        "score": row.get('score'),
    }

    # Flatten sub_scores (if present)
    sub_scores = row.get("sub_scores", {})
    if isinstance(sub_scores, dict):
        base.update(sub_scores)

    indicators = row.get("indicators", {})

    # ✅ If indicators missing entirely, skip this row
    if not indicators or not isinstance(indicators, dict):
        return None

    # Always extract these core features
    expected_keys = ['EMA10', 'EMA50', 'MACD', 'RSI', 'ATR']
    for key in expected_keys:
        val = indicators.get(key, np.nan)
        base[key] = val

    # Flatten remaining indicators
    for key, val in indicators.items():
        if key in expected_keys:
            continue
        if isinstance(val, (list, tuple)):
            if key == "BOLLINGER" and len(val) == 3:
                base["BOLLINGER_MID"] = val[0]
                base["BOLLINGER_UPPER"] = val[1]
                base["BOLLINGER_LOWER"] = val[2]
        else:
            base[key] = val

    return base

# Build full feature dataframe
def build_features(df):
    records = []
    for _, row in df.iterrows():
        record = flatten_record(row)
        if record:
            records.append(record)

    features = pd.DataFrame(records)

    # Always ensure core columns exist
    min_required_fields = ['EMA10', 'EMA50', 'MACD', 'RSI', 'ATR']
    for col in min_required_fields:
        if col not in features.columns:
            features[col] = np.nan

    # ✅ DO NOT DROP rows here — allow missing values to stay
    return features

# Entry point for manual test
if __name__ == "__main__":
    df = load_sessions()
    features = build_features(df)

    print(f"\n✅ Loaded {len(features)} usable records after flattening (no filtering applied)")
    print(features.head())

    features.to_csv("training_dataset_v4_full.csv", index=False)
    print("\n✅ Saved: training_dataset_v4_full.csv")