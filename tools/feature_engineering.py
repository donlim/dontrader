import os
import json
import glob
import pandas as pd
import numpy as np
from trading_bot.config import parameters

# === 1️⃣ Load all sessions from JSONL logs ===
def load_sessions(log_path="logs"):
    records = []

    if os.path.isdir(log_path):
        session_folders = glob.glob(os.path.join(log_path, "session_*"))
        if session_folders:
            for session_path in session_folders:
                log_file = os.path.join(session_path, "trade_logs.jsonl")
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        for line in f:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
        else:
            log_file = os.path.join(log_path, "trade_logs.jsonl")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    for line in f:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    elif log_path.endswith(".jsonl") and os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return pd.DataFrame(records)


# === 2️⃣ Flatten one record into a row ===
def flatten_record(row):
    base = {
        "timestamp": row.get("timestamp"),
        "symbol": row.get("symbol"),
        "price": row.get("price"),
        "signal": row.get("signal"),
        "decision": row.get("decision"),
        "score": row.get("score"),
        "meta_confidence": row.get("meta_confidence"),
        "mode": row.get("mode_used"),
        "reason": row.get("reason"),
    }

    # Category subscores
    cat_scores = row.get("category_subscores", {})
    if isinstance(cat_scores, dict):
        for k, v in cat_scores.items():
            base[f"cat_{k}"] = v

    # Sub-scores
    sub_scores = row.get("sub_scores", {})
    if isinstance(sub_scores, dict):
        for k, v in sub_scores.items():
            base[f"sub_{k}"] = v

    # Top indicators string
    top_ind = row.get("top_indicators", [])
    if isinstance(top_ind, list):
        try:
            base["top_indicators_str"] = ", ".join([f"{k} ({v:.2f})" for k, v in top_ind])
        except:
            base["top_indicators_str"] = str(top_ind)
    else:
        base["top_indicators_str"] = str(top_ind)

    # Raw indicators
    indicators = row.get("indicators", {})
    if not isinstance(indicators, dict):
        return None

    for key, val in indicators.items():
        # Standard numerical types
        if isinstance(val, (int, float, np.integer, np.floating)):
            base[key] = float(val)
        elif isinstance(val, list) and len(val) == 1:
            base[key] = val[0]
        elif isinstance(val, str):
            val_lower = val.lower().strip()
            if val_lower == "true":
                base[key] = 1
            elif val_lower == "false":
                base[key] = 0
            else:
                try:
                    base[key] = float(val)
                except:
                    base[key] = np.nan
        else:
            try:
                base[key] = float(val)
            except:
                base[key] = np.nan

    return base


# === 3️⃣ Build full feature dataframe ===
def build_features(df):
    records = []
    for _, row in df.iterrows():
        record = flatten_record(row)
        if record:
            records.append(record)

    features = pd.DataFrame(records)

    # Ensure all signal weight keys are present
    for col in parameters.SIGNAL_WEIGHTS.keys():
        if col not in features.columns:
            features[col] = np.nan

    return features


# === 🧪 Manual test ===
if __name__ == "__main__":
    df = load_sessions()
    features = build_features(df)

    print(f"\n✅ Loaded {len(features)} usable records after flattening")
    print(features.head())

    features.to_csv("training_dataset_full.csv", index=False)
    print("\n✅ Saved: training_dataset_full.csv")