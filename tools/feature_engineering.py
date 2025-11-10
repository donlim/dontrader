# trading_bot/tools/feature_engineering.py

import os
import json
import glob
from typing import List, Dict, Any, Set

import numpy as np
import pandas as pd
from trading_bot.config import parameters

# === Canonical feature set (what we ultimately want as columns) ===
FEATURE_SET: Set[str] = set(getattr(parameters, "INDICATOR_NAMES", [])) | set(parameters.SIGNAL_WEIGHTS.keys())

# Common price aliases that may appear in raw files
PRICE_ALIASES = ("price", "PRICE", "close", "CLOSE")

META_COLS = {
    "timestamp", "symbol", "price",
    "signal", "decision", "score", "score_smoothed",
    "meta_confidence", "mode", "mode_used", "reason",
    "top_indicators_str"
}


# -----------------------------
# 1) Load & normalize sessions
# -----------------------------
def load_sessions(path: str = "logs") -> pd.DataFrame:
    """
    Robust loader:
      - If `path` is a file: read that .jsonl
      - If `path` is a directory: recursively read **/*.jsonl
      - Accepts both trade_logs.jsonl (with 'indicators' dict) and raw_indicators.jsonl (top-level indicators)
      - Ensures:
          * 'price' is present if any PRICE_ALIASES exist
          * 'mode_used' filled from 'mode' when missing
          * 'indicators' is a dict (builds one if absent by collecting known feature keys)
      - Keeps the full objects; downstream flattening happens in build_features()
    """
    # Discover files
    if os.path.isfile(path) and path.endswith(".jsonl"):
        files = [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
    else:
        print(f"[load_sessions] Path not found or unsupported: {path}")
        return pd.DataFrame()

    if not files:
        print(f"[load_sessions] No .jsonl files found under: {path}")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    bad = 0

    for fp in files:
        try:
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        bad += 1
                        continue

                    row = dict(obj)  # keep everything

                    # Normalize/ensure price
                    if "price" not in row or row.get("price") is None:
                        for cand in PRICE_ALIASES:
                            if cand in row and row[cand] is not None:
                                row["price"] = row[cand]
                                break

                    # Normalize/ensure mode_used
                    if "mode_used" not in row and "mode" in row:
                        row["mode_used"] = row["mode"]

                    # Ensure we have an indicators dict
                    if not isinstance(row.get("indicators"), dict):
                        inds = {}
                        # collect only known features to avoid bloating
                        for feat in FEATURE_SET:
                            if feat in row:
                                inds[feat] = row[feat]
                        # sometimes PRICE is useful for derived features
                        if "PRICE" in row and "PRICE" not in inds:
                            inds["PRICE"] = row["PRICE"]
                        if inds:
                            row["indicators"] = inds

                    rows.append(row)
        except Exception as e:
            print(f"[load_sessions] Error reading {fp}: {e}")

    if bad:
        print(f"[load_sessions] Skipped {bad} malformed lines.")

    df = pd.DataFrame.from_records(rows)
    # Drop rows with no timestamp at all
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])
    return df


# --------------------------------
# 2) Flatten into feature rows
# --------------------------------
def _coerce_numeric(val):
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, list) and len(val) == 1:
        try:
            return float(val[0])
        except Exception:
            return np.nan
    if isinstance(val, str):
        low = val.lower().strip()
        if low == "true":
            return 1.0
        if low == "false":
            return 0.0
        try:
            return float(val)
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens each row into features:
      - base meta columns (timestamp, symbol, price, signal, decision, score, meta_confidence, mode, reason)
      - flattens category_subscores and sub_scores
      - flattens indicators (from 'indicators' dict when present; otherwise from known FEATURE_SET at top-level)
    """
    records: List[Dict[str, Any]] = []

    for _, row in df_raw.iterrows():
        rec: Dict[str, Any] = {}

        # Meta
        rec["timestamp"]       = row.get("timestamp")
        rec["symbol"]          = row.get("symbol")
        rec["price"]           = row.get("price")
        rec["signal"]          = row.get("signal")
        rec["decision"]        = row.get("decision")
        rec["score"]           = row.get("score")
        rec["score_smoothed"]  = row.get("score_smoothed")
        rec["meta_confidence"] = row.get("meta_confidence")
        rec["mode"]            = row.get("mode_used") or row.get("mode")
        rec["reason"]          = row.get("reason")

        # Category subscores
        cat_scores = row.get("category_subscores")
        if isinstance(cat_scores, dict):
            for k, v in cat_scores.items():
                rec[f"cat_{k}"] = _coerce_numeric(v)

        # Sub-scores
        sub_scores = row.get("sub_scores")
        if isinstance(sub_scores, dict):
            for k, v in sub_scores.items():
                rec[f"sub_{k}"] = _coerce_numeric(v)

        # Top indicators (string form for debugging)
        top_ind = row.get("top_indicators")
        if isinstance(top_ind, list):
            try:
                rec["top_indicators_str"] = ", ".join([f"{k} ({float(v):.2f})" for k, v in top_ind])
            except Exception:
                rec["top_indicators_str"] = str(top_ind)
        elif top_ind is not None:
            rec["top_indicators_str"] = str(top_ind)

        # Indicators: prefer 'indicators' dict; else pick known feature keys from top-level
        inds = row.get("indicators")
        if isinstance(inds, dict) and inds:
            for k, v in inds.items():
                rec[k] = _coerce_numeric(v)
        else:
            for feat in FEATURE_SET:
                if feat in df_raw.columns:
                    rec[feat] = _coerce_numeric(row.get(feat))

        records.append(rec)

    features = pd.DataFrame.from_records(records)

    # If price came through as object/str, coerce
    if "price" in features.columns:
        features["price"] = pd.to_numeric(features["price"], errors="coerce")

    # Ensure every SIGNAL_WEIGHTS key exists (so optimizer always sees the same columns)
    for col in parameters.SIGNAL_WEIGHTS.keys():
        if col not in features.columns:
            features[col] = np.nan

    # Coerce non-meta columns to numeric
    for col in features.columns:
        if col not in META_COLS:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    return features


# --- Manual sanity test ---
if __name__ == "__main__":
    df_raw = load_sessions("logs")
    print("rows:", len(df_raw))
    print("columns:", list(df_raw.columns)[:20], "...")
    # quick price non-null check
    if "price" in df_raw.columns:
        print("non-null price rows:", int(df_raw["price"].notna().sum()))

    df_feat = build_features(df_raw)
    print("features rows:", len(df_feat))
    print("feature columns (sample):", list(df_feat.columns)[:20], "...")
    # save a quick snapshot
    df_feat.to_csv("training_dataset_full.csv", index=False)
    print("saved → training_dataset_full.csv")