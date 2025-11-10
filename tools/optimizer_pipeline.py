# trading_bot/tools/optimizer_pipeline.py

import os
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any
from trading_bot.config import paths 
from trading_bot.tools.feature_engineering import load_sessions, build_features
from trading_bot.tools.optimizer_engine import optimize_weights
from trading_bot.tools.simulator_engine import run_simulation, simulate_portfolio_with_execution
from trading_bot.tools.validation_engine import train_test_split
from trading_bot.utils.data_utils import clean_nan
from trading_bot.config.parameters import SIGNAL_WEIGHTS  # baseline
from trading_bot.config.parameters import INDICATOR_NAMES as FEATURES  # single source of truth


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimizer → produces versioned weight artifacts + pointer"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Path to JSONL logs dir or a single .jsonl"
    )
    parser.add_argument(
        "--min_records",
        type=int,
        default=500,
        help="Minimum rows required to proceed"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="Root artifacts dir (default: ../artifacts)"
    )
    parser.add_argument(
        "--gate_min_equity_ratio",
        type=float,
        default=1.00,
        help="Require optimized_test / baseline_test ≥ this"
    )
    parser.add_argument(
        "--gate_max_drawdown",
        type=float,
        default=0.35,
        help="Require max drawdown ≤ this (fraction, e.g., 0.35 = 35%)"
    )
    parser.add_argument(
        "--opt_profile",
        type=str,
        default=None,
        choices=["aggressive", "balanced", "conservative"],
        help="Optimizer risk profile (overrides OPT_PROFILE env)"
    )

    return parser.parse_args()

def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)

def _portfolio_equity_curve(equity_curve_df: pd.DataFrame) -> pd.Series:
    if equity_curve_df is None or equity_curve_df.empty:
        return pd.Series(dtype=float)
    vc = equity_curve_df[["timestamp", "equity"]].copy()
    port = vc.groupby("timestamp")["equity"].sum().sort_index()
    return port

def _max_drawdown(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max.replace(0, np.nan)
    return float(-dd.min()) if len(dd) else 0.0

def main():
    args = parse_args()

    # --- roots (shared paths, with CLI overrides supported) ---
    LOG_DIR = os.path.abspath(args.log_dir) if args.log_dir else paths.LOG_ROOT

    if args.artifacts_dir:
        ARTIFACTS_ROOT = os.path.abspath(args.artifacts_dir)
        WEIGHTS_DIR  = os.path.join(ARTIFACTS_ROOT, "weights")
        REPORTS_DIR  = os.path.join(ARTIFACTS_ROOT, "reports")
        DATASETS_DIR = os.path.join(ARTIFACTS_ROOT, "datasets")
        _ensure_dir(WEIGHTS_DIR); _ensure_dir(REPORTS_DIR); _ensure_dir(DATASETS_DIR)
    else:
        ARTIFACTS_ROOT = paths.ARTIFACTS_ROOT
        WEIGHTS_DIR    = paths.WEIGHTS_DIR
        REPORTS_DIR    = os.path.join(paths.ARTIFACTS_ROOT, "reports")
        DATASETS_DIR   = paths.DATASETS_DIR
        _ensure_dir(REPORTS_DIR)  # weights/datasets usually created by paths.py

    TS = _ts()
    FEATURES_CSV           = os.path.join(DATASETS_DIR,  f"training_dataset_{TS}.csv")
    VERSIONED_WEIGHTS_JSON = os.path.join(WEIGHTS_DIR,   f"{TS}_best.json")
    CURRENT_POINTER        = os.path.join(WEIGHTS_DIR,   "current.json")
    VERSIONED_REPORT_JSON  = os.path.join(REPORTS_DIR,   f"{TS}_metrics.json")

    # --- load + clean raw logs ---
    print("\n🔄 Loading raw JSONL logs from:", LOG_DIR)
    df_raw = load_sessions(LOG_DIR)
    df_raw = clean_nan(df_raw)
    print(f"✅ Loaded {len(df_raw)} raw rows")
    if df_raw.empty:
        raise ValueError("❌ No valid trade logs found in directory")

    # --- flatten indicators blob when present (trade_logs.jsonl path) ---
    if "indicators" in df_raw.columns:
        try:
            inds = pd.json_normalize(df_raw["indicators"])
            inds.columns = [str(c) for c in inds.columns]
            df_raw = pd.concat([df_raw.drop(columns=["indicators"]), inds], axis=1)
        except Exception as e:
            print(f"[optimizer_pipeline] warning: failed to flatten 'indicators': {e}")

    # --- price rescue BEFORE feature build ---
    if ("price" not in df_raw.columns) or (df_raw["price"].isna().all()):
        for alt in ("PRICE", "close", "CLOSE"):
            if alt in df_raw.columns:
                df_raw["price"] = pd.to_numeric(df_raw[alt], errors="coerce")
                break
    nn_price = 0 if "price" not in df_raw.columns else int(df_raw["price"].notna().sum())
    print(f"ℹ️  Non-null price rows after flatten/fallback: {nn_price}")

    # --- feature engineering ---
    df_features = build_features(df_raw)
    print(f"✅ Features generated: {len(df_features)} rows")
    print(f"✅ Columns (sample): {list(df_features.columns)[:12]} ... (+{max(0, len(df_features.columns)-12)} more)")

    # --- reattach meta columns needed by simulator ---
    if "timestamp" in df_raw.columns:
        df_features["timestamp"] = df_raw["timestamp"].values
    if "symbol" in df_raw.columns:
        df_features["symbol"] = df_raw["symbol"].values
    if "price" in df_raw.columns:
        df_features["price"] = pd.to_numeric(df_raw["price"], errors="coerce")
    elif "price" in df_features.columns:
        df_features["price"] = pd.to_numeric(df_features["price"], errors="coerce")

    required_meta = {"timestamp", "symbol", "price"}
    missing = [c for c in required_meta if c not in df_features.columns]
    if missing:
        raise ValueError(f"Required meta columns missing after feature build: {missing}")

    # --- build keep schema: meta + active features (+ a few extras if present) ---
    META_COLS = ["timestamp", "symbol", "price"]
    candidate_features = [f for f in FEATURES if f in df_features.columns]
    EXTRA_KEEP = ["meta_confidence", "score", "score_smoothed", "decision"]
    keep_cols = [c for c in (META_COLS + candidate_features + EXTRA_KEEP) if c in df_features.columns]
    df_features = df_features[keep_cols].copy()

    # --- numeric coercion for non-meta columns ---
    for col in df_features.columns:
        if col not in META_COLS:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

    # --- null handling (never drop meta) ---
    meta_guard = set(META_COLS + ["mode", "mode_used", "reason", "top_indicators", "top_indicators_str", "category_subscores"])
    nulls = df_features.isnull().sum()
    bad_cols = nulls[nulls > 0]
    if not bad_cols.empty:
        print("\n⚠️ Null counts per column (non-zero):")
        print(bad_cols.sort_values(ascending=False).head(30))
        fully_null = [c for c, n in bad_cols.items() if n == len(df_features) and c not in meta_guard]
        if fully_null:
            print(f"\n🗑️ Dropping fully-null non-meta columns: {fully_null}")
            df_features.drop(columns=fully_null, inplace=True)
            candidate_features = [f for f in candidate_features if f in df_features.columns]
        df_features.fillna(0, inplace=True)

    # --- final sanity ---
    if any(c not in df_features.columns for c in META_COLS):
        raise ValueError("Meta columns were lost unexpectedly after cleanup.")
    if len([c for c in df_features.columns if c not in META_COLS]) == 0:
        raise ValueError("No feature columns remain after cleanup; cannot optimize.")

    # --- persist dataset ---
    df_features.to_csv(FEATURES_CSV, index=False)
    print(f"📁 Feature CSV saved to: {FEATURES_CSV}")

    # --- split, optimize, validate ---
    train_df, test_df = train_test_split(df_features, test_ratio=0.2, seed=args.seed)
    print(f"\n📊 Split: {len(train_df)} train rows | {len(test_df)} test rows")

    print("\n⚙️ Running optimizer engine...")
    best_weights = optimize_weights(train_df, profile=args.opt_profile)

    print("\n🔍 Validating optimized weights on train/test...")
    train_res = run_simulation(train_df, best_weights)
    test_res  = run_simulation(test_df,  best_weights)

    opt_curve        = _portfolio_equity_curve(test_res.equity_curve)
    opt_test_equity  = float(sum(test_res.final.values()))
    opt_mdd          = _max_drawdown(opt_curve)

    print("\n📦 Baseline (default SIGNAL_WEIGHTS) on the same test window...")
    base_final_by_sym = simulate_portfolio_with_execution(test_df, SIGNAL_WEIGHTS)
    base_test_equity  = float(sum(base_final_by_sym.values()))

    # --- artifact blob ---
    artifact = {
        "version": TS,
        "schema": "weights.v1",
        "source": "optimizer_pipeline:GA",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "features": sorted(best_weights.keys()),
        "regimes": ["default"],
        "weights_by_regime": {"default": best_weights},
        "metrics": {
            "train_equity_sum": float(sum(train_res.final.values())),
            "test_equity_sum":  opt_test_equity,
            "baseline_test_equity_sum": base_test_equity,
            "equity_ratio_vs_baseline": (opt_test_equity / base_test_equity) if base_test_equity != 0 else None,
            "max_drawdown": opt_mdd,
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
        },
    }

    _atomic_write_json(VERSIONED_WEIGHTS_JSON, artifact)
    _atomic_write_json(VERSIONED_REPORT_JSON, artifact["metrics"])
    print(f"\n🧩 Wrote versioned weights → {VERSIONED_WEIGHTS_JSON}")
    print(f"🧾 Wrote metrics report   → {VERSIONED_REPORT_JSON}")

    # --- pointer promotion with gates ---
    ratio = artifact["metrics"]["equity_ratio_vs_baseline"]
    mdd   = artifact["metrics"]["max_drawdown"]
    passes_ratio = (ratio is None) or (ratio >= args.gate_min_equity_ratio)
    passes_mdd   = (mdd is None)   or (mdd <= args.gate_max_drawdown)

    if passes_ratio and passes_mdd:
        pointer_payload = {
            "version": TS,
            "path": VERSIONED_WEIGHTS_JSON,
            "promoted_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        _atomic_write_json(CURRENT_POINTER, pointer_payload)
        print(f"\n✅ Promotion gates passed → updated pointer: {CURRENT_POINTER}")
        print(f"   ratio={None if ratio is None else round(ratio,3)} (min {args.gate_min_equity_ratio}) "
              f"| mdd={round(mdd,3)} (max {args.gate_max_drawdown})")
    else:
        print("\n🚫 Promotion gates FAILED (pointer not updated)")
        print(f"   equity_ratio_vs_baseline={ratio} (need ≥ {args.gate_min_equity_ratio})")
        print(f"   max_drawdown={mdd:.3f} (need ≤ {args.gate_max_drawdown})")

    diff = opt_test_equity - base_test_equity
    print("\n🏁 Optimization Complete")
    print(f"💰 Optimized Test Equity: ${opt_test_equity:,.2f}")
    print(f"🧠 Baseline  Test Equity: ${base_test_equity:,.2f}")
    print(f"📉 Difference:             ${diff:,.2f}")

if __name__ == "__main__":
    main()