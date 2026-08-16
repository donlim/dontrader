# trading_bot/tests/test_feature_engineering.py

import pandas as pd

from trading_bot.tools.feature_engineering import build_features


def test_build_features_preserves_meta_columns():
    row = {
        "timestamp": 1_700_000_000,
        "symbol": "BTC",
        "price": 101.5,
        "signal": "HOLD",
        "decision": "HOLD",
        "score": 0.2,
        "score_smoothed": 0.15,
        "meta_confidence": 0.6,
        "mode": "default",
        "mode_used": "default",
        "category_subscores": {"orderbook": 0.5},
        "indicators": {"DELTA_FLOW": 1.0, "RSI": 55.0},
    }
    df_raw = pd.DataFrame([row])

    features = build_features(df_raw)

    assert {"timestamp", "symbol", "price"} <= set(features.columns)
    assert "DELTA_FLOW" in features.columns
    assert features["DELTA_FLOW"].notna().all()
