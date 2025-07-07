from trading_bot.config.parameters import INDICATOR_CATEGORIES
import pandas as pd
import numpy as np

def to_scalar_safe(x):
    """
    Ensures that input x is a scalar (float) by extracting last value if it's a Series or ndarray.
    """
    if isinstance(x, pd.Series):
        return x.iloc[-1]
    elif isinstance(x, np.ndarray):
        return x[-1]
    return x

def compute_category_subscores(signal_outputs: dict) -> dict:
    """
    Compute average score per indicator category (e.g., trend, volume, etc.)
    """
    category_scores = {}

    # Flip mapping: indicator -> category → category -> list of indicators
    category_to_indicators = {}
    for indicator, category in INDICATOR_CATEGORIES.items():
        category_to_indicators.setdefault(category, []).append(indicator)

    for category, indicators in category_to_indicators.items():
        values = [signal_outputs.get(ind, None) for ind in indicators]
        values = [to_scalar_safe(v) for v in values if v is not None]
        category_scores[category] = sum(values) / len(values) if values else None

    return category_scores

def compute_meta_confidence(category_subscores: dict) -> float:
    """
    Compute a single meta-confidence score from category sub-scores.
    Simple average of absolute values — safely handles Series/numpy inputs.
    """
    valid_scores = [to_scalar_safe(v) for v in category_subscores.values() if v is not None]
    if not valid_scores:
        return 0.0
    return min(sum(abs(v) for v in valid_scores) / len(valid_scores), 1.0)