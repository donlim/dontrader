# === Dynamic Overrides ===
SIGNAL_THRESHOLD = 0.6
KURTOSIS_THRESHOLD = 0.25
STDDEV_THRESHOLD = 0.2
SPREAD_MAX = 0.3
VOLATILITY_MAX = 0.4

# === Fallback to static for all others ===
try:
    from trading_bot.config import parameters as static_params
    _static_keys = [k for k in dir(static_params) if not k.startswith("__")]
    for k in _static_keys:
        if k not in globals():
            globals()[k] = getattr(static_params, k)
except ImportError:
    pass