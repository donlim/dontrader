from trading_bot.config import strategy_parameters as params

def normalize(val, scale):
    return max(min(val / scale, 1), -1)

def calculate_score(indicators, orderbook_data):
    score = 0

    # EMA Crossover (normalized)
    ema_fast = indicators['EMA10']
    ema_slow = indicators['EMA50']
    if ema_fast and ema_slow:
        diff = ema_fast - ema_slow - params.EMA_CROSS_THRESHOLD
        score += normalize(diff, params.EMA_NORM_SCALE) * params.EMA_WEIGHT

    # MACD (normalized)
    macd = indicators['MACD']
    if macd is not None:
        score += normalize(macd - params.MACD_ZERO_THRESHOLD, params.MACD_NORM_SCALE) * params.MACD_WEIGHT

    # RSI (normalized)
    rsi = indicators['RSI']
    if rsi is not None:
        rsi_signal = 50 - rsi  # overbought → negative, oversold → positive
        score += normalize(rsi_signal, params.RSI_NORM_SCALE) * params.RSI_WEIGHT

    # Momentum (normalized)
    momentum = indicators['MOMENTUM']
    if momentum is not None:
        score += normalize(momentum - params.MOMENTUM_THRESHOLD, params.MOMENTUM_NORM_SCALE) * params.MOMENTUM_WEIGHT

    # Bollinger breakout (binary)
    bollinger = indicators['BOLLINGER']
    if bollinger:
        _, upper, lower = bollinger
        price = indicators['PRICE']
        if price > upper:
            score += params.BOLLINGER_WEIGHT
        elif price < lower:
            score += params.BOLLINGER_WEIGHT

    # ATR (normalized)
    atr = indicators['ATR']
    if atr is not None:
        score += normalize(atr, params.ATR_NORM_SCALE) * params.ATR_WEIGHT

    # ✅ Orderbook imbalance (direct normalized since already [-1, 1])
    imbalance = orderbook_data['imbalance']
    score += imbalance * params.ORDERBOOK_WEIGHT

    return score