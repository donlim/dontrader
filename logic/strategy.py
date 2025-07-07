# trading_bot/logic/strategy.py
# NOTE: Currently unused in main pipeline.
# This module defines legacy rule-based scoring logic.
# Consider integrating as a sub-score feature in Phase 5 ML tabular models.
from trading_bot.config import strategy_loader

def calculate_score(symbol, indicators, orderbook_data):
    params = strategy_loader.get_profile(symbol)
    score = 0

    ema_fast = indicators['EMA10']
    ema_slow = indicators['EMA50']
    if ema_fast and ema_slow and (ema_fast > ema_slow + params["EMA_CROSS_THRESHOLD"]):
        score += params["EMA_WEIGHT"]

    macd = indicators['MACD']
    if macd and macd > params["MACD_ZERO_THRESHOLD"]:
        score += params["MACD_WEIGHT"]

    rsi = indicators['RSI']
    if rsi:
        if rsi < params["RSI_OVERSOLD"]:
            score += params["RSI_WEIGHT"]
        elif rsi > params["RSI_OVERBOUGHT"]:
            score -= params["RSI_WEIGHT"]

    momentum = indicators['MOMENTUM']
    if momentum and momentum > params["MOMENTUM_THRESHOLD"]:
        score += params["MOMENTUM_WEIGHT"]

    bollinger = indicators['BOLLINGER']
    if bollinger:
        _, upper, lower = bollinger
        price = indicators['PRICE']
        if price > upper or price < lower:
            score += params["BOLLINGER_WEIGHT"]

    atr = indicators['ATR']
    if atr and atr > 0:
        score += params["ATR_WEIGHT"]

    imbalance = orderbook_data['imbalance']
    if abs(imbalance) > params["ORDERBOOK_IMBALANCE_THRESHOLD"]:
        score += imbalance * params["ORDERBOOK_WEIGHT"]

    return score