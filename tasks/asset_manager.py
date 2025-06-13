# trading_bot/tasks/asset_manager.py

import asyncio
from trading_bot.api.websocket import handle_websocket, orderbooks
from trading_bot.config.config import SYMBOLS
from trading_bot.state.buffers import get_buffer
from trading_bot.config import parameters, strategy_loader
from trading_bot.logic import indicators, timeframes, strategy

positions = {symbol: None for symbol in SYMBOLS}

for symbol in SYMBOLS:
    for window in parameters.EMA_WINDOWS:
        indicators.update_ema(symbol, window, None)

async def debug_loop():
    while True:
        await asyncio.sleep(5)
        for symbol in SYMBOLS:
            buffer = get_buffer(symbol)
            prices = [p['price'] for p in buffer]

            if prices:
                price = prices[-1]
                for window in parameters.EMA_WINDOWS:
                    indicators.update_ema(symbol, window, price)

                ema = {f"EMA{w}": indicators.get_ema(symbol, w) for w in parameters.EMA_WINDOWS}
                sma = {f"SMA{w}": indicators.compute_sma(prices[-w:]) for w in parameters.SMA_WINDOWS}
                rsi = indicators.compute_rsi(prices, parameters.RSI_WINDOW)
                mom = indicators.compute_momentum(prices, parameters.MOMENTUM_WINDOW)
                macd = indicators.compute_macd(symbol)
                bb = indicators.compute_bollinger(prices, parameters.BOLLINGER_WINDOW, parameters.BOLLINGER_K)
                atr = indicators.compute_atr(prices, parameters.ATR_WINDOW)

                indicator_pack = {
                    'PRICE': price, 'EMA10': ema.get("EMA10"), 'EMA50': ema.get("EMA50"),
                    'MACD': macd, 'RSI': rsi, 'MOMENTUM': mom, 'BOLLINGER': bb, 'ATR': atr
                }
                orderbook_pack = { 'imbalance': orderbooks[symbol].get_imbalance() }
                score = strategy.calculate_score(symbol, indicator_pack, orderbook_pack)

                params = strategy_loader.get_profile(symbol)
                threshold = params["MASTER_THRESHOLD"]

                if positions[symbol] is None:
                    if score > threshold:
                        print(f"[{symbol}] ENTER LONG at {price:.2f} (Score: {score:.2f} > Threshold {threshold})")
                        positions[symbol] = ("LONG", price)
                    elif score < -threshold:
                        print(f"[{symbol}] ENTER SHORT at {price:.2f} (Score: {score:.2f} < Threshold -{threshold})")
                        positions[symbol] = ("SHORT", price)
                    else:
                        print(f"[{symbol}] HOLD — No entry (Score: {score:.2f} inside neutral zone ±{threshold})")
                else:
                    side, entry_price = positions[symbol]
                    pnl = (price - entry_price) if side == "LONG" else (entry_price - price)
                    if (side == "LONG" and score < 0) or (side == "SHORT" and score > 0):
                        print(f"[{symbol}] EXIT {side} at {price:.2f} | PNL: {pnl:.2f} (Score reverted)")
                        positions[symbol] = None
                    else:
                        print(f"[{symbol}] {side} | Price: {price:.2f} | PNL: {pnl:.2f} | Holding (Score: {score:.2f})")

async def launch_all():
    await asyncio.gather(
        handle_websocket(),
        debug_loop()
    )