import asyncio
from trading_bot.api.websocket import handle_websocket, orderbooks
from trading_bot.config.config import SYMBOLS
from trading_bot.state.buffers import get_buffer
from trading_bot.config import parameters
from trading_bot.logic import indicators, strategy

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

                # ✅ NEW 3.2 indicators:
                stddev = indicators.compute_stddev(prices, parameters.STDDEV_WINDOW)
                skew_ = indicators.compute_skew(prices, parameters.SKEW_WINDOW)
                kurt_ = indicators.compute_kurtosis(prices, parameters.KURTOSIS_WINDOW)

                indicator_pack = {
                    'PRICE': price, 'EMA10': ema.get("EMA10"), 'EMA50': ema.get("EMA50"),
                    'MACD': macd, 'RSI': rsi, 'MOMENTUM': mom, 'BOLLINGER': bb, 'ATR': atr,
                    'STDDEV': stddev, 'SKEW': skew_, 'KURTOSIS': kurt_
                }

                orderbook_pack = { 'imbalance': orderbooks[symbol].get_imbalance() }

                score = strategy.calculate_score(indicator_pack, orderbook_pack)

                print(f"[{symbol}] Price: {price} | Score: {score:.4f}")

async def launch_all():
    await asyncio.gather(
        handle_websocket(),
        debug_loop()
    )