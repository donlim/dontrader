# trading_bot/tasks/asset_manager.py

import asyncio
from trading_bot.api import websocket
from trading_bot.config.config import SYMBOLS
from trading_bot.state.buffers import get_buffer, orderbooks
from trading_bot.config import parameters
from trading_bot.logic import indicators
from trading_bot.state.book_features import book_feature_buffers

positions = {symbol: None for symbol in SYMBOLS}

# Initialize EMA states
for symbol in SYMBOLS:
    for window in parameters.EMA_WINDOWS:
        indicators.update_ema(symbol, window, None)

async def debug_loop():
    while True:
        await asyncio.sleep(5)
        for symbol in SYMBOLS:
            buffer = get_buffer(symbol)
            book_buffer = book_feature_buffers[symbol]

            prices = [p['price'] for p in buffer]
            volumes = [p['volume'] or 0 for p in buffer]

            if prices and volumes:
                price = prices[-1]

                # ✅ Update EMAs live
                for window in parameters.EMA_WINDOWS:
                    indicators.update_ema(symbol, window, price)

                ema = {f"EMA{w}": indicators.get_ema(symbol, w) for w in parameters.EMA_WINDOWS}
                sma = {f"SMA{w}": indicators.compute_sma(prices[-w:]) for w in parameters.SMA_WINDOWS}
                rsi = indicators.compute_rsi(prices, parameters.RSI_WINDOW)
                stoch_rsi = indicators.compute_stoch_rsi(prices, parameters.RSI_WINDOW, parameters.STOCH_WINDOW)
                mom = indicators.compute_momentum(prices, parameters.MOMENTUM_WINDOW)
                macd = indicators.compute_macd(symbol)
                bb = indicators.compute_bollinger(prices, parameters.BOLLINGER_WINDOW, parameters.BOLLINGER_K)
                atr = indicators.compute_atr(prices, parameters.ATR_WINDOW)
                vwap = indicators.compute_vwap(prices, volumes)
                obv = indicators.compute_obv(prices, volumes)
                ad = indicators.compute_accumulation_distribution(prices, volumes, parameters.AD_WINDOW)
                support, resistance = indicators.detect_support_resistance(
                    prices, parameters.SUPPORT_RESISTANCE_WINDOW, parameters.SUPPORT_RESISTANCE_TOLERANCE
                )
                stddev = indicators.compute_stddev(prices, parameters.STDDEV_WINDOW)
                skew = indicators.compute_skew(prices, parameters.SKEW_WINDOW)
                kurt = indicators.compute_kurtosis(prices, parameters.KURTOSIS_WINDOW)

                # ✅ Order Book Snapshot (Level 2)
                orderbook = orderbooks[symbol]
                bids, asks = orderbook.get_depth()
                full_imbalance = indicators.compute_full_book_imbalance(bids, asks)
                top_imbalance = indicators.compute_book_imbalance(bids, asks, depth=5)
                density_bid, density_ask = indicators.compute_book_density(bids, asks)
                min_bid_gap, min_ask_gap = indicators.compute_liquidity_gap(bids, asks)
                spread = indicators.compute_spread(bids, asks)
                bid_volatility, ask_volatility = indicators.compute_top_volatility(bids, asks)

                # ✅ Push raw liquidity features into book_feature buffer
                liquidity_features = {
                    'BID_DENSITY': density_bid,
                    'ASK_DENSITY': density_ask,
                    'BID_GAP': min_bid_gap,
                    'ASK_GAP': min_ask_gap,
                    'SPREAD': spread,
                    'BID_VOL': bid_volatility,
                    'ASK_VOL': ask_volatility
                }
                book_buffer.update(liquidity_features)

                # ✅ Grab smoothed versions (liquidity + delta flow)
                smoothed_features = book_buffer.get_smoothed()
                delta_flow = book_buffer.get_delta_flow()

                # ✅ Combine everything
                indicator_pack = {
                    'PRICE': price, 'EMA10': ema.get("EMA10"), 'EMA50': ema.get("EMA50"),
                    'MACD': macd, 'RSI': rsi, 'STOCH_RSI': stoch_rsi, 'MOMENTUM': mom,
                    'BOLLINGER': bb, 'ATR': atr, 'VWAP': vwap, 'OBV': obv, 'AD': ad,
                    'SUPPORT': support, 'RESISTANCE': resistance,
                    'STDDEV': stddev, 'SKEW': skew, 'KURTOSIS': kurt,
                    'FULL_BOOK_IMB': full_imbalance, 'BOOK_IMB': top_imbalance,
                    'DELTA_FLOW': delta_flow
                }

                if smoothed_features:
                    indicator_pack.update(smoothed_features)

                print(f"[{symbol}] Indicators: {indicator_pack}")

async def launch_all():
    await asyncio.gather(
        websocket.handle_websocket(),
        debug_loop()
    )