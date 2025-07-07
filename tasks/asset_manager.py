# trading_bot/tasks/asset_manager.py

import asyncio
import json
import os
import time
from datetime import datetime

from trading_bot.api import websocket
from trading_bot.config.config import SYMBOLS
from trading_bot.config import parameters
from trading_bot.state.buffers import get_buffer, orderbooks
from trading_bot.state.book_features import book_feature_buffers
from trading_bot.logic import indicators
from trading_bot.logic.signals import generate_signal
from trading_bot.logic.risk import initialize_positions, evaluate_position
from trading_bot.logic import risk_manager
from trading_bot.execution import execution_engine
from trading_bot.execution import paper_engine

# === Setup dynamic log directory for each run ===
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = os.path.join("logs", f"session_{run_timestamp}")
os.makedirs(session_dir, exist_ok=True)

# ✅ Save parameters snapshot into the session folder
param_snapshot = {k: getattr(parameters, k) for k in dir(parameters) if not k.startswith("__") and not callable(getattr(parameters, k))}
param_file = os.path.join(session_dir, "parameters_snapshot.json")
with open(param_file, "w") as f:
    json.dump(param_snapshot, f, indent=4, default=str)

# === Setup trade log file for this run ===
log_file = os.path.join(session_dir, "trade_logs.jsonl")

# === Live price store ===
price_store = {symbol: parameters.SYMBOL_STARTING_PRICES.get(symbol, 1) for symbol in SYMBOLS}
positions = {symbol: None for symbol in SYMBOLS}

# Initialize EMA states
for symbol in SYMBOLS:
    for window in parameters.EMA_WINDOWS:
        indicators.update_ema(symbol, window, None)

initialize_positions(SYMBOLS)
print("🚀 Starting asset manager...")
async def debug_loop():
    while True:
        await asyncio.sleep(5)

        for symbol in SYMBOLS:
            buffer = get_buffer(symbol)
            book_buffer = book_feature_buffers[symbol]

            prices = [p['price'] for p in buffer]
            volumes = [p['volume'] or 0 for p in buffer]

            if not prices or not volumes:
                continue

            price = prices[-1]
            price_store[symbol] = price

            for window in parameters.EMA_WINDOWS:
                indicators.update_ema(symbol, window, price)

            # ✅ Extract prices/volumes safely
            prices = [p['price'] for p in buffer]
            volumes = [p['volume'] or 0 for p in buffer]
            highs = [p.get('high', p['price']) for p in buffer]
            lows = [p.get('low', p['price']) for p in buffer]
            opens = [p.get('open', p['price']) for p in buffer]
            closes = prices

            ema = {f"EMA{w}": indicators.get_ema(symbol, w) for w in parameters.EMA_WINDOWS}
            ema10 = ema.get("EMA10")
            ema50 = ema.get("EMA50")
            ema_diff = ema10 - ema50 if ema10 is not None and ema50 is not None else 0

            sma = {f"SMA{w}": indicators.compute_sma(prices[-w:]) for w in parameters.SMA_WINDOWS}
            sma50 = sma.get("SMA50")
            sma200 = sma.get("SMA200")

            rsi = indicators.compute_rsi(prices, parameters.RSI_WINDOW)
            stoch_rsi = indicators.compute_stoch_rsi(prices, parameters.RSI_WINDOW, parameters.STOCH_WINDOW)
            mom = indicators.compute_momentum(prices, parameters.MOMENTUM_WINDOW)
            macd = indicators.compute_macd(symbol)
            bb = indicators.compute_bollinger(prices, parameters.BOLLINGER_WINDOW, parameters.BOLLINGER_K)
            atr = indicators.compute_atr(prices, parameters.ATR_WINDOW)
            vwap = indicators.compute_vwap(prices, volumes)
            obv = indicators.compute_obv(prices, volumes)
            ad = indicators.compute_accumulation_distribution(prices, volumes, parameters.AD_WINDOW)
            support, resistance = indicators.detect_support_resistance(prices, parameters.SUPPORT_RESISTANCE_WINDOW, parameters.SUPPORT_RESISTANCE_TOLERANCE)
            stddev = indicators.compute_stddev(prices, parameters.STDDEV_WINDOW)
            skew_val = indicators.compute_skew(prices, parameters.SKEW_WINDOW)
            kurt_val = indicators.compute_kurtosis(prices, parameters.KURTOSIS_WINDOW)

            # ✅ Pass highs/lows for ADX
            adx = indicators.compute_adx(highs, lows, prices, parameters.ADX_WINDOW)
            cci = indicators.compute_cci(highs, lows, closes, parameters.CCI_WINDOW)
            roc = indicators.compute_roc(prices, parameters.ROC_WINDOW)
            tsi = indicators.compute_tsi(prices, parameters.TSI_FAST, parameters.TSI_SLOW)
            kvo = indicators.compute_kvo(closes, volumes, parameters.KVO_FAST, parameters.KVO_SLOW)
            williams_r = indicators.compute_williams_r(highs, lows, closes, parameters.WILLIAMS_R_WINDOW)
            donchian_upper, donchian_lower = indicators.compute_donchian_channels(highs, lows, parameters.DONCHIAN_WINDOW)
            parabolic_sar = indicators.compute_parabolic_sar(highs, lows, parameters.PARABOLIC_SAR_STEP, parameters.PARABOLIC_SAR_MAX_STEP)
            trend_strength = indicators.compute_trend_strength(prices, highs, lows)
            heikin_ratio = indicators.compute_heikin_ashi_ratio(opens, highs, lows, closes)
            cmf = indicators.compute_cmf(highs, lows, closes, volumes, parameters.CMF_WINDOW)
            donchian_width = indicators.compute_donchian_width(highs, lows, parameters.DONCHIAN_WINDOW)

            # === Order book features ===
            orderbook = orderbooks[symbol]
            bids, asks = orderbook.get_depth()
            full_imbalance = indicators.compute_full_book_imbalance(bids, asks)
            top_imbalance = indicators.compute_book_imbalance(bids, asks, depth=5)
            density_bid, density_ask = indicators.compute_book_density(bids, asks)
            min_bid_gap, min_ask_gap = indicators.compute_liquidity_gap(bids, asks)
            spread = indicators.compute_spread(bids, asks)
            bid_volatility, ask_volatility = indicators.compute_top_volatility(bids, asks)

            liquidity_features = {
                'BID_DENSITY': density_bid, 'ASK_DENSITY': density_ask,
                'BID_GAP': min_bid_gap, 'ASK_GAP': min_ask_gap,
                'SPREAD': spread, 'BID_VOL': bid_volatility, 'ASK_VOL': ask_volatility
            }
            book_buffer.update(liquidity_features)

            smoothed_features = book_buffer.get_smoothed()
            delta_flow = book_buffer.get_delta_flow()

            indicator_pack = {
                'PRICE': price, 'EMA10': ema10, 'EMA50': ema50, 'EMA_DIFF': ema_diff,
                'MACD': macd, 'RSI': rsi, 'STOCH_RSI': stoch_rsi, 'MOMENTUM': mom,
                'BOLLINGER': bb, 'ATR': atr, 'VWAP': vwap, 'OBV': obv, 'AD': ad,
                'SUPPORT': support, 'RESISTANCE': resistance,
                'STDDEV': stddev, 'SKEW': skew_val, 'KURTOSIS': kurt_val,
                'FULL_BOOK_IMB': full_imbalance, 'BOOK_IMB': top_imbalance, 'DELTA_FLOW': delta_flow,
                'ADX': adx, 'CCI': cci, 'ROC': roc, 'TSI': tsi, 'KVO': kvo,
                'WILLIAMS_R': williams_r, 'DONCHIAN_UPPER': donchian_upper,
                'DONCHIAN_LOWER': donchian_lower, 'PARABOLIC_SAR': parabolic_sar,
                'TREND_STRENGTH': trend_strength, 'SMA50': sma50, 'SMA200': sma200,
                'HEIKIN_RATIO': heikin_ratio, 'CMF': cmf, 'DONCHIAN_WIDTH': donchian_width
            }

            if smoothed_features:
                indicator_pack.update(smoothed_features)

            signal, sub_scores, final_score = generate_signal(indicator_pack)
            final_decision = evaluate_position(symbol, signal)

            execution_engine.execute_trade(symbol, price, final_decision, final_score, atr)

            print(f"[{symbol}] Signal: {signal} | Decision: {final_decision} | Score: {final_score:.3f} | Sub-Scores: {sub_scores}")
            print(f"[{symbol}] Indicators: {indicator_pack}")

            log_entry = {
                "timestamp": time.time(),
                "symbol": symbol,
                "price": price,
                "decision": final_decision,
                "score": final_score,
                "sub_scores": {k: float(v) for k, v in sub_scores.items()},
                "indicators": {k: float(v) if isinstance(v, (int, float)) else v for k, v in indicator_pack.items()}
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        risk_manager.update_live_prices(price_store)
        paper_engine.update_total_pnl(price_store)

async def launch_all():
    await asyncio.gather(
        websocket.handle_websocket(),
        debug_loop()
    )

if __name__ == "__main__":
    asyncio.run(launch_all())