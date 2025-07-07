# trading_bot/tasks/asset_manager.py

import asyncio
import json
import numpy as np
import os
import time
from datetime import datetime
import pandas as pd

from trading_bot.api import websocket
from trading_bot.config.config import SYMBOLS
from trading_bot.config import parameters
from trading_bot.state.buffers import get_buffer, orderbooks, update_multi_timeframe_buffers, multi_timeframe_buffers, signal_smoothing_buffers
from trading_bot.state.book_features import book_feature_buffers
from trading_bot.logic import indicators
from trading_bot.logic.signals import generate_signal
from trading_bot.utils.math_utils import compute_smoothed_score
from trading_bot.logic.risk import initialize_positions, evaluate_position
from trading_bot.logic import risk_manager
from trading_bot.execution import execution_engine
from trading_bot.execution import paper_engine
from trading_bot.execution.trade_executor import process_trade_decision

def to_scalar_safe(x):
    if isinstance(x, pd.Series):
        return x.iloc[-1] if hasattr(x, 'iloc') else x[-1]
    elif isinstance(x, np.ndarray):
        return x[-1]
    return x

# === Setup dynamic log directory for each run ===
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = os.path.join("logs", f"session_{run_timestamp}")
os.makedirs(session_dir, exist_ok=True)

# ✅ Save parameters snapshot into the session folder
param_snapshot = {k: getattr(parameters, k) for k in dir(parameters) if not k.startswith("__") and not callable(getattr(parameters, k))}
param_file = os.path.join(session_dir, "parameters_snapshot.json")
with open(param_file, "w") as f:
    json.dump(param_snapshot, f, indent=4, default=str)

# === Setup log files ===
log_file = os.path.join(session_dir, "trade_logs.jsonl")
raw_log_file = os.path.join(session_dir, "raw_indicators.jsonl")

# === Live price store ===
price_store = {symbol: parameters.SYMBOL_STARTING_PRICES.get(symbol, 1) for symbol in SYMBOLS}
positions = {symbol: None for symbol in SYMBOLS}

# Initialize EMA states
for symbol in SYMBOLS:
    for window in parameters.EMA_WINDOWS:
        indicators.update_ema(symbol, window, None)

initialize_positions(SYMBOLS)
print("\U0001F680 Starting asset manager...")
def flatten_indicator_pack(indicator_pack):
    flat = {}

    for k, v in indicator_pack.items():
        if k == "BOLLINGER_components":
            # ✅ Extract middle, upper, lower explicitly and skip the parent key itself
            if isinstance(v, dict):
                flat["BOLLINGER_middle"] = float(v.get("middle")) if v.get("middle") is not None else None
                flat["BOLLINGER_upper"] = float(v.get("upper")) if v.get("upper") is not None else None
                flat["BOLLINGER_lower"] = float(v.get("lower")) if v.get("lower") is not None else None
            continue  # Skip adding BOLLINGER_components as a full dict

        if isinstance(v, (int, float, np.integer, np.floating)):
            flat[k] = float(v)

        elif isinstance(v, (list, tuple)) and len(v) == 2 and k in {"SUPPORT", "RESISTANCE"}:
            flat[f"{k}_LOW"], flat[f"{k}_HIGH"] = float(v[0]), float(v[1])

        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}_{sub_k}"] = float(sub_v) if isinstance(sub_v, (int, float, np.integer, np.floating)) else None

        elif isinstance(v, str):
            try:
                flat[k] = float(v)
            except:
                flat[k] = None

        else:
            try:
                flat[k] = float(v)
            except:
                flat[k] = None

    return flat

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

            if len(prices) < 200:
                continue

            price = prices[-1]
            price_store[symbol] = price

            update_multi_timeframe_buffers(symbol, price, volumes[-1])
            print(f"[{symbol}] Multi-TF buffer snapshot: {[ (tf, len(buf)) for tf, buf in multi_timeframe_buffers[symbol].items() ]}")

            for window in parameters.EMA_WINDOWS:
                indicators.update_ema(symbol, window, price)

            highs = [p.get('high', p['price']) for p in buffer]
            lows = [p.get('low', p['price']) for p in buffer]
            opens = [p.get('open', p['price']) for p in buffer]
            closes = prices

            orderbook = orderbooks[symbol]
            bids, asks = orderbook.get_depth()

            indicators_dict = indicators.compute_all_indicators(
                prices=prices,
                volumes=volumes,
                highs=highs,
                lows=lows,
                opens=opens,
                closes=closes,
                bids=bids,
                asks=asks,
                symbol=symbol,
                book_feature_buffers=book_feature_buffers,
            )

            book_buffer.update({
                'BID_DENSITY': indicators_dict.get('BID_DENSITY'),
                'ASK_DENSITY': indicators_dict.get('ASK_DENSITY'),
                'BID_GAP': indicators_dict.get('BID_GAP'),
                'ASK_GAP': indicators_dict.get('ASK_GAP'),
                'SPREAD': indicators_dict.get('SPREAD'),
                'BID_VOL': indicators_dict.get('BID_VOL'),
                'ASK_VOL': indicators_dict.get('ASK_VOL')
            })

            smoothed_features = book_buffer.get_smoothed()
            delta_flow = book_buffer.get_delta_flow()

            indicator_pack = {'PRICE': price}
            for key in parameters.INDICATOR_NAMES:
                indicator_pack[key] = indicators_dict.get(key, None)
            indicator_pack['DELTA_FLOW'] = delta_flow

            if smoothed_features:
                indicator_pack.update(smoothed_features)

            if "BOLLINGER_components" in indicator_pack:
                del indicator_pack["BOLLINGER_components"]

            raw_log_entry = {
                "timestamp": time.time(),
                "symbol": symbol,
            }
            raw_log_entry.update({
                k: float(to_scalar_safe(v)) if isinstance(to_scalar_safe(v), (int, float, np.floating)) else None
                for k, v in indicator_pack.items()
            })
            with open(raw_log_file, "a") as f:
                f.write(json.dumps(raw_log_entry, default=str) + "\n")

            signal, category_subscores, final_score, normalized_scores, meta_confidence, mode_used, top_indicators = generate_signal(indicator_pack)

            # === 🆕 Update smoothing buffer ===
            signal_smoothing_buffers[symbol].append(final_score)

            # === 🆕 Calculate smoothed score ===
            smoothed_score = compute_smoothed_score(signal_smoothing_buffers[symbol])
            if smoothed_score is None:
                smoothed_score = final_score  # fallback
                
            final_decision = evaluate_position(symbol, smoothed_score)
            print(f"[{symbol}] Final Score: {final_score:.4f}, Smoothed Score: {smoothed_score:.4f}, Decision: {final_decision}")

            process_trade_decision(
                symbol=symbol,
                price=price,
                decision=final_decision,
                score=smoothed_score,
                atr=indicator_pack.get('ATR'),
                meta_confidence=meta_confidence
            )

            print(f"\n[{symbol}] \u25b8 Mode: {mode_used} | Signal: {signal} | Decision: {final_decision} | Score: {final_score:.3f}")
            print(f"↳ Top Indicators: {[f'{k} ({v:.2f})' for k, v in top_indicators]}")
            print(f"↳ Confidence: {meta_confidence:.2f} | Category Subscores: " +
                  ", ".join([f"{k}: {v:.2f}" for k, v in category_subscores.items()]))
            log_entry = {
                "timestamp": time.time(),
                "symbol": symbol,
                "price": price,
                "mode_used": parameters.CURRENT_MODE,
                "signal": signal,
                "decision": final_decision,
                "score": final_score,
                "score_smoothed": smoothed_score,
                "meta_confidence": meta_confidence,
                "mode_used": mode_used,
                "sub_scores": {
                    k: float(to_scalar_safe(v)) if isinstance(to_scalar_safe(v), (int, float, np.floating)) else None
                    for k, v in normalized_scores.items()
                },
                "category_subscores": {k: float(v) if v is not None else None for k, v in category_subscores.items()},
                "reason": max(category_subscores.items(), key=lambda x: abs(x[1] or 0))[0],
                "top_indicators": top_indicators,
                "indicators": flatten_indicator_pack(indicator_pack)
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")

        risk_manager.update_live_prices(price_store)
        paper_engine.update_total_pnl(price_store, positions)

async def launch_all():
    await asyncio.gather(
        websocket.handle_websocket(),
        debug_loop()
    )

if __name__ == "__main__":
    asyncio.run(launch_all())