# trading_bot/tasks/live_trader.py
"""
Lightweight paper live-trader that:
- subscribes to websocket prices
- builds indicators via indicators.compute_all_indicators()
- scores via logic.signals.generate_signal()
- pulls dynamic thresholds/sizing knobs from config/parameters_live.runtime_params()
- applies hysteresis + smoothing
- logs a self-contained session under logs/session_YYYYMMDD_HHMMSS
- routes decisions through execution.trade_executor.process_trade_decision()
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from trading_bot.api import websocket
from trading_bot.config.config import SYMBOLS
from trading_bot.config import parameters
from trading_bot.config.parameters_live import runtime_params  # <-- dynamic live knobs
from trading_bot.state.buffers import (
    get_buffer,
    orderbooks,
    update_multi_timeframe_buffers,
    multi_timeframe_buffers,
    signal_smoothing_buffers,
)
from trading_bot.state.book_features import book_feature_buffers
from trading_bot.logic import indicators
from trading_bot.logic.signals import generate_signal
from trading_bot.utils.math_utils import compute_smoothed_score
from trading_bot.execution.trade_executor import process_trade_decision
from trading_bot.logic import risk_manager
from trading_bot.execution import paper_engine
from trading_bot.config import paths

# ============================================================
#  🔁 Auto-load newest optimizer weights
# ============================================================

DEFAULT_WEIGHTS_DIR = paths.WEIGHTS_DIR  # e.g. data/artifacts/weights
ARTIFACT_WEIGHTS_DIR = os.getenv("TB_ARTIFACTS_ROOT")
if ARTIFACT_WEIGHTS_DIR:
    ARTIFACT_WEIGHTS_DIR = os.path.join(ARTIFACT_WEIGHTS_DIR, "weights")
else:
    ARTIFACT_WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR

_weights_cache: Dict[str, float] | None = None
_weights_pointer_mtime: float | None = None

def _load_current_weights() -> Dict[str, float]:
    """
    Load latest optimized weights produced by optimizer_pipeline.
    Looks for artifacts/weights/current.json.

    Supported formats:
    (A) Pointer file: {"path": "artifacts/weights/20251103_210002_best.json"}
    (B) Direct blob: {"weights": {...}} or {"weights_by_regime": {"default": {...}}}
    """
    global _weights_cache, _weights_pointer_mtime
    pointer_path = os.path.join(ARTIFACT_WEIGHTS_DIR, "current.json")
    try:
        if not os.path.exists(pointer_path):
            return _weights_cache or parameters.SIGNAL_WEIGHTS

        mtime = os.path.getmtime(pointer_path)
        if _weights_cache is not None and _weights_pointer_mtime == mtime:
            return _weights_cache

        with open(pointer_path) as f:
            curr = json.load(f)

        if isinstance(curr, dict) and "path" in curr:
            with open(curr["path"]) as g:
                blob = json.load(g)
        else:
            blob = curr

        if isinstance(blob, dict):
            if "weights_by_regime" in blob:
                w = blob["weights_by_regime"].get("default") or next(iter(blob["weights_by_regime"].values()), None)
            elif "weights" in blob:
                w = blob["weights"]
            else:
                w = blob if all(isinstance(v, (int, float)) for v in blob.values()) else None

            if w and isinstance(w, dict):
                _weights_cache = w
                _weights_pointer_mtime = mtime
                return _weights_cache

        return parameters.SIGNAL_WEIGHTS
    except Exception as e:
        print(f"[live_trader] weights load error: {e}")
        return _weights_cache or parameters.SIGNAL_WEIGHTS

# -------- session init ----------
run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(paths.LOG_ROOT, f"session_{run_ts}")
os.makedirs(SESSION_DIR, exist_ok=True)

# snapshot static params for provenance
with open(os.path.join(SESSION_DIR, "parameters_snapshot.json"), "w") as f:
    snap = {
        k: getattr(parameters, k)
        for k in dir(parameters)
        if not k.startswith("__") and not callable(getattr(parameters, k))
    }
    f.write(json.dumps(snap, indent=2, default=str))

LOG_FILE = os.path.join(SESSION_DIR, "trade_logs.jsonl")
RAW_FILE = os.path.join(SESSION_DIR, "raw_indicators.jsonl")
SUMMARY_FILE = os.path.join(SESSION_DIR, "session_summary.json")

# price memory for PnL + risk mgr
price_store: Dict[str, float] = {
    sym: parameters.SYMBOL_STARTING_PRICES.get(sym, 1.0) for sym in SYMBOLS
}

# live knobs (will be refreshed each loop so drawdown/regime changes apply)
_current_live: Dict[str, Any] = {}
_session_stats = {
    "equity": parameters.STARTING_BALANCE,
    "equity_peak": parameters.STARTING_BALANCE,
    "drawdown": 0.0,
    "max_drawdown": 0.0,
    "trades": 0,
    "last_equity_ts": time.time(),
}

def _persist_session_summary():
    try:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **_session_stats,
        }
        with open(SUMMARY_FILE, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"[live_trader] failed to write session summary: {exc}")

atexit.register(_persist_session_summary)
_persist_session_summary()

def _refresh_live_params() -> None:
    """
    Pull a fresh set of live parameters.
    """
    regime = "default"
    global _current_live
    _current_live = runtime_params(
        name_or_regime=regime,
        session_drawdown=_session_stats["drawdown"],
    )

# first load
_refresh_live_params()

def _to_scalar(x):
    if isinstance(x, pd.Series):
        return x.iloc[-1] if len(x) else None
    if isinstance(x, np.ndarray):
        return x[-1] if x.size else None
    return x

def _update_session_state(pnl_snapshot: dict | None) -> None:
    if not pnl_snapshot:
        return
    equity = pnl_snapshot.get("equity")
    if equity is None:
        return
    stats = _session_stats
    stats["equity"] = float(equity)
    stats["equity_peak"] = max(stats["equity_peak"], stats["equity"])
    peak = stats["equity_peak"]
    drawdown = 0.0 if peak <= 0 else max(0.0, (peak - stats["equity"]) / peak)
    stats["drawdown"] = drawdown
    stats["max_drawdown"] = max(stats["max_drawdown"], drawdown)
    stats["last_equity_ts"] = time.time()
    _persist_session_summary()

def _flatten(pack: dict) -> dict:
    flat = {}
    for k, v in pack.items():
        if k == "BOLLINGER_components" and isinstance(v, dict):
            flat["BOLLINGER_middle"] = _to_scalar(v.get("middle"))
            flat["BOLLINGER_upper"] = _to_scalar(v.get("upper"))
            flat["BOLLINGER_lower"] = _to_scalar(v.get("lower"))
            continue
        if isinstance(v, (int, float, np.floating, np.integer, pd.Series, np.ndarray)):
            v = _to_scalar(v)
        elif isinstance(v, (list, tuple)) and len(v) == 2 and k in {"SUPPORT", "RESISTANCE"}:
            flat[f"{k}_LOW"] = _to_scalar(v[0])
            flat[f"{k}_HIGH"] = _to_scalar(v[1])
            continue
        flat[k] = v if (v is None or isinstance(v, (int, float))) else None
    return flat

def _fmt_top(top_list):
    out = []
    for k, v in top_list:
        val = _to_scalar(v)
        if isinstance(val, (int, float, np.floating)):
            out.append(f"{k}:{val:.2f}")
        else:
            out.append(f"{k}:NA")
    return "[" + ", ".join(out) + "]"

# hysteresis state per symbol to prevent flip-flop
last_action: Dict[str, str] = {sym: "HOLD" for sym in SYMBOLS}
last_trade_ts: Dict[str, float] = {sym: 0.0 for sym in SYMBOLS}

def _rate_limited(symbol: str) -> bool:
    """Simple per-symbol cooldown & per-minute cap."""
    now = time.time()
    cooldown = _current_live["cooldown_sec"]
    if cooldown and now - last_trade_ts[symbol] < cooldown:
        return True
    # soft per-minute limiter handled in executor; we keep a cooldown here
    return False

def _apply_hysteresis(sym: str, smoothed_score: float) -> str:
    """Convert score to decision using dynamic thresholds + hysteresis deadband."""
    if smoothed_score is None:
        return "HOLD"

    prev = last_action.get(sym, "HOLD")
    buy_th  = _current_live["buy_thr"]  if prev != "BUY"  else _current_live["buy_thr"]  + _current_live["hysteresis"]
    sell_th = _current_live["sell_thr"] if prev != "SELL" else _current_live["sell_thr"] - _current_live["hysteresis"]

    if smoothed_score > buy_th:
        last_action[sym] = "BUY"
    elif smoothed_score < sell_th:
        last_action[sym] = "SELL"
    else:
        last_action[sym] = "HOLD"
    return last_action[sym]

async def _loop_once():
    # refresh live knobs so drawdown/regime changes take effect
    _refresh_live_params()

    # pull latest optimized weights (cached unless current.json mtime changes)
    current_weights = _load_current_weights()

    smooth_n = int(_current_live.get("smooth_window", parameters.SIGNAL_SMOOTH_WINDOW))
    smooth_n = max(1, smooth_n)

    for symbol in SYMBOLS:
        buf = get_buffer(symbol)
        book_buf = book_feature_buffers[symbol]

        if not buf:
            continue

        prices = [p["price"] for p in buf if p.get("price") is not None]
        volumes = [p.get("volume", 0) or 0 for p in buf]

        # need at least a couple hundred samples for stable indicators
        if len(prices) < 200:
            continue

        price = float(prices[-1])
        price_store[symbol] = price

        # multi-timeframe queues
        update_multi_timeframe_buffers(symbol, price, volumes[-1])

        # stream EMAs (stateful)
        for w in parameters.EMA_WINDOWS:
            indicators.update_ema(symbol, w, price)

        highs  = [p.get("high", p["price"]) for p in buf]
        lows   = [p.get("low",  p["price"]) for p in buf]
        opens  = [p.get("open", p["price"]) for p in buf]
        closes = prices

        # orderbook snapshot
        ob = orderbooks[symbol]
        bids, asks = ob.get_depth()

        # unified indicator dict
        feats = indicators.compute_all_indicators(
            prices=prices, volumes=volumes,
            highs=highs, lows=lows, opens=opens, closes=closes,
            bids=bids, asks=asks, symbol=symbol,
            book_feature_buffers=book_feature_buffers,
        )

        # keep book/liquidity smoothed set updated
        book_buf.update({
            "BID_DENSITY": feats.get("BID_DENSITY"),
            "ASK_DENSITY": feats.get("ASK_DENSITY"),
            "BID_GAP": feats.get("BID_GAP"),
            "ASK_GAP": feats.get("ASK_GAP"),
            "SPREAD": feats.get("SPREAD"),
            "BID_VOL": feats.get("BID_VOL"),
            "ASK_VOL": feats.get("ASK_VOL"),
        })
        smoothed = book_buf.get_smoothed()
        delta_flow = book_buf.get_delta_flow()

        # build indicator pack expected by generate_signal
        pack = {"PRICE": price, "DELTA_FLOW": delta_flow}
        for name in parameters.INDICATOR_NAMES:
            pack[name] = feats.get(name)

        if smoothed:
            pack.update(smoothed)

        # keep JSONL raw dump flat
        raw = {"timestamp": time.time(), "symbol": symbol}
        raw.update({k: (_to_scalar(v) if isinstance(v, (pd.Series, np.ndarray)) else v)
                    for k, v in pack.items()})
        with open(RAW_FILE, "a") as f:
            f.write(json.dumps(raw, default=str) + "\n")

        # score & meta
        signal, category_subscores, final_score, normalized_scores, meta_confidence, mode_used, top_inds = generate_signal(
            pack, weights=current_weights
        )

        # smoothing buffer
        buf_roll = signal_smoothing_buffers[symbol]
        while len(buf_roll) >= smooth_n:
            buf_roll.popleft()
        buf_roll.append(final_score)

        smoothed_score = compute_smoothed_score(buf_roll) or final_score

        # rate limit & hysteresis → decision
        if _rate_limited(symbol):
            decision = "HOLD"
        else:
            decision = _apply_hysteresis(symbol, smoothed_score)

        # route to paper executor (this updates paper positions/cash)
        process_trade_decision(
            symbol=symbol,
            price=price,
            decision=decision,
            score=smoothed_score,
            atr=pack.get("ATR"),
            meta_confidence=meta_confidence,
        )
        if decision != "HOLD":
            last_trade_ts[symbol] = time.time()
            _session_stats["trades"] += 1

        # structured trade log entry
        log_entry = {
            "timestamp": time.time(),
            "symbol": symbol,
            "price": price,
            "mode_used": mode_used,
            "signal": signal,
            "decision": decision,
            "score": float(final_score) if final_score is not None else None,
            "score_smoothed": float(smoothed_score) if smoothed_score is not None else None,
            "meta_confidence": float(meta_confidence) if meta_confidence is not None else None,
            "top_indicators": [(k, float(_to_scalar(v))) for k, v in top_inds],
            "category_subscores": {k: float(v) if v is not None else None for k, v in category_subscores.items()},
            "indicators": _flatten(pack),
            "live_profile": _current_live.get("profile_name", "default"),
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

        # update risk/PnL views
        risk_manager.update_live_prices(price_store)
        # your paper_engine accepts (price_store, positions) in some places; using positions=None is safe in current codebase
        pnl_snapshot = paper_engine.update_total_pnl(price_store, positions=None)
        _update_session_state(pnl_snapshot)

        # console peek
        print(
            f"[{symbol}] prof={_current_live.get('profile_name','?')} "
            f"score={float(final_score) if final_score is not None else float('nan'):.3f} "
            f"smoothed={float(smoothed_score) if smoothed_score is not None else float('nan'):.3f} "
            f"decision={decision} top={_fmt_top(top_inds)}"
        )

async def _runner():
    while True:
        try:
            await _loop_once()
        except Exception as e:
            print(f"[live_trader] loop error: {e}")
        await asyncio.sleep(int(_current_live["log_every_sec"]))

async def _main():
    await asyncio.gather(
        websocket.handle_websocket(),
        _runner(),
    )

if __name__ == "__main__":
    print("🚀 Live trader started (dynamic thresholds from parameters_live.runtime_params)")
    asyncio.run(_main())
