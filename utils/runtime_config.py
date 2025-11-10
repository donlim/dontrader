# trading_bot/utils/runtime_config.py
import glob, os, json, time
from collections import deque, defaultdict
from typing import Dict, Optional
import numpy as np

from trading_bot.config import parameters
from trading_bot.config.parameters_live import DEFAULT_PROFILES, REGIME_TO_PROFILE

def load_latest_weights(artifacts_dir: Optional[str]=None) -> Dict[str, float]:
    """
    Load newest best_weights_*.json if present. Fallback: parameters.SIGNAL_WEIGHTS.
    """
    try:
        root = artifacts_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        pattern = os.path.join(root, "best_weights_*.json")
        cand = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if cand:
            with open(cand[0], "r") as f:
                return json.load(f)
    except Exception:
        pass
    return dict(parameters.SIGNAL_WEIGHTS)

class RateLimiter:
    def __init__(self, per_min:int):
        self.per_min = per_min
        self.ts = deque(maxlen=128)

    def allow(self, now: float) -> bool:
        # drop older than 60s
        while self.ts and now - self.ts[0] > 60.0: self.ts.popleft()
        if len(self.ts) < self.per_min:
            self.ts.append(now)
            return True
        return False

class LiveConfig:
    """
    Holds current live knobs, supports dynamic routing and guards.
    """
    def __init__(self, default_profile: str = "default"):
        self.profile_name = default_profile
        self.profile = DEFAULT_PROFILES[self.profile_name]
        self.rate_limiters = defaultdict(lambda: RateLimiter(self.profile.max_trades_per_min))
        self.session_start = time.time()
        self.equity_peak = 1.0  # normalized; replace with real if available
        self.equity_now = 1.0

    # --- state from PnL to drive DD guards
    def update_equity(self, normalized_equity: float):
        self.equity_now = normalized_equity
        self.equity_peak = max(self.equity_peak, normalized_equity)

    def drawdown(self) -> float:
        if self.equity_peak <= 0: return 0.0
        return max(0.0, (self.equity_peak - self.equity_now) / self.equity_peak)

    # --- simple regime router (upgrade later)
    def route_profile(self, regime: Optional[str]=None):
        name = REGIME_TO_PROFILE.get(regime or "low_vol", self.profile_name)
        if name != self.profile_name:
            self.profile_name = name
            self.profile = DEFAULT_PROFILES[self.profile_name]
            # refresh rate limiters with new cap
            self.rate_limiters = defaultdict(lambda: RateLimiter(self.profile.max_trades_per_min))

    def thresholds(self) -> (float, float, float):
        buy, sell, h = self.profile.buy_thr, self.profile.sell_thr, self.profile.hysteresis
        dd = self.drawdown()
        if dd >= self.profile.dd_hard:
            # hard risk state: force HOLD by widening outside practical reach
            return (+999, -999, h)
        if dd >= self.profile.dd_soft:
            # widen thresholds when in drawdown
            return (buy + self.profile.dd_widen, sell - self.profile.dd_widen, h)
        return (buy, sell, h)

    def hysteresis_decision(self, prev_decision: str, score: float, buy_thr: float, sell_thr: float, hyst: float) -> str:
        if prev_decision == "BUY":
            # require crossing below (sell_thr + hyst) to flip
            if score < (sell_thr + hyst): return "SELL"
            return "BUY" if score > (buy_thr - hyst) else "HOLD"
        if prev_decision == "SELL":
            if score > (buy_thr - hyst): return "BUY"
            return "SELL" if score < (sell_thr + hyst) else "HOLD"
        # previous HOLD
        if score > buy_thr: return "BUY"
        if score < sell_thr: return "SELL"
        return "HOLD"

    def allow_trade(self, symbol: str, now: float) -> bool:
        return self.rate_limiters[symbol].allow(now)

    def size_for(self, price: float, stddev: Optional[float]) -> float:
        # vol-aware sizing: scale down when stddev in top pctile (proxy: z-score clip)
        base = self.profile.base_notional
        if stddev is None or stddev <= 0:
            return min(self.profile.max_notional, base / max(1.0, price))
        # crude, replace with rolling pctile later:
        z = min(3.0, float(stddev) / 1.0)  # assume 1.0 ~ typical unit; refine later
        scale = 1.0 / (1.0 + (z**self.profile.size_power))
        notional = max(0.1 * base, base * scale)
        return min(self.profile.max_notional, notional / max(price, 1e-9))