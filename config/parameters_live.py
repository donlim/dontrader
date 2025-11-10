# trading_bot/config/parameters_live.py
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Dict, Any, Tuple, Optional


# =========================
# Core Live Trading Profile
# =========================
@dataclass(frozen=True)
class Profile:
    # decision thresholds
    buy_thr: float
    sell_thr: float
    hysteresis: float

    # sizing / risk
    base_notional: float          # $ per trade before scaling
    max_notional: float           # hard cap per trade
    size_power: float             # exponent for convex sizing from |score|

    # session guardrails
    max_trades_per_min: int       # throttle per symbol
    dd_soft: float                # widen thresholds above this session drawdown
    dd_hard: float                # flat/halt when session DD exceeds this
    dd_widen: float               # how much to widen when dd > dd_soft
    vol_pct_cap: float            # if realized vol percentile > cap, damp size

    # execution realism (phase-0 defaults; refine in later phases)
    fee_rate: float = 0.0005      # taker fee
    slippage_bps: int = 5         # baseline slippage (bps)
    impact_bps: int = 10          # basic market impact (bps)
    latency_ms: int = 200

    # pacing
    smooth_window: int = 5        # smoothing for score → decision
    log_every_sec: int = 5
    cooldown_sec: int = 0         # min seconds between same-side trades
    min_hold_sec: int = 0         # optional minimum hold time


# ====================
# Default Live Profiles
# ====================
DEFAULT_PROFILES: Dict[str, Profile] = {
    # Phase-0 baseline
    "default": Profile(
        buy_thr=0.20, sell_thr=-0.20, hysteresis=0.02,
        base_notional=500.0, max_notional=3000.0, size_power=1.0,
        max_trades_per_min=3, dd_soft=0.05, dd_hard=0.12, dd_widen=0.05,
        vol_pct_cap=0.80,
    ),
    "aggressive": Profile(
        buy_thr=0.15, sell_thr=-0.15, hysteresis=0.02,
        base_notional=750.0, max_notional=5000.0, size_power=1.15,
        max_trades_per_min=5, dd_soft=0.07, dd_hard=0.15, dd_widen=0.05,
        vol_pct_cap=0.85,
    ),
    "conservative": Profile(
        buy_thr=0.30, sell_thr=-0.30, hysteresis=0.03,
        base_notional=300.0, max_notional=1500.0, size_power=0.90,
        max_trades_per_min=2, dd_soft=0.04, dd_hard=0.08, dd_widen=0.07,
        vol_pct_cap=0.70,
    ),
}

# simple routing (Phase-0; can be replaced by regime detector)
REGIME_TO_PROFILE: Dict[str, str] = {
    "trend": "aggressive",
    "chop": "conservative",
    "high_vol": "conservative",
    "low_vol": "default",
}


# ===============================
# Convenience + Runtime Utilities
# ===============================
def profile_to_dict(name: str) -> Dict[str, Any]:
    """Return a plain dict for easy logging/serialization."""
    p = DEFAULT_PROFILES[name]
    return asdict(p)


def get_profile(name_or_regime: str) -> Tuple[str, Profile]:
    """
    Accepts either a profile name ('default') or a regime label ('trend').
    Returns (resolved_profile_name, Profile).
    """
    if name_or_regime in DEFAULT_PROFILES:
        return name_or_regime, DEFAULT_PROFILES[name_or_regime]
    resolved = REGIME_TO_PROFILE.get(name_or_regime, "default")
    return resolved, DEFAULT_PROFILES[resolved]


def widen_thresholds_for_drawdown(
    profile: Profile, session_drawdown: float
) -> Profile:
    """
    If drawdown crosses soft threshold, widen thresholds by dd_widen.
    If drawdown crosses hard threshold, force flat by setting infinite thresholds.
    """
    if session_drawdown >= profile.dd_hard:
        # lock into HOLD by making thresholds unreachable
        return replace(profile, buy_thr=1e9, sell_thr=-1e9)

    if session_drawdown >= profile.dd_soft:
        widen = profile.dd_widen
        buy = profile.buy_thr + widen
        sell = profile.sell_thr - widen
        return replace(profile, buy_thr=buy, sell_thr=sell)

    return profile


def compute_notional_for_signal(
    score: float,
    price: float,
    profile: Profile,
    vol_pct: Optional[float] = None,
) -> float:
    """
    Convert a signed score into trade notional ($). Uses convex scaling via |score|**size_power.
    Caps by vol percentile (if provided) and profile.max_notional.
    """
    s = abs(float(score))
    base = profile.base_notional * (s ** profile.size_power)

    # dampen if current realized vol percentile is too high
    if vol_pct is not None and vol_pct > profile.vol_pct_cap:
        # linear damping  (e.g., 90th pct with cap 80 → 50% of base)
        excess = min(1.0, (vol_pct - profile.vol_pct_cap) / (1.0 - profile.vol_pct_cap))
        base *= max(0.0, 1.0 - excess)

    return min(base, profile.max_notional)


def runtime_params(
    name_or_regime: str = "default",
    session_drawdown: float = 0.0,
) -> Dict[str, Any]:
    """
    Return a fully-materialized set of live parameters for the current session,
    including any drawdown-based widening.
    """
    name, prof = get_profile(name_or_regime)
    adj = widen_thresholds_for_drawdown(prof, session_drawdown)

    # expose friendly keys expected by simple live loops
    return {
        "profile_name": name,
        "buy_thr": adj.buy_thr,
        "sell_thr": adj.sell_thr,
        "hysteresis": adj.hysteresis,
        "base_notional": adj.base_notional,
        "max_notional": adj.max_notional,
        "size_power": adj.size_power,
        "max_trades_per_min": adj.max_trades_per_min,
        "dd_soft": adj.dd_soft,
        "dd_hard": adj.dd_hard,
        "dd_widen": adj.dd_widen,
        "vol_pct_cap": adj.vol_pct_cap,
        "fee_rate": adj.fee_rate,
        "slippage_bps": adj.slippage_bps,
        "impact_bps": adj.impact_bps,
        "latency_ms": adj.latency_ms,
        "smooth_window": adj.smooth_window,
        "log_every_sec": adj.log_every_sec,
        "cooldown_sec": adj.cooldown_sec,
        "min_hold_sec": adj.min_hold_sec,
        # helpers (delegate functions)
        "compute_notional_for_signal": compute_notional_for_signal,
    }