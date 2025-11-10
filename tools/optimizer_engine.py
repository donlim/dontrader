# trading_bot/tools/optimizer_engine.py

from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed

from trading_bot.tools.simulator_engine import run_simulation
from trading_bot.config.parameters import INDICATOR_NAMES as FEATURES
from trading_bot.config.optimizer_config import get_optimizer_settings  # ← profiles & knobs


# ========= Utilities (risk math) ============================================

def _portfolio_series(equity_curve_df) -> np.ndarray:
    """
    Collapse per-symbol equity_curve to a single portfolio time series.
    Expects columns ['timestamp','equity'] in the DataFrame.
    """
    if equity_curve_df is None or equity_curve_df.empty:
        return np.array([], dtype=float)
    df = equity_curve_df[["timestamp", "equity"]].copy()
    port = df.groupby("timestamp")["equity"].sum().sort_index()
    return port.values.astype(float)


def _max_drawdown(series: np.ndarray) -> float:
    if series.size == 0:
        return 0.0
    roll_max = np.maximum.accumulate(series)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (series - roll_max) / np.where(roll_max == 0, np.nan, roll_max)
    dd = dd[~np.isnan(dd)]
    return float(-dd.min()) if dd.size else 0.0


def _realized_vol(series: np.ndarray) -> float:
    if series.size < 2:
        return 0.0
    pct = np.diff(series) / np.where(series[:-1] == 0, np.nan, series[:-1])
    pct = pct[~np.isnan(pct)]
    return float(np.std(pct)) if pct.size else 0.0


def _risk_adjusted_objective(sim_result, settings: Dict) -> float:
    """
    Convert a SimulationResult into a scalar fitness with risk penalties.
    Higher is better.
    """
    try:
        equity_sum = float(sum(sim_result.final.values()))
    except Exception:
        return -float("inf")

    series = _portfolio_series(sim_result.equity_curve)
    mdd = _max_drawdown(series)
    vol = _realized_vol(series)

    over = max(0.0, mdd - settings["MDD_TARGET"])
    penalty = settings["PENALTY_MDD"] * over + settings["PENALTY_VOL"] * vol
    # linear risk aversion on top (optional)
    risk_aversion = float(settings.get("RISK_AVERSION", 0.0))

    fitness = equity_sum - penalty * abs(equity_sum) - risk_aversion * equity_sum
    if not np.isfinite(fitness):
        return -float("inf")
    return fitness


# ========= GA operators ======================================================

def _clip(val: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, val)))


def _tournament_select(pop: List[List[float]], scores: List[float], k: int) -> List[float]:
    k = max(2, min(k, len(pop)))
    idxs = np.random.choice(len(pop), size=k, replace=False)
    best = max(idxs, key=lambda i: scores[i])
    return pop[best][:]


def _uniform_crossover(a: List[float], b: List[float], p: float = 0.5) -> List[float]:
    if not a:
        return []
    mask = np.random.rand(len(a)) < p
    return [ai if m else bi for ai, bi, m in zip(a, b, mask)]


def _mutate(ind: List[float], rate: float, sigma: float, bounds_by_index: Dict[int, Tuple[float, float]]) -> List[float]:
    out = ind[:]
    for i in range(len(out)):
        if np.random.rand() < rate:
            lo, hi = bounds_by_index.get(i, (0.0, 1.0))
            out[i] = _clip(out[i] + np.random.normal(0.0, sigma), lo, hi)
    return out


# ========= Public API ========================================================

def optimize_weights(train_df, profile: str | None = None) -> Dict[str, float]:
    """
    GA optimizer for feature weights.
    - Uses settings from config/optimizer_config.py (select profile via arg or OPT_PROFILE env).
    - Returns {feature_name: weight}.
    """
    settings = get_optimizer_settings(profile)

    # Seeds for reproducibility (optional; remove if you want more randomness run-to-run)
    np.random.seed(42)
    random.seed(42)

    # Build active feature list from training columns
    active_features: List[str] = [f for f in FEATURES if f in train_df.columns]
    print(f"🧪 Optimizing {len(active_features)} features out of {len(FEATURES)} total...")

    if not active_features:
        print("⚠️ No overlapping feature columns; returning empty weights.")
        return {}

    # Map index → (lo, hi) bounds using FEATURE_BOUNDS (default [0,1])
    idx_bounds: Dict[int, Tuple[float, float]] = {
        i: tuple(settings.get("FEATURE_BOUNDS", {}).get(name, (0.0, 1.0)))
        for i, name in enumerate(active_features)
    }

    POP = int(settings["POPULATION_SIZE"])
    GEN = int(settings["GENERATIONS"])
    MUT = float(settings["MUTATION_RATE"])
    ELT = int(settings["ELITISM_COUNT"])
    TOURN = 3
    N_JOBS = -1

    # ----- initialize population within bounds
    population: List[List[float]] = []
    for _ in range(POP):
        vec = []
        for i, name in enumerate(active_features):
            lo, hi = idx_bounds[i]
            vec.append(float(np.random.uniform(lo, hi)))
        population.append(vec)

    # ----- fitness wrapper (parallel safe)
    def _fitness(vec: List[float]) -> float:
        w = dict(zip(active_features, vec))
        sim = run_simulation(train_df, w)
        return _risk_adjusted_objective(sim, settings)

    # ----- GA loop
    for g in range(GEN):
        try:
            scores: List[float] = Parallel(n_jobs=N_JOBS)(
                delayed(_fitness)(ind) for ind in population
            )
        except Exception as e:
            print(f"[optimizer_engine] Parallel eval error (gen {g+1}): {e}")
            scores = [_fitness(ind) for ind in population]

        best = max(scores) if scores else float("-inf")
        print(f"Generation {g+1}: Best (risk-adjusted) = {best:,.2f}")

        # elitism
        ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        elites = [ind[:] for ind, _ in ranked[:ELT]]

        # parent pool via tournament selection
        parents = [_tournament_select(population, scores, TOURN) for _ in range(max(2, POP // 2))]

        # next gen
        next_pop: List[List[float]] = elites[:]
        while len(next_pop) < POP:
            p1, p2 = random.choice(parents), random.choice(parents)
            child = _uniform_crossover(p1, p2, p=0.5)
            child = _mutate(child, rate=MUT, sigma=0.10, bounds_by_index=idx_bounds)
            next_pop.append(child)

        population = next_pop

    # ----- final selection
    try:
        final_scores: List[float] = Parallel(n_jobs=N_JOBS)(
            delayed(_fitness)(ind) for ind in population
        )
    except Exception as e:
        print(f"[optimizer_engine] Final eval error: {e}")
        final_scores = [_fitness(ind) for ind in population]

    best_idx = int(np.argmax(final_scores))
    best_vec = population[best_idx]
    best_score = final_scores[best_idx]
    best_weights = dict(zip(active_features, map(float, best_vec)))

    print("\n✅ Optimization Complete")
    print(f"🏆 Best risk-adjusted score: {best_score:,.2f}")
    print(json.dumps(best_weights, indent=4))

    return best_weights