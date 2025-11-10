# trading_bot/config/optimizer_config.py

import os
from typing import Optional

# You can switch profiles at runtime with the env var: OPT_PROFILE=balanced
ACTIVE_PROFILE = os.getenv("OPT_PROFILE", "balanced")

RISK_PROFILES = {
    # Higher returns allowed, looser penalties
    "aggressive": {
        "MDD_TARGET": 0.70,      # allow up to 70% DD before strong penalty
        "PENALTY_MDD": 0.5,      # weak DD penalty
        "PENALTY_VOL": 0.25,     # weak volatility penalty
        "RISK_AVERSION": 0.0,    # only penalize risk via MDD/Vol
        "POPULATION_SIZE": 60,
        "GENERATIONS": 30,
        "MUTATION_RATE": 0.15,
        "ELITISM_COUNT": 2,
        # Per-feature bounds (optional): default [0,1] if not specified here
        "FEATURE_BOUNDS": {},
    },

    # Good default for research cycles
    "balanced": {
        "MDD_TARGET": 0.50,
        "PENALTY_MDD": 1.0,
        "PENALTY_VOL": 0.50,
        "RISK_AVERSION": 0.0,
        "POPULATION_SIZE": 50,
        "GENERATIONS": 30,
        "MUTATION_RATE": 0.10,
        "ELITISM_COUNT": 1,
        "FEATURE_BOUNDS": {},
    },

    # Much stricter on drawdowns/vol
    "conservative": {
        "MDD_TARGET": 0.35,
        "PENALTY_MDD": 3.0,
        "PENALTY_VOL": 1.0,
        "RISK_AVERSION": 0.0,
        "POPULATION_SIZE": 50,
        "GENERATIONS": 40,
        "MUTATION_RATE": 0.08,
        "ELITISM_COUNT": 2,
        "FEATURE_BOUNDS": {},
    },
}

def get_optimizer_settings(profile: Optional[str] = None) -> dict:
    name = (profile or ACTIVE_PROFILE).lower()
    return RISK_PROFILES.get(name, RISK_PROFILES["balanced"]) | {"profile": name}