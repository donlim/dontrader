# trading_bot/tests/test_optimizer_smoke.py

from collections import namedtuple
from unittest.mock import patch

import pandas as pd

from trading_bot.tools import optimizer_engine


SimStub = namedtuple("SimStub", ["final", "equity_curve", "trade_pairs"])


@patch("trading_bot.tools.optimizer_engine.run_simulation")
@patch("trading_bot.tools.optimizer_engine.get_optimizer_settings")
def test_optimize_weights_smoke(mock_settings, mock_run_sim):
    mock_settings.return_value = {
        "MDD_TARGET": 0.50,
        "PENALTY_MDD": 1.0,
        "PENALTY_VOL": 0.5,
        "RISK_AVERSION": 0.0,
        "POPULATION_SIZE": 4,
        "GENERATIONS": 1,
        "MUTATION_RATE": 0.1,
        "ELITISM_COUNT": 1,
        "FEATURE_BOUNDS": {},
        "profile": "test",
    }

    def _stub_simulation(df, weights):
        equity = float(sum(abs(v) for v in weights.values()) + len(df))
        equity_curve = pd.DataFrame({"timestamp": [0, 1], "equity": [equity, equity + 1]})
        return SimStub(final={"BTC": equity}, equity_curve=equity_curve, trade_pairs={})

    mock_run_sim.side_effect = _stub_simulation

    df = pd.DataFrame({
        "timestamp": [1, 2, 3, 4],
        "symbol": ["BTC"] * 4,
        "price": [100, 101, 102, 103],
        "DELTA_FLOW": [0.5, 0.5, -0.5, -0.5],
    })

    weights = optimizer_engine.optimize_weights(df, profile="test")

    assert weights
    assert all(isinstance(v, float) for v in weights.values())
