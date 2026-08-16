# trading_bot/tests/test_simulator_engine.py

import pandas as pd

from trading_bot.tools.simulator_engine import PortfolioSimulator, SimulationResult
from trading_bot.config import parameters


def test_portfolio_simulator_respects_position_caps():
    rows = []
    for i in range(6):
        rows.append({
            "timestamp": 1_700_000_000 + i,
            "symbol": "BTC",
            "price": 100 + i,
            "DELTA_FLOW": 10 if i < 3 else -10,
            "meta_confidence": 1.0,
        })
    df = pd.DataFrame(rows)

    sim = PortfolioSimulator(["BTC"])
    result = sim.simulate(df, {"DELTA_FLOW": 1.0})

    assert isinstance(result, SimulationResult)
    assert not result.equity_curve.empty
    assert max(sim.positions.values()) <= parameters.MAX_POSITION_SIZE + 1e-9
