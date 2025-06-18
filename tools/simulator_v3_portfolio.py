# simulator_v3_portfolio.py

import pandas as pd

class PortfolioSimulator:
    def __init__(self, symbols, starting_balance=100000, fee_rate=0.0005, threshold=0.25):
        self.symbols = symbols
        self.balances = {sym: starting_balance for sym in symbols}
        self.positions = {sym: 0 for sym in symbols}
        self.fee_rate = fee_rate
        self.threshold = threshold

    def simulate(self, df, weights):
        for _, row in df.iterrows():
            symbol = row['symbol']
            price = row['price']
            cash = self.balances[symbol]
            position = self.positions[symbol]

            # Compute weighted score
            score = sum(weights[k] * row[k] for k in weights)

            size = 1000 / price
            fee = price * size * self.fee_rate

            if score > self.threshold:
                self.balances[symbol] -= (price * size + fee)
                self.positions[symbol] += size

            elif score < -self.threshold:
                self.balances[symbol] += (price * size - fee)
                self.positions[symbol] -= size

        final = {}
        for sym in self.symbols:
            final[sym] = self.balances[sym] + self.positions[sym] * df[df['symbol'] == sym].iloc[-1]['price']
        return final