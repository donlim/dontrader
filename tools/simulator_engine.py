# trading_bot/tools/simulator_engine.py

import os
import json
from collections import namedtuple
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd

from trading_bot.config import parameters

# ✅ Structured result type
SimulationResult = namedtuple("SimulationResult", ["final", "equity_curve", "trade_pairs"])


def _target_notional(score: float) -> float:
    """
    Map a score into dollar notional using the same convex scaling as the paper engine.
    """
    confidence = max(0.0, abs(float(score)))
    dollars = parameters.RISK_PER_TRADE * (confidence ** parameters.POSITION_SCALING_POWER)
    return min(dollars, parameters.MAX_POSITION_NOTIONAL)


def _resolve_meta_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Try to find meta columns for price, symbol, timestamp under several common names.
    Returns (price_col, symbol_col, ts_col).
    Raises ValueError with guidance if any is missing.
    """
    def pick(cands: Iterable[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        # loose fallback: substring match (e.g., 'PRICE', 'close_price')
        for c in df.columns:
            low = str(c).lower()
            for k in cands:
                if k.lower() in low:
                    return c
        return None

    price_col = pick(["price", "PRICE", "close", "Close"])
    symbol_col = pick(["symbol", "SYMBOL", "ticker", "Ticker"])
    ts_col = pick(["timestamp", "time", "datetime", "Date", "date"])

    missing = [nm for nm, v in [("price", price_col), ("symbol", symbol_col), ("timestamp", ts_col)] if v is None]
    if missing:
        raise ValueError(
            "Simulator requires meta columns but could not resolve: "
            f"{', '.join(missing)}.\n"
            "→ Ensure your optimizer pipeline keeps these columns when it calls run_simulation().\n"
            "  For example, after build_features(df_raw), reattach:\n"
            "    df_features['symbol']    = df_raw['symbol'].values\n"
            "    df_features['price']     = df_raw['price'].values\n"
            "    df_features['timestamp'] = df_raw['timestamp'].values\n"
            f"Columns present now: {list(df.columns)[:20]}..."
        )
    return price_col, symbol_col, ts_col


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


class PortfolioSimulator:
    """
    Industry-grade portfolio simulation engine for quant strategy evaluation.
    Robust to missing/renamed meta columns and missing features.
    """
    def __init__(self, symbols: Iterable[str]):
        self.symbols = list(symbols)
        self.cash: float = parameters.STARTING_BALANCE
        self.positions: Dict[str, float] = {sym: 0.0 for sym in self.symbols}

        mode_config = parameters.BOT_MODES.get(parameters.CURRENT_MODE, {})
        self.buy_threshold = mode_config.get("buy_threshold", parameters.DEFAULT_BUY_THRESHOLD)
        self.sell_threshold = mode_config.get("sell_threshold", parameters.DEFAULT_SELL_THRESHOLD)
        self.fee_rate = parameters.FEE_RATE

        # Tracking
        self.equity_curve = []  # list of dicts
        self.trade_pairs = {sym: [] for sym in self.symbols}
        self.last_entry_time = {sym: None for sym in self.symbols}
        self.in_position = {sym: False for sym in self.symbols}
        self.last_price = {sym: None for sym in self.symbols}

    def simulate(self, df: pd.DataFrame, weights: Dict[str, float]) -> SimulationResult:
        """
        Simulate portfolio execution based on weighted signal scores.

        Expects df to include meta columns (price, symbol, timestamp) under any of:
           - price/PRICE/close
           - symbol/SYMBOL/ticker
           - timestamp/time/datetime
        """
        if df is None or df.empty:
            return SimulationResult(final={}, equity_curve=pd.DataFrame(), trade_pairs=self.trade_pairs)

        price_col, symbol_col, ts_col = _resolve_meta_columns(df)

        # Ensure sorted by time for realistic P&L
        try:
            # tolerate epoch floats/ints or ISO strings
            ts_vals = pd.to_datetime(df[ts_col], errors="coerce", unit="s")
            if ts_vals.isna().all():
                ts_vals = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.assign(_ts=ts_vals).sort_values("_ts").drop(columns=["_ts"])
        except Exception:
            df = df.sort_values(ts_col)

        # Numeric coercions for safety
        df[price_col] = df[price_col].map(_as_float)
        df = df[df[price_col].notna()]

        # Only use features that exist in df
        usable_features = [k for k in weights.keys() if k in df.columns]
        if not usable_features:
            # Nothing to score on—neutral simulation (hold cash)
            final = {}
            for sym in df[symbol_col].unique():
                last_price = df.loc[df[symbol_col] == sym, price_col].tail(1)
                last_price = float(last_price.values[0]) if not last_price.empty else 0.0
                final[sym] = self.balances.get(sym, parameters.STARTING_BALANCE) + self.positions.get(sym, 0.0) * last_price
            return SimulationResult(final=final, equity_curve=pd.DataFrame(self.equity_curve), trade_pairs=self.trade_pairs)

        # Iterate
        for _, row in df.iterrows():
            symbol = row[symbol_col]
            if symbol not in self.positions:
                # new symbol encountered; initialize on the fly
                self.positions.setdefault(symbol, 0.0)
                self.trade_pairs.setdefault(symbol, [])
                self.last_entry_time.setdefault(symbol, None)
                self.in_position.setdefault(symbol, False)
                self.last_price.setdefault(symbol, None)

            price = _as_float(row[price_col])
            if not np.isfinite(price) or price <= 0:
                continue

            timestamp = row[ts_col]
            self.last_price[symbol] = price

            # Weighted signal score (robust to missing cols)
            score = 0.0
            for k in usable_features:
                try:
                    score += float(weights.get(k, 0.0)) * float(row.get(k, 0.0))
                except Exception:
                    # ignore ill-typed feature row values
                    continue

            desired_notional = _target_notional(score)
            desired_qty = desired_notional / price if price > 0 else 0.0
            fee = 0.0

            # Decision
            if score > self.buy_threshold:
                decision = "BUY"
                max_affordable_notional = max(0.0, self.cash / (1 + self.fee_rate))
                affordable_qty = max_affordable_notional / price if price > 0 else 0.0
                qty = min(desired_qty, affordable_qty)
                if qty > 0:
                    fee = price * qty * self.fee_rate
                    cost = price * qty + fee
                    self.cash -= cost
                    self.positions[symbol] += qty

                    if not self.in_position[symbol]:
                        self.last_entry_time[symbol] = timestamp
                        self.in_position[symbol] = True
                else:
                    decision = "HOLD"

            elif score < self.sell_threshold:
                decision = "SELL"
                held = self.positions.get(symbol, 0.0)
                qty = min(held, desired_qty if desired_qty > 0 else held)
                if qty > 0:
                    fee = price * qty * self.fee_rate
                    proceeds = price * qty - fee
                    self.cash += proceeds
                    self.positions[symbol] -= qty

                    if self.in_position[symbol] and self.positions[symbol] <= 0:
                        entry_time = self.last_entry_time[symbol]
                        if entry_time is not None:
                            self.trade_pairs[symbol].append((entry_time, timestamp))
                        self.last_entry_time[symbol] = None
                        self.in_position[symbol] = False
                else:
                    decision = "HOLD"
            else:
                decision = "HOLD"

            equity = self.cash
            for sym, qty in self.positions.items():
                last_px = self.last_price.get(sym)
                if last_px is None:
                    continue
                equity += qty * last_px
            self.equity_curve.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "equity": float(equity),
                "score": float(score),
                "decision": decision,
                "meta_confidence": float(row.get("meta_confidence", 0.0) or 0.0),
            })

        # Final equity snapshot
        final: Dict[str, float] = {}
        for sym in set(self.symbols) | set(df[symbol_col].unique().tolist()):
            latest_price = self.last_price.get(sym)
            if latest_price is None:
                sym_df = df[df[symbol_col] == sym]
                latest_price = float(sym_df.iloc[-1][price_col]) if not sym_df.empty else 0.0
            final[sym] = float(self.positions.get(sym, 0.0) * (latest_price or 0.0))
        final["cash"] = float(self.cash)

        return SimulationResult(final, pd.DataFrame(self.equity_curve), self.trade_pairs)


# === Compatibility Wrappers ===

def run_simulation(df: pd.DataFrame, weights: Dict[str, float]) -> SimulationResult:
    """
    Runs a full simulation and returns SimulationResult.
    """
    # Resolve symbols robustly
    symbol_col = None
    for c in ("symbol", "SYMBOL", "ticker", "Ticker"):
        if c in df.columns:
            symbol_col = c
            break
    symbols = df[symbol_col].unique().tolist() if symbol_col else []
    sim = PortfolioSimulator(symbols)
    return sim.simulate(df, weights)


def simulate_portfolio_with_execution(df: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
    """
    Returns only final equity per symbol for quick evaluation.
    """
    result = run_simulation(df, weights)
    return result.final


# === Entry Test ===
if __name__ == "__main__":
    # These “latest” files are illustrative. Your optimizer_pipeline now writes
    # versioned artifacts under artifacts/{datasets,weights}/.
    latest_csv = "training_dataset_latest.csv"
    latest_weights = "best_weights_latest.json"

    if not os.path.exists(latest_csv) or not os.path.exists(latest_weights):
        raise FileNotFoundError("Missing latest CSV or weights. Run optimizer_pipeline first.")

    df = pd.read_csv(latest_csv)
    with open(latest_weights) as f:
        weights = json.load(f)

    # Run simulation
    result = run_simulation(df, weights)

    # Save equity curve
    result.equity_curve.to_csv("equity_curve_latest.csv", index=False)

    # Display results
    print("\n✅ Final equity per symbol:")
    for sym, eq in result.final.items():
        print(f"{sym}: ${eq:,.2f}")

    print("\n📈 Equity curve preview:")
    print(result.equity_curve.tail())
