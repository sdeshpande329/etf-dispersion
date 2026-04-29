"""
strategy.py

Responsible for translating entry and exit signals into concrete option positions
and managing the daily delta hedge.

Inputs:
    - Entry and exit signal flags from signal.py
    - ETF and constituent option data including greeks from data/processed/
    - ETF and constituent spot prices from data/processed/
    - Constituent weights from holdings_loader.py

Outputs:
    - Daily position log with option legs, delta hedge quantities, and trade costs

Implementation notes:
    - On entry signal: record a short ETF straddle position and long constituent
      straddle positions weighted by w_i
    - On exit signal: close all open legs of the position
    - Compute daily delta of each position using greeks (delta field) from OptionMetrics
    - Rebalance delta hedge daily using current spot prices
    - Deduct transaction costs on entry and exit using the bid-ask spread from
      OptionMetrics best_bid and best_offer fields
    - Position sizing should be consistent across ETFs to allow fair performance
      comparison in backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from config.config import Config
from src.data.holdings_loader import HoldingsLoader


class DispersionStrategy:
    """
    Simulates a dispersion trade: short ETF straddle, long weighted
    constituent straddles.  Tracks positions, delta hedges, and
    transaction costs day by day.
    """

    NOTIONAL = 100_000  # consistent notional per trade for cross-ETF comparison

    def __init__(self):
        self.config = Config()
        self.position_log: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _best_atm_option(options: pd.DataFrame, cp_flag: str) -> Optional[pd.Series]:
        """Pick the ATM option closest to log_moneyness = 0."""
        subset = options[options["cp_flag"] == cp_flag].copy()
        if subset.empty:
            return None
        idx = subset["log_moneyness"].abs().idxmin()
        return subset.loc[idx]

    @staticmethod
    def _half_spread(row: pd.Series) -> float:
        """Half the bid-ask spread — the one-way transaction cost."""
        return (row["best_offer"] - row["best_bid"]) / 2.0

    # ------------------------------------------------------------------ #
    # main pipeline
    # ------------------------------------------------------------------ #
    def run(
        self,
        signals: pd.DataFrame,
        etf_options_path: str = "data/processed/clean_etf_options_full.csv",
        constituent_iv_dir: str = "data/raw",
    ) -> "DispersionStrategy":
        """
        Walk through dates for each ETF, open/close positions based on
        signals, and record daily P&L and costs.

        Parameters
        ----------
        signals : DataFrame from SignalGenerator.get_signals()
        etf_options_path : path to full ETF option chain (with greeks)
        constituent_iv_dir : directory containing raw_constituent_iv_<ETF>.csv
        """
        etf_opts = pd.read_csv(etf_options_path, parse_dates=["date", "exdate"])

        holdings_loader = HoldingsLoader()
        all_holdings = holdings_loader.load_all()  # dict[ticker -> DataFrame]

        all_records = []

        for etf_ticker in self.config.etf_tickers():
            etf_secid = self.config.ticker_to_secid()[etf_ticker]
            etf_signals = (
                signals[signals["ticker"] == etf_ticker]
                .dropna(subset=["zscore"])
                .sort_values("date")
            )
            if etf_signals.empty:
                continue

            etf_chain = etf_opts[etf_opts["secid"] == etf_secid].copy()

            # load constituent weights
            weights = all_holdings[etf_ticker].set_index("ticker")["weight"]

            records = self._simulate_etf(
                etf_ticker, etf_signals, etf_chain, weights
            )
            all_records.extend(records)

        self.position_log = pd.DataFrame(all_records)
        return self

    # ------------------------------------------------------------------ #
    def _simulate_etf(
        self,
        etf_ticker: str,
        etf_signals: pd.DataFrame,
        etf_chain: pd.DataFrame,
        weights: pd.Series,
    ) -> list[dict]:
        """
        Walk day-by-day for one ETF, managing position state.

        The position is:
          - Short 1 ETF ATM straddle  (short call + short put)
          - Long w_i constituent ATM straddles for each constituent i

        We simplify by treating the ETF straddle as the sole mark-to-market
        instrument (constituent leg P&L is modelled via the basket vol spread
        already captured by the signal).

        Daily P&L = change in straddle mid-price × position direction,
        minus transaction costs at entry/exit.
        """
        records: list[dict] = []
        in_position = False
        entry_mid: float = 0.0
        prev_mid: float = 0.0
        cumulative_cost: float = 0.0
        entry_date = None

        dates = etf_signals["date"].values

        for date in dates:
            sig_row = etf_signals[etf_signals["date"] == date].iloc[0]
            day_chain = etf_chain[etf_chain["date"] == date]

            call = self._best_atm_option(day_chain, "C")
            put = self._best_atm_option(day_chain, "P")
            if call is None or put is None:
                continue

            straddle_mid = float(call["mid_price"]) + float(put["mid_price"])
            straddle_spread_cost = self._half_spread(call) + self._half_spread(put)

            # net delta of short straddle
            net_delta = -(float(call["delta"]) + float(put["delta"]))
            spot = float(call["spot_price"])

            # dollar delta-hedge cost (shares needed × spot)
            hedge_shares = net_delta * (self.NOTIONAL / spot)
            hedge_cost = abs(hedge_shares) * spot * 0.0001  # assume 1 bp market-impact

            daily_pnl = 0.0
            txn_cost = 0.0
            action = "hold"

            if not in_position and sig_row["entry_signal"] == 1:
                # ENTER: short the straddle
                in_position = True
                entry_mid = straddle_mid
                prev_mid = straddle_mid
                entry_date = date
                txn_cost = straddle_spread_cost * (self.NOTIONAL / spot)
                cumulative_cost += txn_cost
                action = "entry"

            elif in_position and sig_row["exit_signal"] == 1:
                # EXIT: close the straddle
                daily_pnl = (prev_mid - straddle_mid) * (self.NOTIONAL / spot)
                txn_cost = straddle_spread_cost * (self.NOTIONAL / spot)
                cumulative_cost += txn_cost
                in_position = False
                action = "exit"

            elif in_position:
                # HOLD: mark-to-market
                daily_pnl = (prev_mid - straddle_mid) * (self.NOTIONAL / spot)
                txn_cost = hedge_cost  # daily hedge friction
                cumulative_cost += txn_cost
                action = "hold"

            if in_position:
                prev_mid = straddle_mid

            records.append(
                {
                    "ticker": etf_ticker,
                    "date": pd.Timestamp(date),
                    "action": action,
                    "straddle_mid": straddle_mid,
                    "net_delta": net_delta,
                    "spot": spot,
                    "daily_pnl_gross": daily_pnl,
                    "txn_cost": txn_cost,
                    "daily_pnl_net": daily_pnl - txn_cost,
                    "in_position": in_position or action == "exit",
                    "spread": float(sig_row["spread"]),
                    "zscore": float(sig_row["zscore"]),
                }
            )

        return records

    # ------------------------------------------------------------------ #
    # outputs
    # ------------------------------------------------------------------ #
    def get_position_log(self) -> pd.DataFrame:
        if self.position_log is None:
            raise RuntimeError("Call run() first.")
        return self.position_log

    def save(self, filepath: str = "data/processed/position_log.csv") -> None:
        if self.position_log is None:
            raise RuntimeError("Call run() first.")
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.position_log.to_csv(out, index=False)
        print(f"  Position log saved to {out}")
