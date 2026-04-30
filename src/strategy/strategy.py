import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from config.config import Config
from src.data.holdings_loader import HoldingsLoader


class DispersionStrategy:
    """Simulates a full dispersion trade: short ETF straddle + long weighted constituent straddles.  Both legs are marked-to-market daily with realistic per-name transaction costs on entry and exit."""

    NOTIONAL = 100_000  # consistent notional per trade for cross-ETF comparison

    def __init__(self):
        self.config = Config()
        self.position_log: Optional[pd.DataFrame] = None

    @staticmethod
    def _best_atm_option(options: pd.DataFrame, cp_flag: str) -> Optional[pd.Series]:
        """Pick the ATM option closest to log_moneyness = 0."""
        subset = options[options["cp_flag"] == cp_flag].copy()
        if subset.empty:
            return None
        idx = subset["log_moneyness"].abs().idxmin()
        return subset.loc[idx]

    @staticmethod
    def _best_atm_constituent(options: pd.DataFrame, cp_flag: str) -> Optional[pd.Series]:
        """Pick the ATM constituent option closest to the money by strike/spot ratio."""
        subset = options[options["cp_flag"] == cp_flag].copy()
        if subset.empty:
            return None
        subset = subset.copy()
        subset["atm_dist"] = (subset["delta"].abs() - 0.5).abs()
        idx = subset["atm_dist"].idxmin()
        return subset.loc[idx]

    @staticmethod
    def _half_spread(row: pd.Series) -> float:
        """Half the bid-ask spread — the one-way transaction cost."""
        return (row["best_offer"] - row["best_bid"]) / 2.0

    def _load_constituent_chains(self, etf_ticker: str, constituent_iv_dir: str, mapping: pd.DataFrame) -> pd.DataFrame:
        """Load constituent option chains with ticker labels and mid prices. Filter to 30-60 DTE for consistency with ETF leg."""
        iv_path = Path(constituent_iv_dir) / f"raw_constituent_iv_{etf_ticker}.csv"
        if not iv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(iv_path, parse_dates=["date", "exdate"])
        df.columns = [c.strip().lower() for c in df.columns]

        if "ticker" not in df.columns and not mapping.empty:
            df = df.merge(
                mapping[["ticker", "secid"]].dropna().drop_duplicates(),
                on="secid",
                how="left",
            )
        df = df.dropna(subset=["ticker"])
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

        df["days_to_expiry"] = (df["exdate"] - df["date"]).dt.days
        df = df[
            (df["days_to_expiry"] >= self.config.DAYS_TO_EXPIRY_MIN)
            & (df["days_to_expiry"] <= self.config.DAYS_TO_EXPIRY_MAX)
        ].copy()

        df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2.0
        df = df[df["mid_price"] > 0]

        return df

    def run(self, signals: pd.DataFrame, etf_options_path: str = "data/processed/clean_etf_options_full.csv", constituent_iv_dir: str = "data/raw") -> "DispersionStrategy":
        """Walk through dates for each ETF, open/close positions based on signals, and record daily P&L and costs.

        Both legs of the dispersion trade are simulated:
        - Short ETF ATM straddle
        - Long weighted constituent ATM straddles

        Transaction costs are charged on each individual leg at entry/exit.
        """
        etf_opts = pd.read_csv(etf_options_path, parse_dates=["date", "exdate"])

        holdings_loader = HoldingsLoader()
        all_holdings = holdings_loader.load_all()

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

            weights = all_holdings[etf_ticker].set_index("ticker")["weight"]

            mapping_path = Path(constituent_iv_dir) / f"raw_constituent_mapping_{etf_ticker}.csv"
            mapping = pd.DataFrame()
            if mapping_path.exists():
                mapping = pd.read_csv(mapping_path)
                mapping.columns = [c.strip().lower() for c in mapping.columns]
                mapping["ticker"] = mapping["ticker"].astype(str).str.strip().str.upper()

            constituent_chain = self._load_constituent_chains(
                etf_ticker, constituent_iv_dir, mapping
            )

            records = self._simulate_etf(
                etf_ticker, etf_signals, etf_chain, weights, constituent_chain
            )
            all_records.extend(records)

        self.position_log = pd.DataFrame(all_records)
        return self

    def _simulate_etf(self, etf_ticker: str, etf_signals: pd.DataFrame, etf_chain: pd.DataFrame, weights: pd.Series, constituent_chain: pd.DataFrame) -> list[dict]:
        """Walk day-by-day for one ETF, managing position state.

        The position is:
          - Short 1 ETF ATM straddle  (short call + short put)
          - Long w_i constituent ATM straddles for each constituent i

        Both legs are marked-to-market daily. Transaction costs are charged per-name on the constituent leg at entry and exit.
        """
        has_constituents = not constituent_chain.empty
        records: list[dict] = []
        in_position = False

        prev_etf_straddle: float = 0.0
        entry_spot: float = 0.0  # fixed at entry for position sizing
        position_scale: float = 0.0  # N / S_entry, fixed at entry

        # Constituent leg state: dict[ticker -> prev_straddle_mid]
        prev_constituent_straddles: dict[str, float] = {}
        constituent_weights_active: dict[str, float] = {}

        dates = etf_signals["date"].values

        for date in dates:
            sig_row = etf_signals[etf_signals["date"] == date].iloc[0]
            day_etf_chain = etf_chain[etf_chain["date"] == date]

            call = self._best_atm_option(day_etf_chain, "C")
            put = self._best_atm_option(day_etf_chain, "P")
            if call is None or put is None:
                continue

            etf_straddle_mid = float(call["mid_price"]) + float(put["mid_price"])
            etf_spread_cost = self._half_spread(call) + self._half_spread(put)
            spot = float(call["spot_price"])

            # net delta of short ETF straddle
            etf_net_delta = -(float(call["delta"]) + float(put["delta"]))

            day_const_chain = pd.DataFrame()
            if has_constituents:
                day_const_chain = constituent_chain[constituent_chain["date"] == date]

            constituent_straddles: dict[str, float] = {}
            constituent_spread_costs: dict[str, float] = {}
            constituent_deltas: dict[str, float] = {}

            if not day_const_chain.empty:
                for ticker_c in weights.index:
                    tc_chain = day_const_chain[day_const_chain["ticker"] == ticker_c]
                    if tc_chain.empty:
                        continue
                    c_call = self._best_atm_constituent(tc_chain, "C")
                    c_put = self._best_atm_constituent(tc_chain, "P")
                    if c_call is None or c_put is None:
                        continue
                    constituent_straddles[ticker_c] = float(c_call["mid_price"]) + float(c_put["mid_price"])
                    constituent_spread_costs[ticker_c] = self._half_spread(c_call) + self._half_spread(c_put)
                    constituent_deltas[ticker_c] = float(c_call["delta"]) + float(c_put["delta"])

            hedge_shares = etf_net_delta * position_scale if in_position else 0.0
            hedge_cost = abs(hedge_shares) * spot * 0.0001

            etf_pnl = 0.0
            constituent_pnl = 0.0
            txn_cost = 0.0
            action = "hold"

            if not in_position and sig_row["entry_signal"] == 1:
                in_position = True
                entry_spot = spot
                position_scale = self.NOTIONAL / entry_spot  # fixed for this trade

                prev_etf_straddle = etf_straddle_mid
                prev_constituent_straddles = {}
                constituent_weights_active = {}

                txn_cost = etf_spread_cost * position_scale

                for ticker_c, w_c in weights.items():
                    if ticker_c in constituent_straddles:
                        prev_constituent_straddles[ticker_c] = constituent_straddles[ticker_c]
                        constituent_weights_active[ticker_c] = w_c
                        txn_cost += constituent_spread_costs[ticker_c] * w_c * position_scale

                action = "entry"

            elif in_position and sig_row["exit_signal"] == 1:
                # ETF leg P&L (short straddle: profit when price drops)
                etf_pnl = (prev_etf_straddle - etf_straddle_mid) * position_scale

                # Constituent leg P&L (long straddles: profit when price rises)
                for ticker_c, prev_mid_c in prev_constituent_straddles.items():
                    w_c = constituent_weights_active[ticker_c]
                    curr_mid_c = constituent_straddles.get(ticker_c)
                    if curr_mid_c is not None:
                        constituent_pnl += (curr_mid_c - prev_mid_c) * w_c * position_scale

                # Transaction costs at exit
                txn_cost = etf_spread_cost * position_scale
                for ticker_c, w_c in constituent_weights_active.items():
                    if ticker_c in constituent_spread_costs:
                        txn_cost += constituent_spread_costs[ticker_c] * w_c * position_scale

                in_position = False
                action = "exit"

            elif in_position:
                # ETF leg
                etf_pnl = (prev_etf_straddle - etf_straddle_mid) * position_scale
                prev_etf_straddle = etf_straddle_mid

                # Constituent leg
                new_prev = {}
                for ticker_c, prev_mid_c in prev_constituent_straddles.items():
                    w_c = constituent_weights_active[ticker_c]
                    curr_mid_c = constituent_straddles.get(ticker_c)
                    if curr_mid_c is not None:
                        constituent_pnl += (curr_mid_c - prev_mid_c) * w_c * position_scale
                        new_prev[ticker_c] = curr_mid_c
                    else:
                        # constituent option missing today, carry forward
                        new_prev[ticker_c] = prev_mid_c
                prev_constituent_straddles = new_prev

                txn_cost = hedge_cost
                action = "hold"

            daily_pnl_gross = etf_pnl + constituent_pnl
            n_constituents = len(constituent_weights_active) if in_position or action == "exit" else 0

            records.append(
                {
                    "ticker": etf_ticker,
                    "date": pd.Timestamp(date),
                    "action": action,
                    "etf_straddle_mid": etf_straddle_mid,
                    "etf_pnl": etf_pnl,
                    "constituent_pnl": constituent_pnl,
                    "n_constituents": n_constituents,
                    "net_delta": etf_net_delta,
                    "spot": spot,
                    "daily_pnl_gross": daily_pnl_gross,
                    "txn_cost": txn_cost,
                    "daily_pnl_net": daily_pnl_gross - txn_cost,
                    "in_position": in_position or action == "exit",
                    "spread": float(sig_row["spread"]),
                    "zscore": float(sig_row["zscore"]),
                }
            )

        return records

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