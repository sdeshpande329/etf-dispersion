import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from config.config import Config


class Backtester:
    """Takes the position log from DispersionStrategy and computes performance analytics: cumulative PnL, Sharpe, drawdown, and a cross-ETF comparison table."""

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self):
        self.config = Config()
        self.pnl_series: Optional[pd.DataFrame] = None
        self.summary: Optional[pd.DataFrame] = None

    @staticmethod
    def _sharpe(daily_pnl: pd.Series, rf_daily: float = 0.0) -> float:
        """Annualised Sharpe ratio from a daily PnL series."""
        excess = daily_pnl - rf_daily
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(252))

    @staticmethod
    def _max_drawdown(cum_pnl: pd.Series) -> float:
        """Max peak-to-trough drawdown on a cumulative PnL series."""
        peak = cum_pnl.cummax()
        dd = cum_pnl - peak
        return float(dd.min())

    @staticmethod
    def _avg_rf_daily(rates_path: str = "data/raw/raw_rates.csv") -> float:
        """Average annualised risk-free rate converted to daily."""
        path = Path(rates_path)
        if not path.exists():
            return 0.0
        rates = pd.read_csv(path)
        avg_annual = rates["rate"].mean() / 100.0  # rates stored as pct
        return avg_annual / 252

    def run(self, position_log: pd.DataFrame, rates_path: str = "data/raw/raw_rates.csv") -> "Backtester":
        """Compute cumulative PnL and summary statistics from the position log produced by DispersionStrategy."""
        rf_daily = self._avg_rf_daily(rates_path)

        pnl_frames = []
        summary_rows = []

        tier_map = self.config.ticker_to_tier()

        for ticker, grp in position_log.groupby("ticker"):
            grp = grp.sort_values("date").copy()

            grp["cum_pnl_gross"] = grp["daily_pnl_gross"].cumsum()
            grp["cum_pnl_net"] = grp["daily_pnl_net"].cumsum()
            grp["cum_txn_cost"] = grp["txn_cost"].cumsum()

            pnl_frames.append(
                grp[
                    [
                        "ticker",
                        "date",
                        "daily_pnl_gross",
                        "daily_pnl_net",
                        "txn_cost",
                        "cum_pnl_gross",
                        "cum_pnl_net",
                        "cum_txn_cost",
                        "in_position",
                        "spread",
                        "zscore",
                    ]
                ]
            )

            all_pnl = grp["daily_pnl_net"]  # includes 0 on inactive days
            sharpe_net = self._sharpe(all_pnl, rf_daily) if len(all_pnl) > 1 else 0.0
            mdd_net = self._max_drawdown(grp["cum_pnl_net"])

            active = grp[grp["in_position"]]

            grp_2022 = grp[grp["date"].dt.year == 2022]
            grp_2324 = grp[grp["date"].dt.year >= 2023]

            sharpe_2022 = self._sharpe(grp_2022["daily_pnl_net"], rf_daily) if len(grp_2022) > 1 else np.nan
            sharpe_2324 = self._sharpe(grp_2324["daily_pnl_net"], rf_daily) if len(grp_2324) > 1 else np.nan

            n_trades = int((grp["action"] == "entry").sum()) if "action" in grp.columns else 0

            summary_rows.append(
                {
                    "ticker": ticker,
                    "liquidity_tier": tier_map.get(ticker, None),
                    "n_trades": n_trades,
                    "n_active_days": len(active),
                    "avg_spread": float(active["spread"].mean()) if len(active) else np.nan,
                    "spread_std": float(active["spread"].std()) if len(active) else np.nan,
                    "total_pnl_gross": float(grp["cum_pnl_gross"].iloc[-1]) if len(grp) else 0.0,
                    "total_pnl_net": float(grp["cum_pnl_net"].iloc[-1]) if len(grp) else 0.0,
                    "total_txn_cost": float(grp["cum_txn_cost"].iloc[-1]) if len(grp) else 0.0,
                    "sharpe_net": sharpe_net,
                    "max_drawdown": mdd_net,
                    "sharpe_2022": sharpe_2022,
                    "sharpe_2023_24": sharpe_2324,
                }
            )

        self.pnl_series = pd.concat(pnl_frames, ignore_index=True)
        self.summary = pd.DataFrame(summary_rows)
        return self

    def get_pnl_series(self) -> pd.DataFrame:
        if self.pnl_series is None:
            raise RuntimeError("Call run() first.")
        return self.pnl_series

    def get_summary(self) -> pd.DataFrame:
        if self.summary is None:
            raise RuntimeError("Call run() first.")
        return self.summary

    def print_summary(self) -> None:
        """Pretty-print the cross-ETF summary table."""
        if self.summary is None:
            raise RuntimeError("Call run() first.")
        print("DISPERSION STRATEGY — CROSS-ETF PERFORMANCE SUMMARY")
        display_cols = [
            "ticker",
            "liquidity_tier",
            "n_trades",
            "avg_spread",
            "total_pnl_net",
            "total_txn_cost",
            "sharpe_net",
            "max_drawdown",
            "sharpe_2022",
            "sharpe_2023_24",
        ]
        cols = [c for c in display_cols if c in self.summary.columns]
        print(self.summary[cols].to_string(index=False, float_format="{:.4f}".format))
    
    def save(self, pnl_path: str = "data/processed/backtest_pnl.csv", summary_path: str = "data/processed/backtest_summary.csv") -> None:
        if self.pnl_series is None:
            raise RuntimeError("Call run() first.")
        for path, df in [(pnl_path, self.pnl_series), (summary_path, self.summary)]:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)