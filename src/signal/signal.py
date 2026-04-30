import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from config.config import Config


class SignalGenerator:
    """Computes the implied correlation spread between ETF market IV and synthetic basket IV, then generates mean-reversion entry/exit signals."""

    def __init__(self, rolling_window: int = Config.SIGNAL_ROLLING_MEAN_WINDOW, entry_zscore: float = Config.SIGNAL_ENTRY_ZSCORE, exit_zscore: float = Config.SIGNAL_EXIT_ZSCORE):
        self.rolling_window = rolling_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.config = Config()
        self.signals: Optional[pd.DataFrame] = None

    def _load_etf_atm_iv(self, atm_path: str) -> pd.DataFrame:
        """Load cleaned ATM ETF options and extract a daily ATM IV per ETF."""
        df = pd.read_csv(atm_path, parse_dates=["date"])
        df["abs_log_m"] = df["log_moneyness"].abs()
        daily_iv = (
            df.sort_values(["ticker", "date", "abs_log_m"])
            .drop_duplicates(subset=["ticker", "date"], keep="first")[
                ["ticker", "date", "impl_volatility"]
            ]
            .rename(columns={"impl_volatility": "sigma_etf"})
        )
        return daily_iv

    def _load_basket_vol(self, basket_path: str, etf_ticker: str) -> pd.DataFrame:
        """Load sigma_basket CSV for one ETF."""
        df = pd.read_csv(basket_path, parse_dates=["date"])
        df["ticker"] = etf_ticker
        return df[["ticker", "date", "sigma_basket"]]

    def generate(self, atm_path: str = "data/processed/clean_etf_options_atm.csv", basket_dir: str = "data/processed") -> "SignalGenerator":
        """Run the full signal generation pipeline for all ETFs."""
        etf_iv = self._load_etf_atm_iv(atm_path)

        basket_frames = []
        for etf_ticker in self.config.etf_tickers():
            bp = Path(basket_dir) / f"sigma_basket_{etf_ticker}.csv"
            if bp.exists():
                basket_frames.append(self._load_basket_vol(str(bp), etf_ticker))
            else:
                print(f"  Warning: {bp} not found, skipping {etf_ticker}")

        if not basket_frames:
            raise FileNotFoundError("No sigma_basket files found.")

        basket_iv = pd.concat(basket_frames, ignore_index=True)

        merged = etf_iv.merge(basket_iv, on=["ticker", "date"], how="inner")
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

        records = []
        for ticker, grp in merged.groupby("ticker"):
            grp = grp.sort_values("date").copy()
            grp["spread"] = grp["sigma_etf"] - grp["sigma_basket"]
            grp["rolling_mean"] = (
                grp["spread"].rolling(self.rolling_window, min_periods=self.rolling_window).mean()
            )
            grp["rolling_std"] = (
                grp["spread"].rolling(self.rolling_window, min_periods=self.rolling_window).std()
            )
            grp["zscore"] = (grp["spread"] - grp["rolling_mean"]) / grp["rolling_std"]

            grp["entry_signal"] = (grp["zscore"] > self.entry_zscore).astype(int)
            grp["exit_signal"] = (grp["zscore"] < self.exit_zscore).astype(int)
            records.append(grp)

        self.signals = pd.concat(records, ignore_index=True)
        return self

    def get_signals(self) -> pd.DataFrame:
        if self.signals is None:
            raise RuntimeError("Call generate() before getting signals.")
        return self.signals

    def save(self, filepath: str = "data/processed/signals.csv") -> None:
        if self.signals is None:
            raise RuntimeError("Call generate() before saving.")
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.signals.to_csv(out, index=False)