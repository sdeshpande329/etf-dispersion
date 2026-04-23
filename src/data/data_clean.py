import pandas as pd
import numpy as np
from pathlib import Path
from config.config import Config


class DataCleaner:
    """Merges and cleans options, spot, and rate data for ETFs."""

    ATM_THRESHOLD = 0.025  # |log_moneyness| < 0.025 is considered ATM

    def __init__(self):
        self.config = Config()

    def merge_options_and_spot(self, options_df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
        """Merges options data with ETF spot prices on (secid, date)."""
        merged = options_df.merge(
            spot_df[["secid", "date", "spot_price"]],
            on=["secid", "date"],
            how="left"
        )
        missing_spot = merged["spot_price"].isna().sum()
        if missing_spot > 0:
            print(f"[DataCleaner] Warning: {missing_spot} rows missing spot price after merge")
        return merged

    def merge_rates(self, df: pd.DataFrame, rate_df: pd.DataFrame) -> pd.DataFrame:
        """Merges in the risk-free rate to df"""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["exdate"] = pd.to_datetime(df["exdate"])
        df["days_to_expiry"] = (df["exdate"] - df["date"]).dt.days

        rate_df = rate_df.copy()
        rate_df["date"] = pd.to_datetime(rate_df["date"])

        # For each row, find the closest 'days' in the zero curve on that date
        rate_pivot = rate_df.pivot_table(index="date", columns="days", values="rate")
        available_days = rate_pivot.columns.values

        def match_rate(row):
            if row["date"] not in rate_pivot.index:
                return np.nan
            closest_days = available_days[
                np.argmin(np.abs(available_days - row["days_to_expiry"]))
            ]
            return rate_pivot.loc[row["date"], closest_days]

        df["risk_free_rate"] = df.apply(match_rate, axis=1)
        missing_rate = df["risk_free_rate"].isna().sum()
        if missing_rate > 0:
            print(f"[DataCleaner] Warning: {missing_rate} rows missing risk-free rate")
        return df

    def compute_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes mid_price, spread, spread_pct, log_moneyness, and tau."""
        df = df.copy()
        df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2.0
        df["spread"] = df["best_offer"] - df["best_bid"]
        df["spread_pct"] = df["spread"] / df["mid_price"]
        df["log_moneyness"] = np.log(df["strike_price"] / df["spot_price"])
        df["tau"] = (df["exdate"] - df["date"]).dt.days / 365.0
        return df

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies log-moneyness band filter and spread_pct filter. Also drops rows with missing spot price or mid_price <= MIN_OPTION_PRICE."""
        n_before = len(df)
        df = df.dropna(subset=["spot_price", "mid_price", "impl_volatility"])
        df = df[
            (df["log_moneyness"].abs() <= self.config.LOG_MONEYNESS_BAND) &
            (df["spread_pct"] <= self.config.MAX_SPREAD_PCT) &
            (df["mid_price"] >= self.config.MIN_OPTION_PRICE)
        ]
        return df.reset_index(drop=True)

    def filter_atm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters to ATM options only, defined as |log_moneyness| < ATM_THRESHOLD. Used to construct the primary implied correlation spread signal."""
        atm = df[df["log_moneyness"].abs() < self.ATM_THRESHOLD].copy()
        return atm.reset_index(drop=True)

    def clean(self, options_df: pd.DataFrame, spot_df: pd.DataFrame, rate_df: pd.DataFrame, atm_only: bool = False) -> pd.DataFrame:
        """Full cleaning pipeline. Returns a cleaned DataFrame. If atm_only=True, also applies the ATM filter after all other steps."""
        df = self.merge_options_and_spot(options_df, spot_df)
        df = self.merge_rates(df, rate_df)
        df = self.compute_derived_fields(df)
        df = self.apply_filters(df)
        if atm_only:
            df = self.filter_atm(df)
        return df

    def save(self, df: pd.DataFrame, filename: str) -> None:
        path = Path(self.config.PROCESSED_DIR) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)