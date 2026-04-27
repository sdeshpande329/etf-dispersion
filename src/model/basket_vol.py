"""
basket_vol.py

Responsible for deriving the theoretical implied volatility of each ETF basket
from its constituent implied volatilities and pairwise correlations.

Inputs:
    - Constituent ATM implied vols from OptionMetrics (loaded from data/raw/)
    - Constituent weights from holdings_loader.py
    - Rolling correlation matrices from correlation.py

Outputs:
    - Daily time series of sigma_basket per ETF
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from config.config import Config
from src.data.holdings_loader import HoldingsLoader
from src.model.correlation import CorrelationEstimator


class BasketVolatilityCalculator:
    """
    Computes the theoretical implied volatility of an ETF basket using
    constituent implied vols, weights, and pairwise correlations.
    """
    
    def __init__(self, etf_ticker: str):
        """
        Initialize the calculator for a specific ETF.
        """
        self.etf_ticker = etf_ticker
        self.config = Config()
        
        # Data containers - these will be populated by load_* methods
        self.constituent_iv: Optional[pd.DataFrame] = None
        self.weights: Optional[pd.Series] = None
        self.correlation_estimator = None
        
        # Output container
        self.sigma_basket_df: Optional[pd.DataFrame] = None
    
    # Data Loading
    def load_constituent_iv(self, iv_data: pd.DataFrame) -> "BasketVolatilityCalculator":
        """
        Load constituent implied volatility data.
        """
        required_cols = {'date', 'exdate', 'strike_price', 'impl_volatility', 'log_moneyness'}
        missing_cols = required_cols - set(iv_data.columns)
        if missing_cols:
            raise ValueError(f"Input iv_data is missing required columns: {missing_cols}")

        iv_data = iv_data.copy()
        iv_data['date'] = pd.to_datetime(iv_data['date'])
        iv_data['exdate'] = pd.to_datetime(iv_data['exdate'])
        if 'days_to_expiry' not in iv_data.columns:
            iv_data['days_to_expiry'] = (iv_data['exdate'] - iv_data['date']).dt.days

        # Make sure options are At the Money and within the expiry window to get best imp vol estimate
        self.constituent_iv = iv_data[
            (iv_data['log_moneyness'].abs() < self.config.ATM_THRESHOLD) &
            (iv_data['days_to_expiry'] >= self.config.DAYS_TO_EXPIRY_MIN) &
            (iv_data['days_to_expiry'] <= self.config.DAYS_TO_EXPIRY_MAX)
        ].copy()
        return self
    
    def load_weights(self, weights: pd.Series) -> "BasketVolatilityCalculator":
        """
        Load constituent weights from holdings data.
        """
        
        # Validate that weights sum to 1 
        TOL = 0.01
        total_weight = weights.sum()
        if abs(1 - total_weight) > TOL:
            print(f" Warning: Weights sum to {total_weight:.4f}, which is outside the tolerance of 1.0 ± {TOL}")
        
        self.weights = weights
        return self
    
    def load_correlation_estimator(self, corr_estimator) -> "BasketVolatilityCalculator":
        """
        Load a pre-fitted CorrelationEstimator
        """

        if not hasattr(corr_estimator, 'get_tickers') or not hasattr(corr_estimator, 'get_all_matrices'):
            raise ValueError("CorrelationEstimator must have estimate() called before loading.")
        corr_matrices = corr_estimator.get_all_matrices()
        if not corr_matrices:
            raise ValueError("CorrelationEstimator must have estimate() called before loading.")
        corr_tickers = set(corr_estimator.get_tickers())
        weight_tickers = set(self.weights.index)
        
        missing = weight_tickers - corr_tickers
        overlap = weight_tickers & corr_tickers
        
        if missing:
            print(f" Warning: CorrelationEstimator is missing tickers present in weights: {missing}")
        
        if not overlap:
            raise ValueError("No overlap found between weights and correlation estimator tickers.")

        self.correlation_estimator = corr_estimator
        return self
    
    def _get_atm_vol(self, date: pd.Timestamp, constituent_ticker: str) -> Optional[float]:
        """
        Get the ATM implied volatility for a specific constituent on a given date.
        """
        if 'ticker' not in self.constituent_iv.columns:
            raise ValueError("constituent_iv must include a 'ticker' column.")

        iv_slice = self.constituent_iv.loc[
            (self.constituent_iv['date'] == date) &
            (self.constituent_iv['ticker'] == constituent_ticker)
        ]
        if iv_slice.empty:
            return None

        atm_idx = iv_slice['log_moneyness'].abs().idxmin()
        return float(iv_slice.at[atm_idx, 'impl_volatility'])
    
    def _compute_basket_variance(
        self, 
        date: pd.Timestamp, 
        weights: np.ndarray, 
        vols: np.ndarray, 
        corr_matrix: np.ndarray
    ) -> float:
        """
        Compute basket variance using the full correlation matrix.
        """
        weighted_vols = weights * vols
        return float(weighted_vols @ corr_matrix @ weighted_vols)
    
    def compute(self) -> "BasketVolatilityCalculator":
        """
        Run the full basket volatility computation for all dates.
        """
        if self.constituent_iv is None:
            raise RuntimeError("Call load_constituent_iv() before compute().")
        if self.weights is None:
            raise RuntimeError("Call load_weights() before compute().")
        if self.correlation_estimator is None:
            raise RuntimeError("Call load_correlation_estimator() before compute().")
        if 'ticker' not in self.constituent_iv.columns:
            raise ValueError("constituent_iv must include a 'ticker' column before compute().")

        dates = pd.Index(self.constituent_iv['date']).unique().sort_values()
        corr_tickers = pd.Index(self.correlation_estimator.get_tickers())
        aligned_weights = self.weights.reindex(corr_tickers)
        valid_weight_mask = aligned_weights.notna().to_numpy()

        atm_vols = (
            self.constituent_iv.assign(abs_log_moneyness=self.constituent_iv['log_moneyness'].abs())
            .sort_values(['date', 'ticker', 'abs_log_moneyness'])
            .drop_duplicates(subset=['date', 'ticker'], keep='first')
            .pivot(index='date', columns='ticker', values='impl_volatility')
            .reindex(columns=corr_tickers)
            .sort_index()
            .ffill()
        )

        corr_matrices = self.correlation_estimator.get_all_matrices()
        common_dates = dates.intersection(pd.Index(corr_matrices.keys())).sort_values()
        weight_values = aligned_weights.to_numpy(dtype=float, copy=False)
        records = []

        for date in common_dates:
            corr_matrix = corr_matrices[date]
            vol_values = atm_vols.loc[date].to_numpy(dtype=float, copy=False)

            corr_valid_mask = ~np.isnan(np.diag(corr_matrix))
            usable_mask = valid_weight_mask & corr_valid_mask & ~np.isnan(vol_values)
            n_constituents = int(usable_mask.sum())

            sigma_basket = np.nan
            if n_constituents >= 2:
                sub_weights = weight_values[usable_mask]
                sub_weights = sub_weights / sub_weights.sum()
                sub_vols = vol_values[usable_mask]
                sub_corr = corr_matrix[np.ix_(usable_mask, usable_mask)]
                basket_variance = self._compute_basket_variance(date, sub_weights, sub_vols, sub_corr)
                sigma_basket = float(np.sqrt(max(basket_variance, 0.0)))

            records.append({
                'date': date,
                'sigma_basket': sigma_basket,
                'n_constituents': n_constituents,
                'missing_constituents': int(len(corr_tickers) - n_constituents),
            })

        self.sigma_basket_df = pd.DataFrame.from_records(records)
        return self
    
    # Outputs and Saving
    def get_sigma_basket(self) -> pd.DataFrame:
        """
        Get the computed sigma_basket time series.
        """
        if self.sigma_basket_df is None:
            raise RuntimeError("Call compute() before getting results.")
        return self.sigma_basket_df
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save the sigma_basket results to CSV.
        """
        if self.sigma_basket_df is None:
            raise RuntimeError("No results to save. Call compute() first.")

        output_path = Path(filepath) if filepath is not None else (
            Path(self.config.PROCESSED_DIR) / f"sigma_basket_{self.etf_ticker}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.sigma_basket_df.to_csv(output_path, index=False)


def _load_constituent_mapping(etf_ticker: str, search_dirs: list[str]) -> pd.DataFrame:
    """
    Load an optional constituent mapping CSV with columns including ticker and permno/secid.
    """
    for directory in search_dirs:
        for filename in (
            f"raw_constituent_mapping_{etf_ticker}.csv",
            f"constituent_mapping_{etf_ticker}.csv",
        ):
            mapping_path = Path(directory) / filename
            if mapping_path.exists():
                mapping_df = pd.read_csv(mapping_path)
                mapping_df.columns = [col.strip().lower() for col in mapping_df.columns]
                if 'ticker' not in mapping_df.columns:
                    raise ValueError(f"{mapping_path} must contain a 'ticker' column.")
                mapping_df['ticker'] = mapping_df['ticker'].astype(str).str.strip().str.upper()
                return mapping_df
    return pd.DataFrame()


def _prepare_returns_matrix(returns_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize returns into a wide date x ticker matrix.
    """
    returns_df = returns_df.copy()
    returns_df.columns = [col.strip().lower() for col in returns_df.columns]
    returns_df['date'] = pd.to_datetime(returns_df['date'])

    if 'ticker' not in returns_df.columns:
        if 'permno' not in returns_df.columns or mapping_df.empty or 'permno' not in mapping_df.columns:
            raise ValueError(
                "Returns data must include a 'ticker' column or data/raw/raw_constituent_mapping_<ETF>.csv "
                "with 'ticker' and 'permno' columns. Re-run scripts/download_data.py to generate it."
            )
        returns_df = returns_df.merge(
            mapping_df[['ticker', 'permno']].dropna().drop_duplicates(),
            on='permno',
            how='left',
        )

    returns_df['ticker'] = returns_df['ticker'].astype(str).str.strip().str.upper()
    return returns_df.pivot_table(index='date', columns='ticker', values='ret', aggfunc='first').sort_index()


def _prepare_constituent_iv(iv_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize constituent IV data to include ticker labels.
    """
    iv_df = iv_df.copy()
    iv_df.columns = [col.strip().lower() for col in iv_df.columns]
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    iv_df['exdate'] = pd.to_datetime(iv_df['exdate'])

    if 'ticker' not in iv_df.columns:
        if 'secid' not in iv_df.columns or mapping_df.empty or 'secid' not in mapping_df.columns:
            raise ValueError(
                "IV data must include a 'ticker' column or data/raw/raw_constituent_mapping_<ETF>.csv "
                "with 'ticker' and 'secid' columns. Re-run scripts/download_data.py to generate it."
            )
        iv_df = iv_df.merge(
            mapping_df[['ticker', 'secid']].dropna().drop_duplicates(),
            on='secid',
            how='left',
        )

    iv_df['ticker'] = iv_df['ticker'].astype(str).str.strip().str.upper()
    return iv_df.dropna(subset=['ticker'])


def compute_all_etfs(
    iv_data_dir: str = "data/raw",
    weights_dir: str = "data/holdings",
    output_dir: str = "data/processed"
) -> dict:
    """
    Compute basket volatility for all ETFs in the universe.
    
    This is a convenience function that:
    1. Load holdings for each ETF
    2. Load the saved constituent ticker-to-id mapping
    3. Load correlation estimator for each ETF
    4. Compute sigma_basket for each ETF
    5. Save results to CSV
    """
    config = Config()
    holdings_loader = HoldingsLoader()
    holdings_loader.holdings_dir = Path(weights_dir)
    results = {}

    for etf_ticker in config.etf_tickers():
        weights_df = holdings_loader.load(etf_ticker)
        weights = weights_df.set_index('ticker')['weight'].astype(float)

        mapping_df = _load_constituent_mapping(etf_ticker, [iv_data_dir, weights_dir, output_dir])

        returns_path = Path(iv_data_dir) / f"raw_constituent_returns_{etf_ticker}.csv"
        returns_raw = pd.read_csv(returns_path)
        returns_matrix = _prepare_returns_matrix(returns_raw, mapping_df)

        corr_estimator = CorrelationEstimator(
            etf_name=etf_ticker,
            window=config.ROLLING_CORR_WINDOW,
            max_constituents=config.CONSTITUENT_CAPS.get(etf_ticker, 30),
        )
        corr_estimator.load_weights(weights).load_returns(returns_matrix).estimate()

        iv_path = Path(iv_data_dir) / f"raw_constituent_iv_{etf_ticker}.csv"
        iv_raw = pd.read_csv(iv_path)
        iv_data = _prepare_constituent_iv(iv_raw, mapping_df)

        calculator = BasketVolatilityCalculator(etf_ticker)
        calculator.load_weights(weights)
        calculator.load_correlation_estimator(corr_estimator)
        calculator.load_constituent_iv(iv_data)
        calculator.compute()
        calculator.save(str(Path(output_dir) / f"sigma_basket_{etf_ticker}.csv"))

        results[etf_ticker] = calculator.get_sigma_basket()

    return results
