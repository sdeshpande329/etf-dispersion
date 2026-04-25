import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from pathlib import Path
from typing import Optional


class CorrelationEstimator:
    """Estimates rolling pairwise correlation matrices for ETF constituents using log returns from CRSP daily security file data."""
    def __init__(self, etf_name: str, window: int = 60, max_constituents: int = 50):
        self.etf_name = etf_name
        self.window = window
        self.max_constituents = max_constituents

        self.returns_df: Optional[pd.DataFrame] = None
        self.weights_series: Optional[pd.Series] = None

        self._corr_matrices: dict[pd.Timestamp, np.ndarray] = {}
        self._tickers_used: Optional[list[str]] = None
        self._diagnostics: list[dict] = []

    
    def load_returns(self, returns_df: pd.DataFrame) -> "CorrelationEstimator":
        """Load the constituent log-return matrix."""
        if not isinstance(returns_df.index, pd.DatetimeIndex):
            returns_df.index = pd.to_datetime(returns_df.index)

        self.returns_df = returns_df.sort_index()
        return self

    def load_weights(self, weights_series: pd.Series) -> "CorrelationEstimator":
        """Load constituent portfolio weights."""
        self.weights_series = weights_series
        return self

    
    def _select_constituents(self) -> list[str]:
        """Intersect the return matrix columns with the weights index, then retain the top-N by weight."""
        if self.returns_df is None:
            raise RuntimeError("Call load_returns() before selecting constituents.")
        if self.weights_series is None:
            raise RuntimeError("Call load_weights() before selecting constituents.")

        available = set(self.returns_df.columns)
        weighted = set(self.weights_series.index)
        overlap = available & weighted

        dropped = weighted - available
        if dropped:
            print(
                f"[{self.etf_name}] Warning: {len(dropped)} tickers in holdings "
                f"have no CRSP return data and will be excluded: "
                f"{sorted(dropped)[:10]}{'...' if len(dropped) > 10 else ''}"
            )

        overlap_weights = self.weights_series.loc[list(overlap)].sort_values(ascending=False)
        selected = list(overlap_weights.head(self.max_constituents).index)

        return selected

    def estimate(self) -> "CorrelationEstimator":
        """Compute a rolling Ledoit-Wolf shrinkage correlation matrix for every date in the return matrix that has at least self.window prior observations."""
        tickers = self._select_constituents()
        self._tickers_used = tickers

        ret = self.returns_df[tickers].copy()
        dates = ret.index
        n_dates = len(dates)

        
        for i in range(self.window - 1, n_dates):
            t = dates[i]
            window_ret = ret.iloc[i - self.window + 1 : i + 1]

            valid_cols = window_ret.columns[window_ret.notna().all(axis=0)]
            n_dropped = len(tickers) - len(valid_cols)
            window_ret = window_ret[valid_cols].values  # (window x n_valid) numpy array

            n_valid = window_ret.shape[1]
            if n_valid < 2:
                self._diagnostics.append(
                    {"date": t, "n_valid": n_valid, "skipped": True, "alpha": None}
                )
                continue

            window_ret = window_ret - window_ret.mean(axis=0)

            lw = LedoitWolf(assume_centered=True)
            lw.fit(window_ret)

            cov_lw = lw.covariance_       
            alpha = lw.shrinkage_  

            std_vec = np.sqrt(np.diag(cov_lw))          
            std_vec = np.where(std_vec > 0, std_vec, np.nan)
            outer_std = np.outer(std_vec, std_vec)       
            corr_lw = cov_lw / outer_std

            np.fill_diagonal(corr_lw, 1.0)
            corr_lw = np.clip(corr_lw, -1.0, 1.0)
            np.fill_diagonal(corr_lw, 1.0)

            full_corr = np.full((len(tickers), len(tickers)), np.nan)
            valid_idx = [tickers.index(c) for c in valid_cols]
            for row_pos, row_idx in enumerate(valid_idx):
                for col_pos, col_idx in enumerate(valid_idx):
                    full_corr[row_idx, col_idx] = corr_lw[row_pos, col_pos]

            self._corr_matrices[t] = full_corr

            off_diag = corr_lw[np.triu_indices(n_valid, k=1)]
            self._diagnostics.append({
                "date": t,
                "n_valid": n_valid,
                "n_dropped": n_dropped,
                "skipped": False,
                "alpha": round(float(alpha), 6),
                "mean_offdiag_corr": round(float(np.nanmean(off_diag)), 6),
                "min_offdiag_corr": round(float(np.nanmin(off_diag)), 6),
            })

        n_computed = sum(1 for d in self._diagnostics if not d.get("skipped", True))
        return self

    
    def get_matrix(self, date: pd.Timestamp) -> np.ndarray:
        """Retrieve the correlation matrix for a specific date."""
        if date not in self._corr_matrices:
            raise KeyError(
                f"No correlation matrix for date {date}. "
                f"Available range: {self.date_range()}"
            )
        return self._corr_matrices[date]

    def get_all_matrices(self) -> dict[pd.Timestamp, np.ndarray]:
        """Return the full dictionary of {date: corr_matrix}."""
        return self._corr_matrices

    def get_tickers(self) -> list[str]:
        """Return the ordered list of tickers corresponding to matrix rows/columns."""
        if self._tickers_used is None:
            raise RuntimeError("Call estimate() before retrieving tickers.")
        return self._tickers_used

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the first and last date for which a matrix was computed."""
        if not self._corr_matrices:
            raise RuntimeError("No matrices computed yet. Call estimate() first.")
        dates = sorted(self._corr_matrices.keys())
        return dates[0], dates[-1]

    def get_diagnostics(self) -> pd.DataFrame:
        """Return per-date diagnostics as a DataFrame."""
        return pd.DataFrame(self._diagnostics)

    def save_diagnostics(self, output_dir: str = "data/raw") -> None:
        """Write per-date diagnostics to a CSV file for inspection."""
        path = Path(output_dir) / f"diagnostics_correlation_{self.etf_name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.get_diagnostics().to_csv(path, index=False)
        print(f"[{self.etf_name}] Diagnostics saved to {path}")

    def save_mean_correlation_series(self, output_dir: str = "data/processed") -> pd.Series:
        """Compute and save the daily mean pairwise (implied) correlation across all constituent pairs, across all dates."""
        records = []
        for date, matrix in sorted(self._corr_matrices.items()):
            n = matrix.shape[0]
            if n < 2:
                continue
            upper_triangle = matrix[np.triu_indices(n, k=1)]
            valid = upper_triangle[~np.isnan(upper_triangle)]
            mean_corr = float(np.mean(valid)) if len(valid) > 0 else np.nan
            records.append({"date": date, "mean_pairwise_correlation": mean_corr})

        series_df = pd.DataFrame(records).set_index("date")
        mean_corr_series = series_df["mean_pairwise_correlation"]

        path = Path(output_dir) / f"mean_pairwise_corr_{self.etf_name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        series_df.to_csv(path)
        print(f"[{self.etf_name}] Mean pairwise correlation series saved to {path}")

        return mean_corr_series