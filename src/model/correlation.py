"""
correlation.py

Responsible for estimating the pairwise correlation matrix across ETF constituents
from historical price data.

Inputs:
    - Constituent daily log returns from CRSP (loaded from data/processed/)

Outputs:
    - Rolling 60-day correlation matrix per ETF, indexed by date

Implementation notes:
    - Compute daily log returns from CRSP closing prices before estimating correlations
    - Use a 60-day rolling window to capture time-varying correlation structure
    - Apply Ledoit-Wolf shrinkage estimator (sklearn.covariance.LedoitWolf) on each
      rolling window to reduce estimation error given the number of constituents
      relative to the number of observations
    - Cap constituents at top N by weight per ETF before estimation to keep the
      matrix tractable: SPY top 50, XLK top 25, others top 20-30
    - Output should be a dictionary or panel indexed by (etf, date) mapping to a
      square numpy array of shape (n_constituents, n_constituents)
"""