"""
basket_vol.py

Responsible for deriving the theoretical implied volatility of each ETF basket
from its constituent implied volatilities and pairwise correlations.

Inputs:
    - Constituent ATM implied vols from OptionMetrics (loaded from data/processed/)
    - Constituent weights from holdings_loader.py
    - Rolling correlation matrices from correlation.py

Outputs:
    - Daily time series of sigma_basket per ETF

Implementation notes:
    - Apply the basket variance formula:
        sigma_basket^2 = sum_i sum_j w_i * w_j * sigma_i * sigma_j * rho_ij
    - Take the square root to obtain sigma_basket
    - Match constituent implied vols to ATM strikes using log-moneyness closest to zero
      for a consistent 30-60 day maturity window
    - Output should be a dataframe indexed by (etf, date) with columns for
      sigma_basket, number of constituents used, and any missing constituent flags
    - Log a warning when a constituent implied vol is missing and note how it was handled
      (dropped or forward-filled)
"""