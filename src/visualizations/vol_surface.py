"""
vol_surface.py

Responsible for constructing and plotting the implied volatility surface for each
ETF alongside its synthetic basket vol surface and the spread between the two.

Inputs:
    - ETF implied vols across all strikes and maturities from data/processed/
    - Synthetic basket vols from basket_vol.py
    - Log-moneyness values computed during data cleaning

Outputs:
    - Vol surface plots saved to results/figures/
    - Three surfaces per ETF: market quoted ETF surface, synthetic basket surface,
      and spread surface (ETF minus synthetic)

Implementation notes:
    - Express strikes as log-moneyness = ln(strike/spot) on the x-axis
    - Bucket maturities into 30, 60, 90, and 180 day groups on the y-axis
    - Plot all three surfaces side by side or as an overlay for each ETF
    - Highlight regions where the spread is largest as these correspond to the
      most attractive entry points for the dispersion trade
    - Make plots interactive where possible to allow inspection by strike and maturity
    - Produce a separate overlay plot comparing the ATM spread over time across
      all ETFs in the universe for the cross-ETF comparison in the writeup
"""