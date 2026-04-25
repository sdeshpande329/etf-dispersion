"""
signal.py

Responsible for computing the implied correlation spread and generating
entry and exit signals for the dispersion trade.

Why this file is in src/signal/ and not src/model/ or src/strategy/:
    - It does not belong in src/model/ because it is not a mathematical model of
      prices or volatility. The model layer (correlation.py, basket_vol.py) is
      purely quantitative and has no knowledge of trading logic. signal.py consumes
      model outputs and applies a decision rule, which is a separate concern.
    - It does not belong in src/strategy/ because it does not construct positions,
      size trades, or interact with prices directly. strategy.py consumes the signal
      produced here and decides what to do with it. Keeping the signal generation
      separate means the signal can be evaluated, plotted, and tested independently
      of any trading logic.

Inputs:
    - Daily sigma_ETF (market quoted ATM implied vol) from data/processed/
    - Daily sigma_basket (derived) from basket_vol.py

Outputs:
    - Daily implied correlation spread per ETF
    - Entry and exit signal flags per ETF indexed by date

Implementation notes:
    - Compute implied correlation spread = sigma_ETF minus sigma_basket
    - Compute 60-day rolling mean and standard deviation of the spread
    - Generate entry signal when spread exceeds rolling mean by 1 standard deviation
    - Generate exit signal when spread reverts to rolling mean
    - Output should be a dataframe indexed by (etf, date) with columns for spread,
      rolling mean, rolling std, entry flag, and exit flag
    - Compare signal width, persistence, and decay speed across ETF liquidity tiers
"""