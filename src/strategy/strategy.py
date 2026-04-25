"""
strategy.py

Responsible for translating entry and exit signals into concrete option positions
and managing the daily delta hedge.

Inputs:
    - Entry and exit signal flags from signal.py
    - ETF and constituent option data including greeks from data/processed/
    - ETF and constituent spot prices from data/processed/
    - Constituent weights from holdings_loader.py

Outputs:
    - Daily position log with option legs, delta hedge quantities, and trade costs

Implementation notes:
    - On entry signal: record a short ETF straddle position and long constituent
      straddle positions weighted by w_i
    - On exit signal: close all open legs of the position
    - Compute daily delta of each position using greeks (delta field) from OptionMetrics
    - Rebalance delta hedge daily using current spot prices
    - Deduct transaction costs on entry and exit using the bid-ask spread from
      OptionMetrics best_bid and best_offer fields
    - Position sizing should be consistent across ETFs to allow fair performance
      comparison in backtest.py
"""