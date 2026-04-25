"""
backtest.py

Responsible for computing strategy performance metrics from the position log
produced by strategy.py.

Inputs:
    - Daily position log from strategy.py
    - ETF and constituent option prices from data/processed/

Outputs:
    - Cumulative PnL time series per ETF
    - Performance metrics: Sharpe ratio, max drawdown, net PnL after costs
    - Cross-ETF summary table for comparison across liquidity tiers

Implementation notes:
    - Compute daily PnL by marking each open option leg to market using mid prices
    - Subtract transaction costs recorded at entry and exit from strategy.py
    - Aggregate into a cumulative PnL series per ETF
    - Compute annualized Sharpe ratio using daily PnL and the risk-free rate from
      optionm.zerocd
    - Compute max drawdown as the largest peak-to-trough decline in cumulative PnL
    - Split results by regime: 2022 (high vol) and 2023-2024 (low vol) to assess
      whether strategy performance is regime-dependent
    - Produce a summary dataframe comparing average spread, spread persistence,
      Sharpe, max drawdown, and net PnL across all ETFs in the universe
"""