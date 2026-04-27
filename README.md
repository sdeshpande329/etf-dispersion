# Implied Correlation Arbitrage: A Dispersion-Based Options Strategy for Equity ETFs
## Problem Statement
Equity ETFs are priced by the market as a single asset, yet their value is mechanically determined by a weighted basket of constituent securities. This creates a testable no-arbitrage relationship between the ETF's implied volatility and the volatilities of its constituents: under correlated geometric Brownian motion, the variance of the ETF basket is fully determined by constituent variances and pairwise correlations. When the ETF's market-quoted implied volatility deviates persistently from the volatility implied by its constituents, a dispersion trading opportunity arises. The size and persistence of this deviation is expected to vary with market liquidity, as less liquid ETFs attract fewer arbitrageurs and may sustain wider mispricings for longer.

This project constructs and tests a dispersion trading strategy across a spectrum of equity ETFs with varying liquidity. Using historical options data from OptionMetrics, we calibrate the constituent correlation structure for each ETF, derive the theoretical basket implied volatility, and compare it to the ETF's observed implied volatility surface. We visualize this comparison directly as a vol surface, plotting the ETF implied vol surface against the synthetic basket vol surface to identify the strike and maturity regions where the gap is largest. The gap between the two, known as the implied correlation spread, serves as the primary trading signal: we sell ETF options and buy constituent options when the spread is sufficiently wide. We hypothesize that this spread is wider and more persistent in less liquid markets, and that net strategy returns increase with decreasing ETF liquidity up to the point where transaction costs dominate. We backtest the strategy across a high-volatility regime and a low-volatility regime, comparing signal behavior and net returns across ETF liquidity tiers to draw conclusions about where dispersion trading is most viable in practice.

## Repository Structure

```sh
etf-dispersion
├── config
│   └── config.py                                       # Store global variables for analysis (tickers, directories)
├── data
│   ├── holdings                                        # Snapshots of ETF constituents at different points of time
│   │   ├── spy_holdings_2022.csv
│   │   ├── spy_holdings_2023.csv
│   │   ├── spy_holdings_2024.csv
│   │   ├── xbi_holdings_2022.csv
│   │   ├── xbi_holdings_2023.csv
│   │   ├── xbi_holdings_2024.csv
│   │   ├── xlk_holdings_2022.csv
│   │   ├── xlk_holdings_2023.csv
│   │   └── xlk_holdings_2024.csv
│   ├── processed                                       # Processed data
│   │   ├── clean_etf_options_atm.csv
│   │   └── clean_etf_options_full.csv
│   └── raw                                             # Raw data of ETFs, constituents, and markets
│       ├── raw_constituent_mapping_SPY.csv
│       ├── raw_constituent_mapping_XBI.csv
│       ├── raw_constituent_mapping_XLK.csv
│       ├── raw_constituent_iv_SPY.csv
│       ├── raw_constituent_iv_XBI.csv
│       ├── raw_constituent_iv_XLK.csv
│       ├── raw_constituent_returns_SPY.csv
│       ├── raw_constituent_returns_XBI.csv
│       ├── raw_constituent_returns_XLK.csv
│       ├── raw_etf_options.csv
│       ├── raw_etf_spot.csv
│       └── raw_rates.csv
├── scripts
│   ├── download_data.py                                # Orchestrator to run code for downloading data from WRDS
│   └── scrape_nport.py                                 # Scrapes data from SEC using NPORT API to get constituent data
├── src
│   ├── data                
│   │   ├── __init__.py
│   │   ├── data_clean.py                               # Processes data and stores as CSV
│   │   ├── data_pull.py                                # Pulls data from WRDS
│   │   └── holdings_loader.py                          # Loads data from constituent csvs scraped from SEC
│   ├── model
│   │   ├── __init__.py
│   │   ├── basket_vol.py                               # Derives Implied Volatility of synthetic ETF                
│   │   └── correlation.py                              # Estimates pairwise correlation matrix across ETF constituents
│   ├── signal
│   │   ├── __init__.py
│   │   └── signal.py                                   # Translates correlation spreads and other data into signals
│   ├── strategy
│   │   ├── __init__.py
│   │   ├── backtest.py                                 # Computes strategy performance metrics
│   │   └── strategy.py                                 # Turns signals to positions
│   ├── visualizations
│   │   ├── __init__.py
│   │   └── vol_surface.py                              # Creates plots of volatility surfaces
│   └── __init__.py
└── requirements.txt                                    # Required packages for analysis
```
