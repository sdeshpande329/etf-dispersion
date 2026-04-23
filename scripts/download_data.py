import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.data_pull import (
    WRDSConnection,
    SecidLookup,
    ETFOptionsPuller,
    ETFSpotPuller,
    RiskFreeRatePuller,
    ConstituentReturnsPuller,
    ConstituentImpliedVolPuller,
    TickerToPermnoCrosswalk,
)
from src.data.data_clean import DataCleaner
from src.data.holdings_loader import HoldingsLoader
from config.config import Config

def main():
    config = Config()
    conn = WRDSConnection()

    try:
        print("Step 1: Secid Validation")
        secid_lookup = SecidLookup(conn)
        secid_lookup.lookup(config.etf_tickers())

        print("Step 2: Pull ETF Options data")
        options_puller = ETFOptionsPuller(conn)
        etf_options = options_puller.pull()
        options_puller.save(etf_options)

        print("Step 3: Pull ETF Spot Prices")
        spot_puller = ETFSpotPuller(conn)
        etf_spot = spot_puller.pull()
        spot_puller.save(etf_spot)

        print("Step 4: Pull Risk-Free Rates")
        rate_puller = RiskFreeRatePuller(conn)
        rates = rate_puller.pull()
        rate_puller.save(rates)

        print("Step 5: Read ETF Holdings Data")
        holdings_loader = HoldingsLoader()
        holdings = holdings_loader.load_all()

        crosswalk = TickerToPermnoCrosswalk(conn)

        for etf_ticker in config.etf_tickers():
            constituent_tickers = holdings[etf_ticker]["ticker"].tolist()

            permno_map = crosswalk.get_permno_map(constituent_tickers)
            permno_list = permno_map["permno"].tolist()

            secid_map = crosswalk.get_secid_map(constituent_tickers)
            constituent_secids = secid_map["secid"].tolist()

            returns_puller = ConstituentReturnsPuller(conn, permno_list)
            constituent_returns = returns_puller.pull()
            if not constituent_returns.empty:
                returns_puller.save(constituent_returns, etf_ticker)

            iv_puller = ConstituentImpliedVolPuller(conn, constituent_secids)
            constituent_iv = iv_puller.pull()
            if not constituent_iv.empty:
                iv_puller.save(constituent_iv, etf_ticker)

        print("Step 6: Cleaning ETF Options Data")
        if etf_options.empty or etf_spot.empty:
            print("\nSkipped Step 6 because ETF options or spot data is empty. Fix secid mismatches above before running the cleaning step.")
        else:
            cleaner = DataCleaner()

            clean_full = cleaner.clean(etf_options, etf_spot, rates, atm_only=False)
            cleaner.save(clean_full, "clean_etf_options_full.csv")

            clean_atm = cleaner.clean(etf_options, etf_spot, rates, atm_only=True)
            cleaner.save(clean_atm, "clean_etf_options_atm.csv")
    finally:
        conn.close()
        print("WRDS connection closed.")


if __name__ == "__main__":
    main()