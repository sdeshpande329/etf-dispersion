import yaml
from pathlib import Path


class Config:
    ETF_UNIVERSE = [
        (136639, "SPY", 1),
        (101740, "XLK", 2),
        (149751, "XBI", 3),
    ]

    CONSTITUENT_CAPS = {
        "SPY": 50,
        "XLK": 25,
        "XBI": 20,
    }

    HOLDINGS_DIR = "data/holdings"

    HOLDINGS_FILES = {
        "SPY": "spy_holdings.csv",
        "XLK": "xlk_holdings.csv",
        "XBI": "xbi_holdings.csv",
    }

    START_DATE = "2022-01-01"
    END_DATE = "2024-12-31"

    DAYS_TO_EXPIRY_MIN = 30
    DAYS_TO_EXPIRY_MAX = 60

    MAX_SPREAD_PCT = 0.10          # Max bid-ask spread as fraction of mid price
    MIN_VOLUME = 10
    MIN_IMPLIED_VOLATILITY = 0.01  
    MAX_IMPLIED_VOLATILITY = 0.50  
    MIN_OPTION_PRICE = 0.01

    # Log-moneyness filter: ln(strike/spot) in [-LOG_MONEYNESS_BAND, +LOG_MONEYNESS_BAND]
    LOG_MONEYNESS_BAND = 0.15

    ROLLING_CORR_WINDOW = 60       # Days for rolling historical correlation

    SIGNAL_ROLLING_MEAN_WINDOW = 60   # Rolling mean window for spread z-score
    SIGNAL_ENTRY_ZSCORE = 1.0         # Enter when spread > mean + 1 std
    SIGNAL_EXIT_ZSCORE = 0.0          # Exit when spread reverts to mean

    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    SAVE_CSV = True

    @staticmethod
    def load_credentials(credentials_path: str = "config/credentials.yaml") -> dict:
        with open(credentials_path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def get_wrds_username(cls, credentials_path: str = "config/credentials.yaml") -> str:
        credentials = cls.load_credentials(credentials_path)
        return credentials["wrds"]["username"]

    @classmethod
    def etf_secids(cls) -> list:
        return [entry[0] for entry in cls.ETF_UNIVERSE]

    @classmethod
    def etf_tickers(cls) -> list:
        return [entry[1] for entry in cls.ETF_UNIVERSE]

    @classmethod
    def secid_to_ticker(cls) -> dict:
        return {entry[0]: entry[1] for entry in cls.ETF_UNIVERSE}

    @classmethod
    def ticker_to_secid(cls) -> dict:
        return {entry[1]: entry[0] for entry in cls.ETF_UNIVERSE}

    @classmethod
    def ticker_to_tier(cls) -> dict:
        return {entry[1]: entry[2] for entry in cls.ETF_UNIVERSE}