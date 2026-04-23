import pandas as pd
from pathlib import Path
from config.config import Config


class HoldingsLoader:
    """Loads ETF constituent holdings from CSV snapshots downloaded from ETF provider websites."""
    def __init__(self):
        self.config = Config()
        self.holdings_dir = Path(self.config.HOLDINGS_DIR)

    def load(self, etf_ticker: str) -> pd.DataFrame:
        """Loads and processes holdings for a single ETF."""
        filename = self.config.HOLDINGS_FILES.get(etf_ticker)
        if filename is None:
            raise ValueError(f"No holdings file configured for {etf_ticker}")

        filepath = self.holdings_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Holdings file not found: {filepath}\n"
                f"Please download it from the ETF provider website and place it at {filepath}"
            )

        df = pd.read_csv(filepath)

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns or "weight" not in df.columns:
            raise ValueError(
                f"Holdings CSV for {etf_ticker} must have 'Ticker' and 'Weight' columns. "
                f"Found: {list(df.columns)}"
            )

        df = df.rename(columns={"ticker": "ticker", "weight": "weight"})
        df = df[["ticker", "weight"]].dropna()

        # Convert weight from percent to decimal if needed
        if df["weight"].max() > 1.5:
            df["weight"] = df["weight"] / 100.0

        # Sort by weight descending, cap at top N
        cap = self.config.CONSTITUENT_CAPS.get(etf_ticker, 30)
        df = df.sort_values("weight", ascending=False).head(cap).copy()

        # Re-normalize weights to sum to 1.0 after capping
        df["weight"] = df["weight"] / df["weight"].sum()
        df = df.reset_index(drop=True)
        return df

    def load_all(self) -> dict:
        """Loads holdings for all ETFs in the universe."""
        result = {}
        for ticker in self.config.etf_tickers():
            result[ticker] = self.load(ticker)
        return result