import pandas as pd
from pathlib import Path
from config.config import Config


class HoldingsLoader:
    """Loads ETF constituent holdings from three CSV snapshots per ETF: beginning, midpoint, and end of the analysis period. This approximates time-averaged portfolio exposure without requiring
    a full historical holdings database, while avoiding the undefined basket vol problem that arises when constituents enter or exit mid-backtest."""

    # Warn if the stable constituent set covers less than this fraction of the union weight (indicates high turnover and large approximation error)
    MIN_WEIGHT_COVERAGE_WARNING = 0.70

    def __init__(self):
        self.config = Config()
        self.holdings_dir = Path(self.config.HOLDINGS_DIR)

    def _load_single_snapshot(self, filepath: Path) -> pd.DataFrame:
        """Loads and normalizes a single holdings CSV snapshot."""
        if not filepath.exists():
            raise FileNotFoundError(
                f"Holdings snapshot not found: {filepath}\n"
                f"Please download it from the ETF provider and place it at {filepath}"
            )

        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower() for c in df.columns]

        if "ticker" not in df.columns or "weight" not in df.columns:
            raise ValueError(
                f"Holdings CSV at {filepath} must have 'Ticker' and 'Weight' columns. "
                f"Found: {list(df.columns)}"
            )

        df = df[["ticker", "weight"]].copy()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df = df.dropna(subset=["ticker", "weight"])

        df = df[df["ticker"].str.match(r'^[A-Z]{1,5}$')]

        if df["weight"].max() > 1.5:
            df["weight"] = df["weight"] / 100.0

        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df = df.dropna(subset=["weight"])
        df = df[df["weight"] > 0]

        return df.reset_index(drop=True)

    def _check_weight_coverage(self, etf_ticker: str, stable_df: pd.DataFrame, all_snapshots: list) -> None:
        """Computes and prints the fraction of total weight covered by the stable constituent set versus the full union of all snapshots."""
        union_df = pd.concat(all_snapshots, ignore_index=True)
        union_avg = (
            union_df.groupby("ticker")["weight"]
            .mean()
            .reset_index()
        )
        union_avg["weight"] = union_avg["weight"] / union_avg["weight"].sum()

        stable_tickers = set(stable_df["ticker"])
        covered_weight = union_avg[
            union_avg["ticker"].isin(stable_tickers)
        ]["weight"].sum()

        print(f"{etf_ticker}: stable set covers {covered_weight:.1%} of union weight across snapshots")

        if covered_weight < self.MIN_WEIGHT_COVERAGE_WARNING:
            print(f"[HoldingsLoader] WARNING: {etf_ticker} coverage "
                  f"{covered_weight:.1%} is below {self.MIN_WEIGHT_COVERAGE_WARNING:.0%} threshold, indicating high constituent turnover")

    def load(self, etf_ticker: str) -> pd.DataFrame:
        """Loads and processes holdings for a single ETF across three snapshots."""
        filenames = self.config.HOLDINGS_FILES.get(etf_ticker)
        if filenames is None:
            raise ValueError(f"No holdings files configured for {etf_ticker}")
        if len(filenames) != 3:
            raise ValueError(
                f"Expected exactly 3 snapshot files for {etf_ticker}, "
                f"got {len(filenames)}: {filenames}"
            )

        snapshots = []
        for filename in filenames:
            filepath = self.holdings_dir / filename
            df = self._load_single_snapshot(filepath)
            snapshots.append(df)
            print(f"[HoldingsLoader] {etf_ticker}: loaded {len(df)} constituents "
                  f"from {filename}")

        # Find stable constituent set: tickers present in all three snapshots
        ticker_sets = [set(df["ticker"]) for df in snapshots]
        stable_tickers = ticker_sets[0] & ticker_sets[1] & ticker_sets[2]

        n_union = len(ticker_sets[0] | ticker_sets[1] | ticker_sets[2])
        print(f"[HoldingsLoader] {etf_ticker}: {len(stable_tickers)} stable "
              f"constituents (present in all 3 snapshots) out of "
              f"{n_union} unique tickers across snapshots")

        # Filter each snapshot to stable tickers only
        stable_snapshots = [
            df[df["ticker"].isin(stable_tickers)].copy()
            for df in snapshots
        ]

        # Average weights across the three snapshots
        combined = pd.concat(stable_snapshots, ignore_index=True)
        averaged = (
            combined.groupby("ticker")["weight"]
            .mean()
            .reset_index()
            .rename(columns={"weight": "weight"})
        )

        # Check weight coverage before capping
        self._check_weight_coverage(etf_ticker, averaged, snapshots)

        # Sort by averaged weight, cap at top N
        cap = self.config.CONSTITUENT_CAPS.get(etf_ticker, 30)
        averaged = averaged.sort_values("weight", ascending=False).head(cap).copy()

        # Renormalize to sum to 1.0 after capping
        averaged["weight"] = averaged["weight"] / averaged["weight"].sum()
        averaged = averaged.reset_index(drop=True)

        print(f"{etf_ticker}: final {len(averaged)} constituents after cap of {cap}, weights sum to {averaged['weight'].sum():.4f}")

        return averaged

    def load_all(self) -> dict:
        """Loads holdings for all ETFs in the universe."""
        result = {}
        for ticker in self.config.etf_tickers():
            print(f"\n--- Loading holdings for {ticker} ---")
            result[ticker] = self.load(ticker)
        return result