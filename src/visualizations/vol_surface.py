import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from pathlib import Path
from config.config import Config


FIGURES_DIR = Path("results/figures")
MATURITY_BUCKETS = [
    (20, 40, "30d"),
    (40, 70, "60d"),
]


class VolSurfacePlotter:
    """Builds and plots implied-volatility surfaces from ETF option data, overlays the synthetic basket vol, and highlights the spread."""

    def __init__(self):
        self.config = Config()

    @staticmethod
    def _bucket_maturity(dte: pd.Series) -> pd.Series:
        """Assign each option to a maturity bucket label."""
        labels = pd.Series(np.nan, index=dte.index, dtype=object)
        for lo, hi, label in MATURITY_BUCKETS:
            mask = (dte >= lo) & (dte < hi)
            labels[mask] = label
        return labels

    def _load_and_prepare(self, options_path: str) -> pd.DataFrame:
        """Load full ETF option chain and add maturity bucket."""
        df = pd.read_csv(options_path, parse_dates=["date"])
        df["maturity_bucket"] = self._bucket_maturity(df["days_to_expiry"])
        df = df.dropna(subset=["maturity_bucket", "impl_volatility"])
        return df
    
    def plot_vol_surface(self, options_path: str = "data/processed/clean_etf_options_full.csv", save: bool = True) -> None:
        """For each ETF, create a 3-D surface plot:
                x = log-moneyness, y = days-to-expiry, z = implied vol.

        Uses a representative snapshot (median date) so the surface is not an average across time.
        """
        df = self._load_and_prepare(options_path)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        for ticker in self.config.etf_tickers():
            sub = df[df["ticker"] == ticker]
            if sub.empty:
                continue

            date_counts = sub.groupby("date").size()
            snap_date = date_counts.idxmax()
            snap = sub[sub["date"] == snap_date].copy()

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                snap["log_moneyness"],
                snap["days_to_expiry"],
                snap["impl_volatility"],
                c=snap["impl_volatility"],
                cmap="viridis",
                s=8,
                alpha=0.7,
            )
            ax.set_xlabel("Log-moneyness ln(K/S)")
            ax.set_ylabel("Days to Expiry")
            ax.set_zlabel("Implied Volatility")
            ax.set_title(f"{ticker} IV Surface — {snap_date.strftime('%Y-%m-%d')}")

            if save:
                fig.savefig(
                    FIGURES_DIR / f"vol_surface_{ticker}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
            plt.close(fig)

        
    def plot_atm_spread(self, signals_path: str = "data/processed/signals.csv", save: bool = True) -> None:
        """Plot the ATM implied-correlation spread (sigma_ETF - sigma_basket) over time for all ETFs on a single chart."""
        signals = pd.read_csv(signals_path, parse_dates=["date"])
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        ax1 = axes[0]
        for ticker in self.config.etf_tickers():
            grp = signals[signals["ticker"] == ticker].sort_values("date")
            if grp.empty:
                continue
            ax1.plot(grp["date"], grp["spread"], label=ticker, linewidth=0.8)
        ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
        ax1.set_ylabel("Spread (σ_ETF − σ_basket)")
        ax1.set_title("ATM Implied-Correlation Spread by ETF")
        ax1.legend()

        ax2 = axes[1]
        for ticker in self.config.etf_tickers():
            grp = signals[signals["ticker"] == ticker].sort_values("date")
            if grp.empty:
                continue
            ax2.plot(grp["date"], grp["zscore"], label=ticker, linewidth=0.8)
        ax2.axhline(
            self.config.SIGNAL_ENTRY_ZSCORE,
            color="red",
            linestyle="--",
            linewidth=0.7,
            label=f"Entry z={self.config.SIGNAL_ENTRY_ZSCORE}",
        )
        ax2.axhline(
            self.config.SIGNAL_EXIT_ZSCORE,
            color="green",
            linestyle="--",
            linewidth=0.7,
            label=f"Exit z={self.config.SIGNAL_EXIT_ZSCORE}",
        )
        ax2.set_ylabel("Z-score")
        ax2.set_xlabel("Date")
        ax2.set_title("Spread Z-score with Entry / Exit Thresholds")
        ax2.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        if save:
            fig.savefig(
                FIGURES_DIR / "atm_spread_timeseries.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close(fig)

    def plot_cumulative_pnl(self, pnl_path: str = "data/processed/backtest_pnl.csv", save: bool = True) -> None:
        """Plot cumulative net PnL for each ETF side by side."""
        pnl = pd.read_csv(pnl_path, parse_dates=["date"])
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 6))

        for ticker in self.config.etf_tickers():
            grp = pnl[pnl["ticker"] == ticker].sort_values("date")
            if grp.empty:
                continue
            ax.plot(grp["date"], grp["cum_pnl_net"], label=f"{ticker} (net)", linewidth=1)
            ax.plot(
                grp["date"],
                grp["cum_pnl_gross"],
                label=f"{ticker} (gross)",
                linewidth=0.7,
                linestyle="--",
                alpha=0.6,
            )

        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative PnL ($)")
        ax.set_title("Dispersion Strategy — Cumulative PnL by ETF")
        ax.legend()
        fig.tight_layout()

        if save:
            fig.savefig(
                FIGURES_DIR / "cumulative_pnl.png", dpi=150, bbox_inches="tight"
            )
        plt.close(fig)

    def plot_all(self, options_path: str = "data/processed/clean_etf_options_full.csv", signals_path: str = "data/processed/signals.csv", pnl_path: str = "data/processed/backtest_pnl.csv") -> None:
        """Generate all plots in one call."""
        self.plot_vol_surface(options_path)
        self.plot_atm_spread(signals_path)
        self.plot_cumulative_pnl(pnl_path)