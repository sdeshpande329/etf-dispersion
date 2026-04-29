"""
run_strategy.py

End-to-end orchestrator for the trading pipeline:
    1. Generate entry/exit signals from spread z-scores
    2. Simulate the dispersion strategy (position log)
    3. Run backtest analytics (PnL, Sharpe, drawdown)
    4. Generate all visualisation plots

Assumes the data pipeline (download_data.py) and basket vol pipeline
(run_basket_vol.py) have already been run.
"""

import sys
from pathlib import Path

# ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.signal.signal import SignalGenerator
from src.strategy.strategy import DispersionStrategy
from src.strategy.backtest import Backtester
from src.visualizations.vol_surface import VolSurfacePlotter


def main():
    print("\n[1/4] Generating signals …")
    sig = SignalGenerator()
    sig.generate(
        atm_path="data/processed/clean_etf_options_atm.csv",
        basket_dir="data/processed",
    )
    sig.save("data/processed/signals.csv")
    signals = sig.get_signals()
    n_entries = int(signals["entry_signal"].sum())
    print(f"       {len(signals)} signal rows, {n_entries} entry signals")

    print("\n[2/4] Running dispersion strategy …")
    strat = DispersionStrategy()
    strat.run(signals)
    strat.save("data/processed/position_log.csv")
    log = strat.get_position_log()
    print(f"       {len(log)} position-log rows")

    print("\n[3/4] Running backtest …")
    bt = Backtester()
    bt.run(log)
    bt.save(
        pnl_path="data/processed/backtest_pnl.csv",
        summary_path="data/processed/backtest_summary.csv",
    )
    bt.print_summary()

    print("\n[4/4] Generating plots …")
    plotter = VolSurfacePlotter()
    plotter.plot_all()

    print("\nDone. All outputs in data/processed/ and results/figures/\n")


if __name__ == "__main__":
    main()
