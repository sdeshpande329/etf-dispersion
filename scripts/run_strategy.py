import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.signal.signal import SignalGenerator
from src.strategy.strategy import DispersionStrategy
from src.strategy.backtest import Backtester
from src.visualizations.vol_surface import VolSurfacePlotter


def main():
    print("\nStep 1: Generating signals")
    sig = SignalGenerator()
    sig.generate(
        atm_path="data/processed/clean_etf_options_atm.csv",
        basket_dir="data/processed",
    )
    sig.save("data/processed/signals.csv")
    signals = sig.get_signals()
    n_entries = int(signals["entry_signal"].sum())
    print(f"       {len(signals)} signal rows, {n_entries} entry signals")

    print("\nStep 2: Running dispersion strategy")
    strat = DispersionStrategy()
    strat.run(signals)
    strat.save("data/processed/position_log.csv")
    log = strat.get_position_log()
    print(f"       {len(log)} position-log rows")

    print("\nStep 3: Running backtest")
    bt = Backtester()
    bt.run(log)
    bt.save(
        pnl_path="data/processed/backtest_pnl.csv",
        summary_path="data/processed/backtest_summary.csv",
    )
    bt.print_summary()

    print("\nStep 4: Generating plots")
    plotter = VolSurfacePlotter()
    plotter.plot_all()

if __name__ == "__main__":
    main()
