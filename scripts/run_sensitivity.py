import sys
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.signal.signal import SignalGenerator
from src.strategy.strategy import DispersionStrategy
from src.strategy.backtest import Backtester

ENTRY_ZSCORES = [0.5, 0.75, 1.0, 1.25, 1.5]
EXIT_ZSCORES = [-0.5, -0.25, 0.0, 0.25]


def main():
    rows = []
    total = len(ENTRY_ZSCORES) * len(EXIT_ZSCORES)
    i = 0

    for entry_z, exit_z in product(ENTRY_ZSCORES, EXIT_ZSCORES):
        # entry must be above exit to make sense
        if entry_z <= exit_z:
            continue

        i += 1
        print(f"  [{i}/{total}] entry_z={entry_z:.2f}  exit_z={exit_z:.2f}")

        sig = SignalGenerator(
            entry_zscore=entry_z,
            exit_zscore=exit_z,
        )
        sig.generate()
        signals = sig.get_signals()

        strat = DispersionStrategy()
        strat.run(signals)
        log = strat.get_position_log()

        if log.empty:
            continue

        bt = Backtester()
        bt.run(log)
        summary = bt.get_summary()

        for _, r in summary.iterrows():
            rows.append(
                {
                    "entry_z": entry_z,
                    "exit_z": exit_z,
                    "ticker": r["ticker"],
                    "n_trades": int(r["n_trades"]),
                    "total_pnl_net": r["total_pnl_net"],
                    "sharpe_net": r["sharpe_net"],
                    "max_drawdown": r["max_drawdown"],
                    "total_txn_cost": r["total_txn_cost"],
                }
            )

    df = pd.DataFrame(rows)
    out = Path("data/processed/sensitivity_table.csv")
    df.to_csv(out, index=False)
    
    # Pretty-print pivot: Sharpe by (entry_z, exit_z) for each ETF
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker]
        pivot = sub.pivot_table(
            index="entry_z", columns="exit_z", values="sharpe_net"
        )
        print(f"  {ticker} — Annualised Sharpe (net)")
        print(f"  rows = entry_z, cols = exit_z\n")
        print(pivot.to_string(float_format="{:+.2f}".format))

    # Also print trade counts
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker]
        pivot = sub.pivot_table(
            index="entry_z", columns="exit_z", values="n_trades"
        )
        print(f"  {ticker} — Number of Trades")
        print(pivot.to_string(float_format="{:.0f}".format))

    print()


if __name__ == "__main__":
    main()
