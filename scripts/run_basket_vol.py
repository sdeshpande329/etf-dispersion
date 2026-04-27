import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model.basket_vol import compute_all_etfs
from config.config import Config


def main() -> None:
    config = Config()
    print("Running basket volatility computation for all ETFs...")
    results = compute_all_etfs(
        iv_data_dir=config.RAW_DIR,
        weights_dir=config.HOLDINGS_DIR,
        output_dir=config.PROCESSED_DIR,
    )

    for etf_ticker, sigma_basket_df in results.items():
        print(
            f"[{etf_ticker}] computed {len(sigma_basket_df)} dates "
            f"and saved to {Path(config.PROCESSED_DIR) / f'sigma_basket_{etf_ticker}.csv'}"
        )


if __name__ == "__main__":
    main()
