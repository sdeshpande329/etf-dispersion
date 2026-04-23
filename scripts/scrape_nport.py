import sys
import time
import pandas as pd
from pathlib import Path
from sec_api import FormNportApi
from src.data.data_pull import WRDSConnection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.config import Config


ETF_FILINGS = {
    "SPY": [
        ("2022", "0001752724-22-127611"),
        ("2023", "0001752724-23-196486"),
        ("2024", "0001752724-25-043826"),
    ],
    "XLK": [
        ("2022", "0001752724-22-048710"),
        ("2023", "0001752724-23-196456"),
        ("2024", "0001752724-25-044012"),
    ],
    "XBI": [
        ("2022", "0001752724-22-048691"),
        ("2023", "0001752724-23-196444"),
        ("2024", "0001752724-25-044105"),
    ],
}

OUTPUT_DIR = Path("data/holdings")

# Sleep between API calls to be a good citizen
SLEEP_BETWEEN_CALLS = 0.5



class NPortFilingFetcher:
    """Fetches a single N-PORT filing by exact accession number using the sec-api FormNportApi."""

    def __init__(self, api: FormNportApi):
        self.api = api

    def fetch(self, accession_number: str) -> dict:
        query = {
            "query": f'accessionNo:"{accession_number}"',
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}],
        }

        time.sleep(SLEEP_BETWEEN_CALLS)
        response = self.api.get_data(query)
        filings  = response.get("filings", [])

        if not filings:
            raise ValueError(
                f"No filing found for accession number {accession_number}"
            )

        filing     = filings[0]
        series     = filing.get("genInfo", {}).get("seriesName", "N/A")
        rep_pd_end = filing.get("genInfo", {}).get("repPdEnd", "N/A")
        return filing


class NPortHoldingsExtractor:
    """Extracts equity constituent holdings from a single N-PORT filing dict. From the invstOrSecs list we keep only:
        - USD-denominated holdings  (curCd == 'USD')
        - Equity asset category     (assetCat == 'EC')
        - Holdings with a valid ticker in identifiers.ticker.value
        - Holdings with pctVal > 0
    """

    def extract(self, filing: dict) -> pd.DataFrame:
        """Extracts USD equity holdings from a filing dict."""
        holdings    = filing.get("invstOrSecs", [])
        series_name = filing.get("genInfo", {}).get("seriesName", "UNKNOWN")
        rep_pd_end  = filing.get("genInfo", {}).get("repPdEnd", "UNKNOWN")

        if not holdings:
            print(f"[NPortHoldingsExtractor] WARNING: no invstOrSecs in "
                f"filing for '{series_name}'")
            return pd.DataFrame(columns=["cusip", "name", "weight"])

        rows = []
        for h in holdings:
            if h.get("curCd") != "USD":
                continue
            if h.get("assetCat") != "EC":
                continue

            pct_val = h.get("pctVal")
            if pct_val is None or float(pct_val) <= 0:
                continue

            cusip = h.get("cusip", "").strip()
            name  = h.get("name", "").strip()

            if not cusip or cusip == "N/A" or len(cusip) < 8:
                continue

            rows.append({
                "cusip":  cusip,
                "name":   name,
                "weight": float(pct_val) / 100.0,
            })

        df = pd.DataFrame(rows)
        return df

class NPortScraper:
    """Orchestrates fetching and extracting N-PORT holdings for each ETF and target period using hardcoded accession numbers, then saves results as CSVs in the format expected by HoldingsLoader."""
    def __init__(self):
        config   = Config()
        api_key  = config.load_credentials()["sec"]["key"]
        self.api       = FormNportApi(api_key)
        self.fetcher   = NPortFilingFetcher(self.api)
        self.extractor = NPortHoldingsExtractor()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def scrape_one(self, etf_ticker: str, label: str, accession_number: str) -> Path:
        """Fetches and saves holdings for one ETF / period combination."""
        filing   = self.fetcher.fetch(accession_number)
        holdings = self.extractor.extract(filing)

        if holdings.empty:
            raise ValueError(
                f"No holdings extracted for {etf_ticker} {label} "
                f"(accession={accession_number})"
            )

        output_path = OUTPUT_DIR / f"{etf_ticker.lower()}_holdings_{label}.csv"
        holdings.to_csv(output_path, index=False)
        return output_path

    def scrape_all(self) -> None:
        """Scrapes all ETFs across all target periods."""
        results = []
        for etf_ticker, filings in ETF_FILINGS.items():
            for label, accession_number in filings:
                try:
                    path = self.scrape_one(etf_ticker, label, accession_number)
                    results.append((etf_ticker, label, "OK", str(path)))
                except Exception as e:
                    print(
                        f"[NPortScraper] ERROR: {etf_ticker} {label}: {e}"
                    )
                    results.append((etf_ticker, label, "ERROR", str(e)))

class CusipToTickerResolver:
    """Resolves CUSIPs to tickers using the CRSP stocknames table via WRDS."""

    def __init__(self, connection):
        self.conn = connection

    def resolve(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ncusip"] = df["cusip"].str[:8]

        cusip_clause = ", ".join(f"'{c}'" for c in df["ncusip"].unique())
        query = f"""
            SELECT DISTINCT ncusip, ticker
            FROM crsp.stocknames
            WHERE ncusip IN ({cusip_clause})
            ORDER BY ncusip
        """
        crosswalk = self.conn.raw_sql(query)

        crosswalk = (
            crosswalk
            .drop_duplicates(subset="ncusip", keep="last")
            .set_index("ncusip")["ticker"]
            .to_dict()
        )

        df["ticker"] = df["ncusip"].map(crosswalk)

        n_before = len(df)
        df = df.dropna(subset=["ticker"])
        df = df[["ticker", "weight"]].reset_index(drop=True)
        return df


def main():
    scraper  = NPortScraper()
    conn     = WRDSConnection()
    resolver = CusipToTickerResolver(conn)

    try:
        for etf_ticker, filings in ETF_FILINGS.items():
            for label, accession_number in filings:
                try:
                    filing   = scraper.fetcher.fetch(accession_number)
                    holdings = scraper.extractor.extract(filing)

                    if holdings.empty:
                        raise ValueError(
                            f"No holdings extracted for {etf_ticker} {label}"
                        )

                    holdings = resolver.resolve(holdings)

                    if holdings.empty:
                        raise ValueError(
                            f"No tickers resolved for {etf_ticker} {label}"
                        )

                    output_path = (
                        OUTPUT_DIR /
                        f"{etf_ticker.lower()}_holdings_{label}.csv"
                    )
                    holdings.to_csv(output_path, index=False)
                except Exception as e:
                    print(f"[NPortScraper] ERROR: {etf_ticker} {label}: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()