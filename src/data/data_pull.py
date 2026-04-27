import wrds
import pandas as pd
from pathlib import Path
from config.config import Config


class WRDSConnection:
    """Sets connection to WRDS"""
    def __init__(self):
        self.config = Config()
        self.conn = wrds.Connection(wrds_username=self.config.get_wrds_username())

    def raw_sql(self, query: str) -> pd.DataFrame:
        return self.conn.raw_sql(query)

    def close(self):
        self.conn.close()


class SecidLookup:
    """Looks up and validates ETF secids against OptionMetrics tables."""

    def __init__(self, connection: WRDSConnection):
        self.conn = connection

    def lookup(self, tickers: list) -> pd.DataFrame:
        """Pulls tradeable ETF series from optionm.securd with issue_type = 'A' and class = 'I' (Intraday/tradeable ETF), which excludes NAV series and index composites."""
        
        ticker_clause = ", ".join(f"'{t}'" for t in tickers)
        query = f"""
            SELECT secid, ticker, issue_type, class, exchange_d
            FROM optionm.securd
            WHERE ticker IN ({ticker_clause})
                AND issue_type = 'A'
                AND class = 'I'
            ORDER BY ticker
        """
        result = self.conn.raw_sql(query)
        return result


class ETFOptionsPuller:
    """Pulls options data for ETFs from optionm.opprcdYYYY annual tables (merged for all years)"""
    def __init__(self, connection: WRDSConnection):
        self.conn = connection
        self.config = Config()
        self._secid_clause = ", ".join(str(sid) for sid in self.config.etf_secids())
        self._years = self._get_years()

    def _get_years(self) -> list:
        start_year = int(self.config.START_DATE[:4])
        end_year = int(self.config.END_DATE[:4])
        return list(range(start_year, end_year + 1))

    def _build_single_year_query(self, year: int) -> str:
        return f"""
            SELECT
                secid,
                date,
                exdate,
                strike_price / 1000.0 AS strike_price,
                cp_flag,
                best_bid,
                best_offer,
                volume,
                open_interest,
                impl_volatility,
                delta,
                gamma,
                theta,
                vega
            FROM optionm.opprcd{year}
            WHERE secid IN ({self._secid_clause})
                AND date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
                AND (exdate - date) BETWEEN {self.config.DAYS_TO_EXPIRY_MIN}
                    AND {self.config.DAYS_TO_EXPIRY_MAX}
                AND volume >= {self.config.MIN_VOLUME}
                AND impl_volatility BETWEEN {self.config.MIN_IMPLIED_VOLATILITY}
                    AND {self.config.MAX_IMPLIED_VOLATILITY}
        """

    def pull(self) -> pd.DataFrame:
        union_query = "\nUNION ALL\n".join(
            self._build_single_year_query(year) for year in self._years
        )
        full_query = (
            f"SELECT * FROM ({union_query}) AS combined ORDER BY secid, date"
        )
        df = self.conn.raw_sql(full_query)
        df["ticker"] = df["secid"].map(self.config.secid_to_ticker())
        return df

    def save(self, df: pd.DataFrame) -> None:
        path = Path(self.config.RAW_DIR) / "raw_etf_options.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


class ETFSpotPuller:
    """Pulls ETF spot prices from optionm.secprdYYYY annual tables."""
    def __init__(self, connection: WRDSConnection):
        self.conn = connection
        self.config = Config()
        self._secid_clause = ", ".join(str(sid) for sid in self.config.etf_secids())
        self._years = self._get_years()

    def _get_years(self) -> list:
        start_year = int(self.config.START_DATE[:4])
        end_year = int(self.config.END_DATE[:4])
        return list(range(start_year, end_year + 1))

    def _build_single_year_query(self, year: int) -> str:
        return f"""
            SELECT
                secid,
                date,
                close AS spot_price
            FROM optionm.secprd{year}
            WHERE secid IN ({self._secid_clause})
                AND date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
        """

    def pull(self) -> pd.DataFrame:
        union_query = "\nUNION ALL\n".join(
            self._build_single_year_query(year) for year in self._years
        )
        full_query = (
            f"SELECT * FROM ({union_query}) AS combined ORDER BY secid, date"
        )
        df = self.conn.raw_sql(full_query)
        df["ticker"] = df["secid"].map(self.config.secid_to_ticker())
        return df

    def save(self, df: pd.DataFrame) -> None:
        path = Path(self.config.RAW_DIR) / "raw_etf_spot.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        

class RiskFreeRatePuller:
    """Pulls zero-coupon risk-free rates from optionm.zerocd."""
    def __init__(self, connection: WRDSConnection):
        self.conn = connection
        self.config = Config()

    def pull(self) -> pd.DataFrame:
        query = f"""
            SELECT date, days, rate
            FROM optionm.zerocd
            WHERE date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
                AND days BETWEEN {self.config.DAYS_TO_EXPIRY_MIN}
                    AND {self.config.DAYS_TO_EXPIRY_MAX}
        """
        df = self.conn.raw_sql(query)
        return df

    def save(self, df: pd.DataFrame) -> None:
        path = Path(self.config.RAW_DIR) / "raw_rates.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


class ConstituentReturnsPuller:
    """Pulls daily constituent returns from crsp.dsf (CRSP Daily Security File). These returns are used to estimate the rolling historical correlation
    matrix for the basket variance calculation:
        sigma_basket^2 = sum_i sum_j w_i * w_j * sigma_i * sigma_j * rho_ij
    """

    def __init__(self, connection: WRDSConnection, permno_list: list):
        self.conn = connection
        self.config = Config()
        self.permno_list = permno_list

    def pull(self) -> pd.DataFrame:
        if not self.permno_list:
            print("Warning: permno_list is empty, skipping.")
            return pd.DataFrame()

        permno_clause = ", ".join(str(p) for p in self.permno_list)
        query = f"""
            SELECT permno, date, ret, prc, shrout
            FROM crsp.dsf
            WHERE permno IN ({permno_clause})
                AND date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
        """
        df = self.conn.raw_sql(query)
        return df

    def save(self, df: pd.DataFrame, ticker: str) -> None:
        path = Path(self.config.RAW_DIR) / f"raw_constituent_returns_{ticker}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        

class ConstituentImpliedVolPuller:
    """Pulls implied volatility for ETF constituent stocks from optionm.opprcdYYYY annual tables, unioning across the date range.
    Used to compute the synthetic basket vol surface:
        sigma_basket^2 = sum_i sum_j w_i * w_j * sigma_i * sigma_j * rho_ij
    """

    def __init__(self, connection: WRDSConnection, secid_list: list):
        self.conn = connection
        self.config = Config()
        self.secid_list = secid_list
        self._years = self._get_years()

    def _get_years(self) -> list:
        start_year = int(self.config.START_DATE[:4])
        end_year = int(self.config.END_DATE[:4])
        return list(range(start_year, end_year + 1))

    def _build_single_year_query(self, year: int) -> str:
        secid_clause = ", ".join(str(sid) for sid in self.secid_list)
        return f"""
            SELECT
                secid,
                date,
                exdate,
                strike_price / 1000.0 AS strike_price,
                cp_flag,
                impl_volatility,
                best_bid,
                best_offer,
                volume,
                delta
            FROM optionm.opprcd{year}
            WHERE secid IN ({secid_clause})
                AND date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
                AND (exdate - date) BETWEEN {self.config.DAYS_TO_EXPIRY_MIN}
                    AND {self.config.DAYS_TO_EXPIRY_MAX}
                AND volume >= {self.config.MIN_VOLUME}
                AND impl_volatility BETWEEN {self.config.MIN_IMPLIED_VOLATILITY}
                    AND {self.config.MAX_IMPLIED_VOLATILITY}
        """

    def pull(self) -> pd.DataFrame:
        if not self.secid_list:
            print("Warning: secid_list is empty, skipping.")
            return pd.DataFrame()

        union_query = "\nUNION ALL\n".join(
            self._build_single_year_query(year) for year in self._years
        )
        full_query = (
            f"SELECT * FROM ({union_query}) AS combined ORDER BY secid, date"
        )
        df = self.conn.raw_sql(full_query)
        return df

    def save(self, df: pd.DataFrame, etf_ticker: str) -> None:
        path = Path(self.config.RAW_DIR) / f"raw_constituent_iv_{etf_ticker}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        

class TickerToPermnoCrosswalk:
    """Maps constituent stock tickers to CRSP PERMNOs and OptionMetrics secids."""

    def __init__(self, connection: WRDSConnection):
        self.conn = connection

    def get_permno_map(self, tickers: list) -> pd.DataFrame:
        """Returns one (ticker, permno) row per ticker using the most recent name record from crsp.stocknames to resolve tickers that have had multiple 
        PERMNOs over time.
        """
        if not tickers:
            print("[TickerToPermnoCrosswalk] Warning: empty ticker list passed "
                  "to get_permno_map")
            return pd.DataFrame(columns=["ticker", "permno"])

        ticker_clause = ", ".join(f"'{t}'" for t in tickers)
        query = f"""
            SELECT ticker, permno, namedt, nameenddt
            FROM crsp.stocknames
            WHERE ticker IN ({ticker_clause})
            ORDER BY ticker, nameenddt DESC
        """
        df = self.conn.raw_sql(query)

        # Keep only the most recent record per ticker
        df = (
            df.sort_values("nameenddt", ascending=False)
              .drop_duplicates(subset="ticker", keep="first")
              [["ticker", "permno"]]
              .reset_index(drop=True)
        )
        return df

    def get_secid_map(self, tickers: list) -> pd.DataFrame:
        """Returns one (ticker, secid) row per ticker from optionm.secnmd."""
        if not tickers:
            print("Warning: empty ticker list passed to get_secid_map")
            return pd.DataFrame(columns=["ticker", "secid"])

        ticker_clause = ", ".join(f"'{t}'" for t in tickers)

        # Primary: try secnmd
        query = f"""
            SELECT ticker, secid
            FROM optionm.secnmd
            WHERE ticker IN ({ticker_clause})
            ORDER BY ticker, secid DESC
        """
        df = self.conn.raw_sql(query)

        # Fallback: securd with no issue_type filter for individual stocks
        if df.empty:
            print("[TickerToPermnoCrosswalk] secnmd returned 0 rows, "
                  "falling back to securd")
            query2 = f"""
                SELECT ticker, MIN(secid) AS secid
                FROM optionm.securd
                WHERE ticker IN ({ticker_clause})
                GROUP BY ticker
                ORDER BY ticker
            """
            df = self.conn.raw_sql(query2)

        # Keep highest secid per ticker (most recent issuance)
        df = (
            df.sort_values("secid", ascending=False)
              .drop_duplicates(subset="ticker", keep="first")
              [["ticker", "secid"]]
              .reset_index(drop=True)
        )
        return df

    def save_mapping(
        self,
        mapping_df: pd.DataFrame,
        etf_ticker: str,
        output_dir: str = "data/raw",
    ) -> None:
        """Persist the constituent ticker-to-id mapping used by downstream models."""
        path = Path(output_dir) / f"raw_constituent_mapping_{etf_ticker}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(path, index=False)
