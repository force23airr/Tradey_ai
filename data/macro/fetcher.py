"""
Macro data fetcher using yfinance — no API key required.
Pulls daily historical data for key macro/market indicators and stores in DuckDB.

Tickers:
    ^VIX    — CBOE Volatility Index (market fear gauge)
    ^GSPC   — S&P 500 Index
    ^TNX    — 10-Year Treasury Yield
    ^FVX    — 5-Year Treasury Yield
    ^IRX    — 13-Week T-Bill Rate (Fed funds proxy)
    GLD     — Gold ETF (inflation hedge proxy)
    TLT     — 20+ Year Treasury Bond ETF (rate sensitivity)
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import date
from db.connection import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MACRO_TICKERS = {
    "vix":          "^VIX",
    "sp500":        "^GSPC",
    "treasury_10y": "^TNX",
    "treasury_5y":  "^FVX",
    "tbill_13w":    "^IRX",
    "gold":         "GLD",
    "tlt":          "TLT",
}


class MacroFetcher:
    def __init__(self):
        self.conn = get_conn()

    def fetch_all(self, start: str = "2021-01-01", end: str = None) -> dict[str, int]:
        """
        Fetch historical daily data for all macro tickers and store in DuckDB.

        Args:
            start: Start date string 'YYYY-MM-DD'. Default: 2021-01-01
                   (covers most of Polymarket's history).
            end:   End date string. Defaults to today.

        Returns:
            Dict of {ticker_name: rows_inserted}.
        """
        if end is None:
            end = str(date.today())

        results = {}
        for name, ticker in MACRO_TICKERS.items():
            try:
                rows = self._fetch_ticker(name, ticker, start, end)
                results[name] = rows
                log.info(f"  {name} ({ticker}): {rows} rows stored")
            except Exception as e:
                log.warning(f"  {name} ({ticker}) failed: {e}")
                results[name] = 0

        return results

    def _fetch_ticker(self, name: str, ticker: str, start: str, end: str) -> int:
        """Download one ticker and insert into macro_data table."""
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        if df.empty:
            log.warning(f"No data returned for {ticker}")
            return 0

        # Use closing price as the daily value
        closes = df["Close"].dropna()

        # yfinance can return MultiIndex columns — flatten if needed
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]

        rows = [
            (pd.Timestamp(dt).date(), name, float(val))
            for dt, val in closes.items()
        ]

        self.conn.executemany("""
            INSERT OR REPLACE INTO macro_data (date, ticker, value)
            VALUES (?, ?, ?)
        """, rows)

        return len(rows)

    def summary(self) -> pd.DataFrame:
        """Return a summary of what's in the macro_data table."""
        return self.conn.execute("""
            SELECT ticker,
                   COUNT(*)        AS rows,
                   MIN(date)       AS from_date,
                   MAX(date)       AS to_date,
                   AVG(value)      AS avg_value
            FROM macro_data
            GROUP BY ticker
            ORDER BY ticker
        """).df()

    def close(self):
        self.conn.close()
