"""
Macro data fetcher — yfinance (FX, equities, commodities) + FRED (economic releases).
No API key needed for yfinance. FRED key loaded from .env.
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from db.connection import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── yfinance tickers ──────────────────────────────────────────────────────────
YFINANCE_TICKERS = {
    # Rates & bonds
    "treasury_10y":  "^TNX",
    "treasury_5y":   "^FVX",
    "treasury_2y":   "^TYX",
    "tbill_13w":     "^IRX",
    "tlt":           "TLT",       # 20yr bond ETF

    # Equities / volatility
    "sp500":         "^GSPC",
    "vix":           "^VIX",

    # FX pairs
    "usdjpy":        "USDJPY=X",
    "eurusd":        "EURUSD=X",
    "gbpusd":        "GBPUSD=X",
    "usdcnh":        "USDCNH=X",  # USD/CNH (offshore yuan)
    "dxy":           "DX-Y.NYB",  # Dollar index

    # Commodities
    "gold":          "GLD",
    "oil_wti":       "CL=F",      # WTI crude futures
}

# ── FRED series ───────────────────────────────────────────────────────────────
FRED_SERIES = {
    "fed_funds":      "FEDFUNDS",    # Effective Fed funds rate (monthly)
    "cpi":            "CPIAUCSL",    # CPI all items (monthly)
    "core_cpi":       "CPILFESL",    # Core CPI ex food/energy (monthly)
    "pce":            "PCEPI",       # PCE price index (Fed preferred, monthly)
    "core_pce":       "PCEPILFE",    # Core PCE (monthly)
    "unemployment":   "UNRATE",      # Unemployment rate (monthly)
    "gdp":            "GDP",         # Nominal GDP (quarterly)
    "t10y2y":         "T10Y2Y",      # 10yr-2yr spread (daily — best recession indicator)
    "t10y3m":         "T10Y3M",      # 10yr-3m spread (daily)
    "nfp":            "PAYEMS",      # Nonfarm payrolls (monthly)
}


class MacroFetcher:
    def __init__(self):
        self.conn = get_conn()
        self.fred_key = os.getenv("FRED_API_KEY")
        if not self.fred_key:
            log.warning("FRED_API_KEY not found in .env — FRED data will be skipped.")

    def fetch_all(self, start: str = "2021-01-01", end: str = None) -> dict:
        if end is None:
            end = str(date.today())

        results = {}

        log.info("Fetching yfinance tickers...")
        for name, ticker in YFINANCE_TICKERS.items():
            try:
                rows = self._fetch_yfinance(name, ticker, start, end)
                results[name] = rows
                log.info(f"  {name} ({ticker}): {rows} rows")
            except Exception as e:
                log.warning(f"  {name} failed: {e}")
                results[name] = 0

        if self.fred_key:
            log.info("Fetching FRED series...")
            for name, series_id in FRED_SERIES.items():
                try:
                    rows = self._fetch_fred(name, series_id, start, end)
                    results[name] = rows
                    log.info(f"  {name} ({series_id}): {rows} rows")
                except Exception as e:
                    log.warning(f"  {name} ({series_id}) failed: {e}")
                    results[name] = 0

        return results

    def _fetch_yfinance(self, name: str, ticker: str, start: str, end: str) -> int:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return 0

        closes = df["Close"].dropna()
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]

        rows = [(pd.Timestamp(dt).date(), name, float(val)) for dt, val in closes.items()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO macro_data (date, ticker, value) VALUES (?, ?, ?)", rows
        )
        return len(rows)

    def _fetch_fred(self, name: str, series_id: str, start: str, end: str) -> int:
        import urllib.request, json

        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&observation_start={start}"
            f"&observation_end={end}"
            f"&api_key={self.fred_key}"
            f"&file_type=json"
        )
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())

        observations = data.get("observations", [])
        rows = []
        for obs in observations:
            val = obs.get("value", ".")
            if val == ".":   # FRED uses "." for missing values
                continue
            rows.append((obs["date"], name, float(val)))

        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO macro_data (date, ticker, value) VALUES (?, ?, ?)", rows
            )
        return len(rows)

    def summary(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT ticker,
                   COUNT(*)   AS rows,
                   MIN(date)  AS from_date,
                   MAX(date)  AS to_date
            FROM macro_data
            GROUP BY ticker
            ORDER BY ticker
        """).df()

    def close(self):
        self.conn.close()
