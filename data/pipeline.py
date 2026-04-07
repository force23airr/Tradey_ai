"""
Main data collection pipeline.

Run this script to populate the database with:
  1. Resolved Polymarket markets + outcome tokens
  2. Full price history for each market
  3. Macro/market indicator data (VIX, S&P, yields, gold)

Usage:
    python data/pipeline.py                  # full run
    python data/pipeline.py --markets-only   # skip price history + macro
    python data/pipeline.py --macro-only     # only pull macro data
    python data/pipeline.py --limit 200      # cap markets at 200 for quick test
"""

import argparse
import logging
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.schema import init_db
from data.polymarket.fetcher import PolymarketFetcher
from data.macro.fetcher import MacroFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run(markets_only=False, macro_only=False, limit=1000, macro_start="2021-01-01"):
    log.info("=" * 60)
    log.info("Tradey_ai — Data Collection Pipeline")
    log.info("=" * 60)

    # Step 1: Initialize DB schema
    log.info("[1/4] Initializing database schema...")
    init_db()

    if not macro_only:
        # Step 2: Fetch resolved markets
        log.info(f"[2/4] Fetching resolved Polymarket markets (limit={limit})...")
        pm = PolymarketFetcher()
        n_markets = pm.fetch_resolved_markets(max_markets=limit)
        log.info(f"      Stored {n_markets} markets.")

        if not markets_only:
            # Step 3: Fetch price history
            log.info("[3/4] Fetching price history for all YES tokens...")
            pm.fetch_price_histories(max_markets=limit)
        else:
            log.info("[3/4] Skipping price history (--markets-only flag set).")

        pm.close()
    else:
        log.info("[2/4] Skipping Polymarket fetch (--macro-only flag set).")
        log.info("[3/4] Skipping price history  (--macro-only flag set).")

    # Step 4: Fetch macro data
    log.info(f"[4/4] Fetching macro data from {macro_start} to today...")
    mf = MacroFetcher()
    results = mf.fetch_all(start=macro_start)
    summary = mf.summary()
    mf.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Pipeline complete. Database summary:")
    log.info("=" * 60)
    log.info("\nMacro data:")
    print(summary.to_string(index=False))
    log.info("")
    log.info("Run notebooks/research/ to start exploring the data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tradey_ai data pipeline")
    parser.add_argument("--markets-only", action="store_true", help="Only fetch markets, skip price history + macro")
    parser.add_argument("--macro-only",   action="store_true", help="Only fetch macro data")
    parser.add_argument("--limit",        type=int, default=1000, help="Max markets to fetch (default 1000)")
    parser.add_argument("--macro-start",  type=str, default="2021-01-01", help="Macro data start date YYYY-MM-DD")
    args = parser.parse_args()

    run(
        markets_only=args.markets_only,
        macro_only=args.macro_only,
        limit=args.limit,
        macro_start=args.macro_start,
    )
