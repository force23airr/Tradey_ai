"""
Feature engineering pipeline.

Reads from: price_history, markets, market_outcomes, macro_data
Writes to:  features table

Each row = one (market, snapshot_date) observation.
Features are computed at a snapshot T days before market close.
Label = resolved_yes (did YES win?).

This is structurally identical to building a PD model training dataset at a bank:
  - snapshot_date = observation date for the loan
  - days_to_close = time remaining on the facility
  - macro variables = CCAR stress inputs
  - resolved_yes   = default indicator (1 = event happened, 0 = did not)
"""

import sys
import logging
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.connection import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


FEATURE_SQL = """
-- ── Step 1: Daily price snapshots with rolling statistics ─────────────────
-- For each YES/Up token, compute rolling features over the price time series.
WITH price_features AS (
    SELECT
        ph.token_id,
        mo.market_id,
        ph.ts::DATE                                                      AS snapshot_date,

        -- Raw implied probability on this day
        ph.price                                                         AS yes_price,

        -- 7-day rolling average (smooths noise)
        AVG(ph.price) OVER (
            PARTITION BY ph.token_id
            ORDER BY ph.ts
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                                                AS rolling_avg_7d,

        -- 7-day rolling volatility (std dev of daily price)
        STDDEV(ph.price) OVER (
            PARTITION BY ph.token_id
            ORDER BY ph.ts
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                                                AS rolling_vol_7d,

        -- Price momentum: today vs 7 days ago (is market moving toward YES or NO?)
        ph.price - LAG(ph.price, 7) OVER (
            PARTITION BY ph.token_id ORDER BY ph.ts
        )                                                                AS momentum_7d,

        -- Price momentum: today vs 3 days ago
        ph.price - LAG(ph.price, 3) OVER (
            PARTITION BY ph.token_id ORDER BY ph.ts
        )                                                                AS momentum_3d,

        -- Distance from 50% (how confident is the market?)
        ABS(ph.price - 0.5)                                             AS conviction

    FROM price_history ph
    JOIN market_outcomes mo ON mo.token_id = ph.token_id
),

-- ── Step 2: Join market metadata ──────────────────────────────────────────
market_features AS (
    SELECT
        pf.*,
        m.question,
        m.category,
        m.resolved_yes,
        m.volume_total,
        m.liquidity,
        m.end_date::DATE                                                 AS end_date,

        -- Days remaining until market closes (urgency feature)
        -- Negative values = snapshot taken after end_date (data artifact)
        DATEDIFF('day', pf.snapshot_date, m.end_date::DATE)             AS days_to_close

    FROM price_features pf
    JOIN markets m ON m.id = pf.market_id
    WHERE m.resolved = true
      AND m.resolved_yes IS NOT NULL
),

-- ── Step 3: Join macro variables on snapshot_date ─────────────────────────
-- Pivot macro_data (long format) into wide columns.
-- This is the same join a credit risk team does:
--   loan snapshot + macro environment at that date = training row
macro_pivot AS (
    SELECT
        date,
        MAX(CASE WHEN ticker = 'vix'          THEN value END) AS vix,
        MAX(CASE WHEN ticker = 'sp500'         THEN value END) AS sp500,
        MAX(CASE WHEN ticker = 'treasury_10y'  THEN value END) AS treasury_10y,
        MAX(CASE WHEN ticker = 'treasury_5y'   THEN value END) AS treasury_5y,
        MAX(CASE WHEN ticker = 'tbill_13w'     THEN value END) AS tbill_13w,
        MAX(CASE WHEN ticker = 'gold'          THEN value END) AS gold,
        MAX(CASE WHEN ticker = 'tlt'           THEN value END) AS tlt
    FROM macro_data
    GROUP BY date
),

-- ── Step 4: Compute macro momentum features ───────────────────────────────
macro_with_momentum AS (
    SELECT
        mp.*,

        -- 7-day return on S&P 500 (risk-on/risk-off signal)
        (mp.sp500 - LAG(mp.sp500, 7) OVER (ORDER BY mp.date))
            / NULLIF(LAG(mp.sp500, 7) OVER (ORDER BY mp.date), 0)       AS sp500_return_7d,

        -- 7-day change in VIX (fear gauge direction)
        mp.vix - LAG(mp.vix, 7) OVER (ORDER BY mp.date)                 AS vix_change_7d,

        -- Yield curve slope: 10y - 3m (recession signal when negative)
        mp.treasury_10y - mp.tbill_13w                                  AS yield_curve_slope,

        -- 7-day gold return (safe haven demand)
        (mp.gold - LAG(mp.gold, 7) OVER (ORDER BY mp.date))
            / NULLIF(LAG(mp.gold, 7) OVER (ORDER BY mp.date), 0)        AS gold_return_7d

    FROM macro_pivot mp
)

-- ── Final: Assemble training dataset ──────────────────────────────────────
SELECT
    mf.market_id,
    mf.snapshot_date,
    mf.question,
    mf.category,

    -- Market features
    mf.yes_price,
    mf.rolling_avg_7d,
    COALESCE(mf.rolling_vol_7d, 0)      AS rolling_vol_7d,
    mf.momentum_7d,
    mf.momentum_3d,
    mf.conviction,
    mf.days_to_close,
    mf.volume_total,
    mf.liquidity,

    -- Macro features (environment at time of snapshot)
    mm.vix,
    mm.treasury_10y,
    mm.treasury_5y,
    mm.tbill_13w,
    mm.sp500_return_7d,
    mm.vix_change_7d,
    mm.yield_curve_slope,
    mm.gold_return_7d,

    -- Label
    CAST(mf.resolved_yes AS INTEGER)    AS label   -- 1 = YES won, 0 = NO won

FROM market_features mf
LEFT JOIN macro_with_momentum mm ON mm.date = mf.snapshot_date
WHERE mf.days_to_close BETWEEN 0 AND 180   -- only pre-close snapshots
  AND mf.yes_price IS NOT NULL
  AND mm.vix IS NOT NULL                    -- require macro data present
ORDER BY mf.market_id, mf.snapshot_date
"""


def build_features(write_to_db: bool = True) -> pd.DataFrame:
    """
    Run the feature engineering SQL and return a DataFrame.
    Optionally writes results to the features table in DuckDB.

    Returns:
        DataFrame with one row per (market, snapshot_date).
    """
    conn = get_conn()
    log.info("Building feature dataset...")

    df = conn.execute(FEATURE_SQL).df()
    log.info(f"  Feature rows built: {len(df)}")
    log.info(f"  Unique markets:     {df['market_id'].nunique()}")
    log.info(f"  Label distribution: YES={df['label'].sum()}  NO={(df['label']==0).sum()}")

    if write_to_db and not df.empty:
        # Write to features table
        conn.execute("DELETE FROM features")
        conn.execute("""
            INSERT INTO features (
                market_id, snapshot_date,
                yes_price, days_to_close, volume_total, liquidity,
                rolling_avg_7d, rolling_vol_7d, price_momentum_7d,
                vix, sp500_return_7d, treasury_10y, treasury_2y,
                gold_return_7d, resolved_yes
            )
            SELECT
                market_id, snapshot_date,
                yes_price, days_to_close, volume_total, liquidity,
                rolling_avg_7d, rolling_vol_7d, momentum_7d,
                vix, sp500_return_7d, treasury_10y, treasury_5y,
                gold_return_7d, CAST(label AS BOOLEAN)
            FROM df
        """)
        log.info("  Written to features table.")

    conn.close()
    return df


def summary(df: pd.DataFrame):
    """Print a quick EDA summary of the feature dataset."""
    print("\n=== Feature Dataset Summary ===")
    print(f"Shape:      {df.shape}")
    print(f"Markets:    {df['market_id'].nunique()}")
    print(f"Date range: {df['snapshot_date'].min()} → {df['snapshot_date'].max()}")
    print(f"\nLabel balance:")
    print(df['label'].value_counts().to_string())
    print(f"\nFeature stats:")
    cols = ['yes_price','rolling_vol_7d','momentum_7d','conviction',
            'days_to_close','vix','treasury_10y','yield_curve_slope']
    print(df[[c for c in cols if c in df.columns]].describe().round(3).to_string())


if __name__ == "__main__":
    df = build_features(write_to_db=True)
    summary(df)
