"""
Feature engineering pipeline — FX/rates/macro focused.

Reads from: price_history, markets, market_outcomes, macro_data
Writes to:  features table

Each row = one (market, snapshot_date) observation taken before resolution.
Label = resolved_yes (did the positive outcome happen?).

Key new features for time-series threshold markets:
  distance_to_threshold  — how far is the underlying asset from the target price?
  trend_velocity         — is the price moving toward or away from the threshold?
  days_to_close          — time remaining (urgency)

These three together answer: "Is the asset on track to hit the target in time?"
This is the core of any FX/rates threshold prediction model.
"""

import sys
import re
import logging
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.connection import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


FEATURE_SQL = """
-- ── Step 1: Price features via window functions ───────────────────────────
WITH price_features AS (
    SELECT
        ph.token_id,
        mo.market_id,
        ph.ts::DATE                                                       AS snapshot_date,
        ph.price                                                          AS yes_price,

        -- 7-day rolling average (smooths daily noise)
        AVG(ph.price) OVER (
            PARTITION BY ph.token_id
            ORDER BY ph.ts
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                                                 AS rolling_avg_7d,

        -- 7-day rolling volatility
        STDDEV(ph.price) OVER (
            PARTITION BY ph.token_id
            ORDER BY ph.ts
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                                                 AS rolling_vol_7d,

        -- Momentum: price change over 7 days
        ph.price - LAG(ph.price, 7) OVER (
            PARTITION BY ph.token_id ORDER BY ph.ts
        )                                                                 AS momentum_7d,

        -- Momentum: price change over 3 days
        ph.price - LAG(ph.price, 3) OVER (
            PARTITION BY ph.token_id ORDER BY ph.ts
        )                                                                 AS momentum_3d,

        -- Conviction: how far from 50/50
        ABS(ph.price - 0.5)                                              AS conviction

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
        m.end_date::DATE                                                  AS end_date,
        DATEDIFF('day', pf.snapshot_date, m.end_date::DATE)              AS days_to_close

    FROM price_features pf
    JOIN markets m ON m.id = pf.market_id
    WHERE m.resolved = true
      AND m.resolved_yes IS NOT NULL
),

-- ── Step 3: Pivot macro data into wide columns ─────────────────────────────
macro_pivot AS (
    SELECT
        date,
        MAX(CASE WHEN ticker = 'vix'          THEN value END) AS vix,
        MAX(CASE WHEN ticker = 'sp500'        THEN value END) AS sp500,
        MAX(CASE WHEN ticker = 'treasury_10y' THEN value END) AS treasury_10y,
        MAX(CASE WHEN ticker = 'treasury_5y'  THEN value END) AS treasury_5y,
        MAX(CASE WHEN ticker = 'tbill_13w'    THEN value END) AS tbill_13w,
        MAX(CASE WHEN ticker = 't10y2y'       THEN value END) AS t10y2y,
        MAX(CASE WHEN ticker = 't10y3m'       THEN value END) AS t10y3m,
        MAX(CASE WHEN ticker = 'usdjpy'       THEN value END) AS usdjpy,
        MAX(CASE WHEN ticker = 'eurusd'       THEN value END) AS eurusd,
        MAX(CASE WHEN ticker = 'gbpusd'       THEN value END) AS gbpusd,
        MAX(CASE WHEN ticker = 'dxy'          THEN value END) AS dxy,
        MAX(CASE WHEN ticker = 'gold'         THEN value END) AS gold,
        MAX(CASE WHEN ticker = 'oil_wti'      THEN value END) AS oil_wti,
        MAX(CASE WHEN ticker = 'bitcoin'      THEN value END) AS bitcoin,
        MAX(CASE WHEN ticker = 'fed_funds'    THEN value END) AS fed_funds,
        MAX(CASE WHEN ticker = 'cpi'          THEN value END) AS cpi,
        MAX(CASE WHEN ticker = 'unemployment' THEN value END) AS unemployment
    FROM macro_data
    GROUP BY date
),

-- ── Step 4: Macro momentum features ──────────────────────────────────────
macro_features AS (
    SELECT
        mp.*,
        -- S&P 7-day return (risk appetite)
        (mp.sp500 - LAG(mp.sp500, 7) OVER (ORDER BY mp.date))
            / NULLIF(LAG(mp.sp500, 7) OVER (ORDER BY mp.date), 0)        AS sp500_return_7d,

        -- VIX change (fear direction)
        mp.vix - LAG(mp.vix, 7) OVER (ORDER BY mp.date)                  AS vix_change_7d,

        -- Yield curve slope: 10y minus 3m (recession signal)
        mp.treasury_10y - mp.tbill_13w                                   AS yield_curve_slope,

        -- USD/JPY 7-day momentum
        mp.usdjpy - LAG(mp.usdjpy, 7) OVER (ORDER BY mp.date)            AS usdjpy_momentum_7d,

        -- EUR/USD 7-day momentum
        mp.eurusd - LAG(mp.eurusd, 7) OVER (ORDER BY mp.date)            AS eurusd_momentum_7d,

        -- DXY 7-day momentum (dollar strength)
        mp.dxy - LAG(mp.dxy, 7) OVER (ORDER BY mp.date)                  AS dxy_momentum_7d,

        -- Gold 7-day return
        (mp.gold - LAG(mp.gold, 7) OVER (ORDER BY mp.date))
            / NULLIF(LAG(mp.gold, 7) OVER (ORDER BY mp.date), 0)         AS gold_return_7d,

        -- Oil 7-day return
        (mp.oil_wti - LAG(mp.oil_wti, 7) OVER (ORDER BY mp.date))
            / NULLIF(LAG(mp.oil_wti, 7) OVER (ORDER BY mp.date), 0)      AS oil_return_7d

    FROM macro_pivot mp
)

-- ── Final: Training dataset ───────────────────────────────────────────────
SELECT
    mf.market_id,
    mf.snapshot_date,
    mf.question,
    mf.category,

    -- Market price features
    mf.yes_price,
    mf.rolling_avg_7d,
    COALESCE(mf.rolling_vol_7d, 0)       AS rolling_vol_7d,
    mf.momentum_7d,
    mf.momentum_3d,
    mf.conviction,
    mf.days_to_close,
    mf.volume_total,
    mf.liquidity,

    -- Macro environment at snapshot date
    mm.vix,
    mm.treasury_10y,
    mm.treasury_5y,
    mm.t10y2y,
    mm.t10y3m,
    mm.fed_funds,
    mm.cpi,
    mm.unemployment,
    mm.usdjpy,
    mm.eurusd,
    mm.gbpusd,
    mm.dxy,
    mm.gold,
    mm.oil_wti,
    mm.sp500_return_7d,
    mm.vix_change_7d,
    mm.yield_curve_slope,
    mm.usdjpy_momentum_7d,
    mm.eurusd_momentum_7d,
    mm.dxy_momentum_7d,
    mm.gold_return_7d,
    mm.oil_return_7d,

    -- Label
    CAST(mf.resolved_yes AS INTEGER)     AS label

FROM market_features mf
LEFT JOIN macro_features mm ON mm.date = mf.snapshot_date
WHERE mf.days_to_close BETWEEN 0 AND 365
  AND mf.yes_price IS NOT NULL
  AND mm.vix IS NOT NULL
ORDER BY mf.market_id, mf.snapshot_date
"""


def extract_threshold(question: str) -> float | None:
    """
    Parse the numeric price threshold from a market question.

    Examples:
      "Will BTC be above $50,000 by Dec 31?"  → 50000.0
      "Will USD/JPY cross 150?"                → 150.0
      "Will 10-year yield exceed 5%?"          → 5.0
      "Will the Fed cut rates?"                → None (no threshold)
    """
    # Match patterns like $50,000 or 50,000 or 150.5 or 5%
    patterns = [
        r'\$([0-9,]+(?:\.[0-9]+)?)',   # $50,000 or $1.85
        r'([0-9,]+(?:\.[0-9]+)?)\s*%', # 5% or 3.5%
        r'\b([0-9]{2,}(?:,[0-9]{3})*(?:\.[0-9]+)?)\b',  # bare numbers like 150 or 50000
    ]
    for pat in patterns:
        m = re.search(pat, question)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def build_features(write_to_db: bool = True) -> pd.DataFrame:
    """
    Run feature engineering SQL and optionally write to features table.
    Also extracts threshold values from question text.
    """
    conn = get_conn()
    log.info("Building feature dataset...")

    df = conn.execute(FEATURE_SQL).df()

    # Extract numeric threshold from question text
    df["threshold"] = df["question"].apply(extract_threshold)

    log.info(f"  Rows:               {len(df)}")
    log.info(f"  Unique markets:     {df['market_id'].nunique()}")
    log.info(f"  With threshold:     {df['threshold'].notna().sum()} rows ({df['threshold'].notna().mean():.0%})")
    log.info(f"  Label: YES={df['label'].sum()}  NO={(df['label']==0).sum()}")

    if write_to_db and not df.empty:
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
    print("\n=== Feature Dataset Summary ===")
    print(f"Shape:      {df.shape}")
    print(f"Markets:    {df['market_id'].nunique()}")
    print(f"Date range: {df['snapshot_date'].min()} → {df['snapshot_date'].max()}")
    print(f"\nLabel balance:\n{df['label'].value_counts().to_string()}")
    cols = ['yes_price','rolling_vol_7d','momentum_7d','conviction',
            'days_to_close','vix','treasury_10y','yield_curve_slope',
            'usdjpy','dxy','fed_funds']
    print(f"\nKey feature stats:")
    print(df[[c for c in cols if c in df.columns]].describe().round(3).to_string())

    print(f"\nSample market questions:")
    sample = df[['question','threshold','label']].drop_duplicates('question').head(15)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    df = build_features(write_to_db=True)
    summary(df)
