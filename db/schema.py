"""
Create all database tables.
Run once (or any time) — all statements use CREATE TABLE IF NOT EXISTS.
"""

from db.connection import get_conn

SCHEMA = """
-- ── Markets ────────────────────────────────────────────────────────────────
-- One row per Polymarket market (question).
CREATE TABLE IF NOT EXISTS markets (
    id              VARCHAR PRIMARY KEY,
    condition_id    VARCHAR,
    question        VARCHAR,
    category        VARCHAR,
    end_date        TIMESTAMP,
    resolved        BOOLEAN DEFAULT FALSE,
    resolved_yes    BOOLEAN,           -- TRUE = YES won, FALSE = NO won, NULL = unresolved
    resolution      VARCHAR,           -- raw resolution string from API
    volume_total    DOUBLE DEFAULT 0,
    liquidity       DOUBLE DEFAULT 0,
    fetched_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Market Outcomes (tokens) ───────────────────────────────────────────────
-- Each market has 2+ outcomes (YES/NO or multiple candidates).
-- Each outcome has a unique ERC-1155 token_id used for CLOB pricing.
CREATE TABLE IF NOT EXISTS market_outcomes (
    token_id    VARCHAR PRIMARY KEY,
    market_id   VARCHAR REFERENCES markets(id),
    name        VARCHAR    -- "Yes", "No", or candidate name
);

-- ── Price History ──────────────────────────────────────────────────────────
-- Daily price snapshots for each outcome token.
-- price is in [0, 1] — represents implied probability.
CREATE TABLE IF NOT EXISTS price_history (
    token_id    VARCHAR,
    ts          TIMESTAMP,
    price       DOUBLE,
    PRIMARY KEY (token_id, ts)
);

-- ── Macro Data ─────────────────────────────────────────────────────────────
-- Daily values for macro/market indicators (VIX, S&P, yields, etc.)
-- ticker matches yfinance symbols or FRED series IDs.
CREATE TABLE IF NOT EXISTS macro_data (
    date        DATE,
    ticker      VARCHAR,
    value       DOUBLE,
    PRIMARY KEY (date, ticker)
);

-- ── Features (built by feature engineering queries) ───────────────────────
-- One row per (market, snapshot_date) — ready to feed into a model.
CREATE TABLE IF NOT EXISTS features (
    market_id           VARCHAR,
    snapshot_date       DATE,

    -- Market features
    yes_price           DOUBLE,     -- implied probability at snapshot
    days_to_close       INTEGER,    -- days until market end_date
    volume_total        DOUBLE,
    liquidity           DOUBLE,
    rolling_avg_7d      DOUBLE,     -- 7-day rolling avg yes price
    rolling_vol_7d      DOUBLE,     -- 7-day rolling std dev (volatility)
    price_momentum_7d   DOUBLE,     -- yes_price - yes_price_7_days_ago

    -- Macro features (joined by snapshot_date)
    vix                 DOUBLE,
    sp500_return_7d     DOUBLE,     -- S&P 7-day return
    treasury_10y        DOUBLE,
    treasury_2y         DOUBLE,
    gold_return_7d      DOUBLE,

    -- Label
    resolved_yes        BOOLEAN,    -- 1 = YES won, 0 = NO won

    PRIMARY KEY (market_id, snapshot_date)
);
"""


def init_db():
    conn = get_conn()
    for statement in SCHEMA.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
    conn.close()
    print("Database schema initialized.")


if __name__ == "__main__":
    init_db()
