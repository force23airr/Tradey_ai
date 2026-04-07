"""
Polymarket historical data fetcher.
Pulls resolved markets from Gamma API and their full price history from CLOB API.
Stores everything into DuckDB.
"""

import json
import logging
from datetime import datetime
from db.connection import get_conn
from polymarket.gamma import GammaAPI
from polymarket.clob import CLOBAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class PolymarketFetcher:
    def __init__(self):
        self.gamma = GammaAPI()
        self.clob  = CLOBAPI()
        self.conn  = get_conn()

    # ── Markets ───────────────────────────────────────────────────────────────

    def fetch_resolved_markets(self, max_markets: int = 1000) -> int:
        """
        Paginate through all resolved/closed Polymarket markets and store them.

        Args:
            max_markets: Cap on how many markets to fetch (for initial runs).

        Returns:
            Number of markets inserted/updated.
        """
        log.info(f"Fetching resolved markets (max={max_markets})...")
        inserted = 0
        offset   = 0
        limit    = 100

        while inserted < max_markets:
            batch = self.gamma.get_markets(
                limit=limit,
                offset=offset,
                closed=True,
                order="volume",
            )
            if not batch:
                break

            for m in batch:
                self._upsert_market(m)
                inserted += 1

            log.info(f"  fetched {inserted} markets so far...")
            offset += limit

            if len(batch) < limit:
                break  # last page

        log.info(f"Done. Total markets stored: {inserted}")
        return inserted

    def _upsert_market(self, m: dict):
        """Insert or update a single market row + its outcome tokens."""
        # outcomes and clobTokenIds come back as JSON strings per the API schema
        outcomes      = m.get("outcomes", "[]") or "[]"
        token_ids_raw = m.get("clobTokenIds", "[]") or "[]"
        outcome_prices_raw = m.get("outcomePrices", "[]") or "[]"

        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids_raw, str):
            token_ids_raw = json.loads(token_ids_raw)
        if isinstance(outcome_prices_raw, str):
            outcome_prices_raw = json.loads(outcome_prices_raw)

        outcome_prices = [float(p) for p in outcome_prices_raw] if outcome_prices_raw else []

        # Determine resolution from outcomePrices.
        # For a resolved binary market, the winning outcome trades at ~1.0.
        # Strategy: find the outcome with the highest price — if it's above 0.5
        # and the market is closed, that outcome won.
        # outcomes[0] is the "positive" outcome (Yes / Up / candidate A).
        resolved_yes = None
        closed = bool(m.get("closed", False))
        if closed and outcome_prices and len(outcome_prices) >= 2:
            max_price = max(outcome_prices)
            max_idx   = outcome_prices.index(max_price)
            if max_price >= 0.5:
                # First outcome (Yes/Up) won → resolved_yes=True
                # Any other outcome won → resolved_yes=False
                resolved_yes = (max_idx == 0)

        end_date = m.get("endDateIso") or m.get("endDate")

        self.conn.execute("""
            INSERT OR REPLACE INTO markets
                (id, condition_id, question, category, end_date,
                 resolved, resolved_yes, resolution, volume_total, liquidity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(m.get("id", "")),
            m.get("conditionId", ""),
            m.get("question", ""),
            m.get("category", "") or m.get("groupItemTitle", ""),
            end_date,
            closed and resolved_yes is not None,
            resolved_yes,
            "YES" if resolved_yes is True else ("NO" if resolved_yes is False else None),
            float(m.get("volumeNum", 0) or m.get("volume", 0) or 0),
            float(m.get("liquidityNum", 0) or m.get("liquidity", 0) or 0),
        ])

        # Store outcome tokens
        for i, name in enumerate(outcomes):
            token_id = token_ids_raw[i] if i < len(token_ids_raw) else None
            if token_id:
                self.conn.execute("""
                    INSERT OR IGNORE INTO market_outcomes (token_id, market_id, name)
                    VALUES (?, ?, ?)
                """, [token_id, str(m.get("id", "")), name])

    # ── Price History ─────────────────────────────────────────────────────────

    def fetch_price_histories(self, max_markets: int = None):
        """
        For every YES token in DB, fetch full daily price history from CLOB.

        Args:
            max_markets: Limit how many markets to pull history for (None = all).
        """
        # Fetch the first/positive outcome token for each market.
        # This covers Yes/No markets AND Up/Down crypto markets.
        # We identify the "positive" token as the one paired with the winning side (index 0).
        query = """
            SELECT mo.token_id, mo.market_id, mo.name
            FROM market_outcomes mo
            JOIN markets m ON m.id = mo.market_id
            WHERE mo.name IN ('Yes', 'Up')
            ORDER BY m.volume_total DESC
        """
        if max_markets:
            query += f" LIMIT {max_markets}"

        tokens = self.conn.execute(query).fetchall()
        log.info(f"Fetching price history for {len(tokens)} YES tokens...")

        for i, (token_id, market_id, name) in enumerate(tokens):
            try:
                response = self.clob.get_price_history(
                    token_id=token_id,
                    interval="all",
                    fidelity=1440,  # daily resolution
                )
                # API returns {"history": [{t, p}, ...]} per the OpenAPI spec
                history = response.get("history", []) if isinstance(response, dict) else response
                self._store_price_history(token_id, history)
                if (i + 1) % 50 == 0:
                    log.info(f"  price history: {i+1}/{len(tokens)} done")
            except Exception as e:
                log.warning(f"  failed token {token_id[:12]}...: {e}")
                continue

        log.info("Price history fetch complete.")

    def _store_price_history(self, token_id: str, history: list[dict]):
        """Insert price history rows, skipping duplicates."""
        rows = []
        for point in history:
            ts    = point.get("t")
            price = point.get("p")
            if ts is None or price is None:
                continue
            dt = datetime.utcfromtimestamp(int(ts))
            rows.append((token_id, dt, float(price)))

        if rows:
            self.conn.executemany("""
                INSERT OR IGNORE INTO price_history (token_id, ts, price)
                VALUES (?, ?, ?)
            """, rows)

    def close(self):
        self.conn.close()
