"""Gamma API client — market discovery, events, tags, search."""

from config.settings import GAMMA_API_BASE
from polymarket.client import PolymarketClient


class GammaAPI(PolymarketClient):
    def __init__(self):
        super().__init__(GAMMA_API_BASE)

    # ── Markets ──────────────────────────────────────────────────────────────

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = None,
        closed: bool = None,
        tag: str = None,
        tag_id: int = None,
        order: str = "volume24hr",
    ) -> list[dict]:
        """
        Fetch a paginated list of markets.

        Args:
            limit:  Number of markets to return (max 100).
            offset: Pagination offset.
            active: Filter to active markets only.
            closed: Filter to closed/resolved markets.
            tag:    Filter by tag slug (e.g. 'politics', 'crypto').
            tag_id: Filter by tag integer ID.
            order:  Sort field — 'volume24hr' | 'startDate' | 'endDate'.
        """
        params = {"limit": limit, "offset": offset, "order": order}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if tag:
            params["tag"] = tag
        if tag_id is not None:
            params["tag_id"] = tag_id
        return self.get("/markets", params=params)

    def get_market(self, market_id: str) -> dict:
        """Fetch a single market by its ID."""
        return self.get(f"/markets/{market_id}")

    # ── Events ───────────────────────────────────────────────────────────────

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = None,
        closed: bool = None,
        tag: str = None,
    ) -> list[dict]:
        """
        Fetch a paginated list of events.
        Events are collections of related markets (e.g. an election event
        containing multiple candidate markets).
        """
        params = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if tag:
            params["tag"] = tag
        return self.get("/events", params=params)

    def get_event(self, event_id: str) -> dict:
        """Fetch a single event by its ID."""
        return self.get(f"/events/{event_id}")

    # ── Tags ─────────────────────────────────────────────────────────────────

    def get_tags(self) -> list[dict]:
        """Fetch all available market tags/categories."""
        return self.get("/tags")

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across markets and events."""
        return self.get("/search", params={"term": query, "limit": limit})
