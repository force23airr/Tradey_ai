"""CLOB API client — orderbook, pricing, and price history (read-only)."""

from config.settings import CLOB_API_BASE
from polymarket.client import PolymarketClient


class CLOBAPI(PolymarketClient):
    def __init__(self):
        super().__init__(CLOB_API_BASE)

    # ── Markets ───────────────────────────────────────────────────────────────

    def get_markets(self, next_cursor: str = "MA==") -> dict:
        """
        Paginate through all CLOB markets.

        Returns a dict with 'data' (list of markets) and 'next_cursor'.
        Pass next_cursor from the previous response to get the next page.
        Use next_cursor='LTE=' to signal end of pagination.
        """
        return self.get("/markets", params={"next_cursor": next_cursor})

    def get_market(self, condition_id: str) -> dict:
        """Fetch a single CLOB market by its condition ID."""
        return self.get(f"/markets/{condition_id}")

    # ── Orderbook ─────────────────────────────────────────────────────────────

    def get_orderbook(self, token_id: str) -> dict:
        """
        Fetch the live orderbook for a token (outcome).

        token_id is the ERC-1155 token ID for a specific outcome
        (YES or NO side of a market). Each market has two token IDs.

        Returns bids and asks with price (0-1) and size (USDC).
        """
        return self.get(f"/book", params={"token_id": token_id})

    def get_orderbooks(self, token_ids: list[str]) -> list[dict]:
        """Fetch orderbooks for multiple tokens in one call."""
        params = [("token_id", tid) for tid in token_ids]
        # Build query string manually for repeated params
        from urllib.parse import urlencode
        query = urlencode(params)
        import requests as _req
        url = f"{self.base_url}/books?{query}"
        self._throttle()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Pricing ───────────────────────────────────────────────────────────────

    def get_midpoints(self, token_ids: list[str]) -> dict:
        """
        Fetch mid-market prices for a list of token IDs.
        Returns {token_id: price} where price is in [0, 1].
        """
        params = {"token_id": token_ids[0]} if len(token_ids) == 1 else None
        if len(token_ids) == 1:
            result = self.get("/midpoint", params={"token_id": token_ids[0]})
            return {token_ids[0]: float(result.get("mid", 0))}
        # Batch endpoint
        from urllib.parse import urlencode
        query = urlencode([("token_id", tid) for tid in token_ids])
        self._throttle()
        import requests as _req
        resp = self.session.get(f"{self.base_url}/midpoints?{query}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_spread(self, token_id: str) -> dict:
        """Fetch bid-ask spread for a token. Returns {'spread': float}."""
        return self.get("/spread", params={"token_id": token_id})

    def get_last_trade_price(self, token_id: str) -> dict:
        """Fetch the last traded price for a token."""
        return self.get("/last-trade-price", params={"token_id": token_id})

    # ── Price History ─────────────────────────────────────────────────────────

    def get_price_history(
        self,
        token_id: str,
        interval: str = "1d",
        start_ts: int = None,
        end_ts: int = None,
        fidelity: int = 60,
    ) -> list[dict]:
        """
        Fetch historical prices for a token.

        Args:
            token_id:  ERC-1155 token ID for the outcome.
            interval:  Time range shorthand — '1h' | '6h' | '1d' | '1w' | 'all'.
            start_ts:  Unix timestamp for range start (overrides interval).
            end_ts:    Unix timestamp for range end.
            fidelity:  Data point resolution in minutes (60 = hourly, 1440 = daily).

        Returns:
            List of {'t': unix_ts, 'p': price} dicts sorted ascending by time.
        """
        params: dict = {"market": token_id, "fidelity": fidelity}
        if start_ts and end_ts:
            params["startTs"] = start_ts
            params["endTs"] = end_ts
        else:
            params["interval"] = interval
        return self.get("/prices-history", params=params)
