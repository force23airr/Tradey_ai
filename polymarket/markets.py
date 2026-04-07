"""
Helpers for merging Gamma and CLOB market data into unified structures
and extracting fields useful for modeling.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Outcome:
    name: str           # "Yes" / "No" / candidate name
    token_id: str       # ERC-1155 token ID
    price: float = 0.0  # current mid-market price in [0, 1]


@dataclass
class Market:
    id: str
    condition_id: str
    question: str
    category: str
    outcomes: list[Outcome] = field(default_factory=list)
    volume_24h: float = 0.0
    volume_total: float = 0.0
    liquidity: float = 0.0
    end_date: Optional[str] = None
    active: bool = True
    closed: bool = False
    resolved: bool = False
    resolution: Optional[str] = None  # "YES" | "NO" | outcome name

    @property
    def yes_price(self) -> Optional[float]:
        for o in self.outcomes:
            if o.name.upper() == "YES":
                return o.price
        return None

    @property
    def no_price(self) -> Optional[float]:
        for o in self.outcomes:
            if o.name.upper() == "NO":
                return o.price
        return None

    @property
    def implied_prob(self) -> Optional[float]:
        """YES price as implied probability (already in [0,1] on Polymarket)."""
        return self.yes_price


def parse_gamma_market(data: dict) -> Market:
    """Convert a raw Gamma API market dict into a Market dataclass."""
    outcomes_raw = data.get("outcomes", [])
    token_ids = data.get("clobTokenIds", [])

    outcomes = []
    for i, name in enumerate(outcomes_raw):
        token_id = token_ids[i] if i < len(token_ids) else ""
        outcomes.append(Outcome(name=name, token_id=token_id))

    return Market(
        id=str(data.get("id", "")),
        condition_id=data.get("conditionId", ""),
        question=data.get("question", ""),
        category=data.get("category", ""),
        outcomes=outcomes,
        volume_24h=float(data.get("volume24hr", 0) or 0),
        volume_total=float(data.get("volume", 0) or 0),
        liquidity=float(data.get("liquidity", 0) or 0),
        end_date=data.get("endDate"),
        active=data.get("active", False),
        closed=data.get("closed", False),
        resolved=data.get("resolved", False),
        resolution=data.get("resolution"),
    )


def attach_clob_prices(market: Market, midpoints: dict) -> Market:
    """
    Attach CLOB mid-market prices to a Market's outcomes in-place.

    Args:
        market:     A Market dataclass (already parsed from Gamma).
        midpoints:  Dict of {token_id: price} from CLOBAPI.get_midpoints().
    """
    for outcome in market.outcomes:
        if outcome.token_id in midpoints:
            outcome.price = float(midpoints[outcome.token_id])
    return market
