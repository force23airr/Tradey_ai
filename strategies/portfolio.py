"""
Kelly criterion bet sizing for binary Polymarket positions.

Polymarket mechanics:
  - YES token costs yes_price dollars. Pays $1 if YES wins.
    Profit per dollar staked = (1 - yes_price) / yes_price  (the "odds")
  - NO token costs (1 - yes_price) dollars. Pays $1 if NO wins.
    Profit per dollar staked = yes_price / (1 - yes_price)

Kelly formula: f* = (b*p - q) / b
  where b = net odds, p = model probability of winning, q = 1 - p

Half-Kelly (f* * 0.5) is standard practice — accounts for model error,
reduces risk of ruin, and has only modest impact on long-run returns.
"""


def kelly_yes(model_prob: float, yes_price: float) -> float:
    """
    Full Kelly fraction for a YES bet.

    Args:
        model_prob: model's estimated P(YES)
        yes_price:  market's implied P(YES) — the price of a YES token

    Returns:
        Fraction of bankroll to bet (positive = bet, may exceed 1.0 before capping)
    """
    if yes_price <= 0 or yes_price >= 1:
        return 0.0
    b = (1.0 - yes_price) / yes_price   # net odds on a YES token
    p = model_prob
    q = 1.0 - model_prob
    return (b * p - q) / b


def kelly_no(model_prob: float, yes_price: float) -> float:
    """
    Full Kelly fraction for a NO bet.

    Args:
        model_prob: model's estimated P(YES)  — so P(NO) = 1 - model_prob
        yes_price:  market's implied P(YES); NO token costs (1 - yes_price)

    Returns:
        Fraction of bankroll to bet on NO (positive)
    """
    no_price = 1.0 - yes_price
    if no_price <= 0 or no_price >= 1:
        return 0.0
    b = yes_price / no_price    # net odds on a NO token
    p = 1.0 - model_prob        # model's P(NO)
    q = model_prob              # model's P(YES) = P(NO bet loses)
    return (b * p - q) / b


def size_bet(
    model_prob: float,
    yes_price: float,
    bankroll: float,
    edge_threshold: float = 0.05,
    kelly_multiplier: float = 0.5,
    max_fraction: float = 0.10,
) -> dict:
    """
    Compute bet size given model probability and market price.

    Args:
        model_prob:       model's P(YES)
        yes_price:        market's P(YES)
        bankroll:         current bankroll in dollars
        edge_threshold:   minimum |model_prob - yes_price| to enter a trade
        kelly_multiplier: fraction of full Kelly to bet (0.5 = half-Kelly)
        max_fraction:     hard cap on bankroll fraction per bet (risk management)

    Returns:
        dict with action, edge, full_kelly, kelly_fraction, capped_fraction, bet_amount
    """
    edge = model_prob - yes_price

    if abs(edge) < edge_threshold:
        return {
            "action": "PASS", "edge": edge,
            "full_kelly": 0.0, "kelly_fraction": 0.0,
            "capped_fraction": 0.0, "bet_amount": 0.0,
        }

    if edge > 0:
        action     = "YES"
        full_kelly = kelly_yes(model_prob, yes_price)
    else:
        action     = "NO"
        full_kelly = kelly_no(model_prob, yes_price)

    # Apply multiplier then hard cap
    kelly_fraction  = full_kelly * kelly_multiplier
    capped_fraction = min(max(kelly_fraction, 0.0), max_fraction)
    bet_amount      = capped_fraction * bankroll

    return {
        "action":          action,
        "edge":            edge,
        "full_kelly":      full_kelly,
        "kelly_fraction":  kelly_fraction,
        "capped_fraction": capped_fraction,
        "bet_amount":      bet_amount,
    }


def compute_pnl(action: str, bet_amount: float, yes_price: float, label: int) -> float:
    """
    Resolve P&L for a completed trade.

    You stake bet_amount dollars:
      YES bet: buy YES tokens at yes_price each.
               Win (label=1): each token pays $1, net = bet_amount * (1/yes_price - 1)
               Lose (label=0): tokens worthless, net = -bet_amount
      NO bet:  buy NO tokens at (1 - yes_price) each.
               Win (label=0): each token pays $1, net = bet_amount * (1/(1-yes_price) - 1)
               Lose (label=1): tokens worthless, net = -bet_amount

    Returns:
        P&L in dollars (positive = profit, negative = loss)
    """
    if bet_amount <= 0:
        return 0.0

    if action == "YES":
        if label == 1:
            return bet_amount * (1.0 / yes_price - 1.0)   # payout minus stake
        else:
            return -bet_amount
    elif action == "NO":
        no_price = 1.0 - yes_price
        if label == 0:
            return bet_amount * (1.0 / no_price - 1.0)
        else:
            return -bet_amount
    return 0.0
