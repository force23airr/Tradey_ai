"""
Backtest performance metrics.

Pure functions — no state. Each can be called independently from a notebook.
All ratio metrics are scaled to trade count (not calendar days), which is correct
when trades are irregular in time.
"""

import numpy as np
import pandas as pd

RISK_FREE_RATE_ANNUAL = 0.05   # ~2024 T-bill rate


def compute_returns(trades: pd.DataFrame) -> pd.Series:
    """Per-trade return as a fraction of amount bet."""
    return trades["pnl"] / trades["bet_amount"]


def sharpe_ratio(returns: pd.Series) -> float:
    """
    Annualized Sharpe ratio scaled to trade count.
    Risk-free rate prorated to per-trade frequency.
    """
    if len(returns) < 2:
        return np.nan
    n = len(returns)
    rf_per_trade = RISK_FREE_RATE_ANNUAL / n
    excess = returns - rf_per_trade
    if returns.std() == 0:
        return np.nan
    return float((excess.mean() / returns.std()) * np.sqrt(n))


def sortino_ratio(returns: pd.Series) -> float:
    """
    Sortino ratio — like Sharpe but only penalizes downside volatility.
    Better metric for strategies with asymmetric payoffs (like binary bets).
    """
    if len(returns) < 2:
        return np.nan
    n = len(returns)
    rf_per_trade = RISK_FREE_RATE_ANNUAL / n
    excess = returns - rf_per_trade
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = downside.std()
    if downside_std == 0:
        return np.nan
    return float((excess.mean() / downside_std) * np.sqrt(n))


def max_drawdown(equity_curve: pd.Series) -> tuple[float, int, int]:
    """
    Maximum peak-to-trough drawdown.

    Returns:
        (max_drawdown_pct, peak_index, trough_index)
        max_drawdown_pct is negative (e.g. -0.12 = -12% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve - rolling_peak) / rolling_peak
    min_idx = int(drawdown.idxmin())
    # Find the peak before the trough
    peak_idx = int(equity_curve.iloc[:min_idx + 1].idxmax())
    return float(drawdown.min()), peak_idx, min_idx


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of trades that ended in profit."""
    if len(trades) == 0:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def roi(trades: pd.DataFrame, initial_bankroll: float) -> float:
    """Total return on initial bankroll. 0.18 = 18%."""
    return float(trades["pnl"].sum() / initial_bankroll)


def brier_score(trades: pd.DataFrame) -> float:
    """
    Mean squared error between model probability and actual binary outcome.
    Lower is better. Random = 0.25. Perfect = 0.00.
    Only uses YES-bet rows (model_prob vs label).
    """
    if len(trades) == 0:
        return np.nan
    return float(((trades["model_prob"] - trades["label"]) ** 2).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    """
    Gross profit / gross loss (absolute).
    > 1.0 = profitable. Standard hedge fund reporting metric.
    """
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss   = trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if gross_loss == 0:
        return np.inf
    return float(gross_profit / abs(gross_loss))


def summarize(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    initial_bankroll: float,
) -> dict:
    """Compute all metrics and return as a single dict."""
    if len(trades) == 0:
        return {"total_trades": 0, "error": "no trades entered"}

    returns = compute_returns(trades)
    dd, dd_start, dd_end = max_drawdown(equity_curve)

    return {
        "total_trades":   len(trades),
        "yes_bets":       int((trades["action"] == "YES").sum()),
        "no_bets":        int((trades["action"] == "NO").sum()),
        "win_rate":       win_rate(trades),
        "roi":            roi(trades, initial_bankroll),
        "total_pnl":      float(trades["pnl"].sum()),
        "final_bankroll": float(equity_curve.iloc[-1]),
        "sharpe":         sharpe_ratio(returns),
        "sortino":        sortino_ratio(returns),
        "max_drawdown":   dd,
        "max_dd_start":   dd_start,
        "max_dd_end":     dd_end,
        "brier_score":    brier_score(trades),
        "profit_factor":  profit_factor(trades),
        "avg_edge":       float(trades["edge"].abs().mean()),
        "avg_bet_pct":    float((trades["bet_amount"] / trades["bankroll_before"]).mean()),
    }
