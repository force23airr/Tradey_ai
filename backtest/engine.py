"""
Backtesting engine for Polymarket prediction market strategies.

Pipeline:
  1. Load feature dataset (build_features)
  2. Time-split: train on pre-2024 data, test on 2024+
  3. Train logistic regression model on train set
  4. For each test market: pick snapshot ~30 days before close
  5. Model predicts P(YES) → compute edge vs market price
  6. Size bets with half-Kelly, cap at 10% of bankroll
  7. Simulate P&L in chronological order
  8. Report performance metrics

This is structurally identical to a bank's backtesting of a PD model —
train/test split prevents lookahead, and all metrics follow SR 11-7 standards.
"""

import sys
import logging
import importlib.util
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import metrics as bm
from strategies.portfolio import size_bet, compute_pnl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    trades:        pd.DataFrame
    equity_curve:  pd.Series
    train_metrics: dict
    test_metrics:  dict
    summary:       dict
    model:         object
    scaler:        object
    feature_names: list
    config:        dict


def _load_model_module():
    """Import logistic_model without requiring it to be a package."""
    spec = importlib.util.spec_from_file_location(
        "logistic_model",
        Path(__file__).resolve().parent.parent / "models" / "regression" / "logistic_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_features_module():
    """Import data/features.py without requiring it to be a package."""
    spec = importlib.util.spec_from_file_location(
        "features",
        Path(__file__).resolve().parent.parent / "data" / "features.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BacktestEngine:
    def __init__(
        self,
        initial_bankroll:  float = 1000.0,
        edge_threshold:    float = 0.05,
        kelly_multiplier:  float = 0.5,
        max_bet_fraction:  float = 0.10,
        days_before_close: int   = 30,
        train_end_year:    int   = 2023,
        test_start_year:   int   = 2024,
    ):
        self.initial_bankroll  = initial_bankroll
        self.edge_threshold    = edge_threshold
        self.kelly_multiplier  = kelly_multiplier
        self.max_bet_fraction  = max_bet_fraction
        self.days_before_close = days_before_close
        self.train_end_year    = train_end_year
        self.test_start_year   = test_start_year

        self.config = {
            "initial_bankroll":  initial_bankroll,
            "edge_threshold":    edge_threshold,
            "kelly_multiplier":  kelly_multiplier,
            "max_bet_fraction":  max_bet_fraction,
            "days_before_close": days_before_close,
            "train_end_year":    train_end_year,
            "test_start_year":   test_start_year,
        }

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        log.info("Loading feature dataset...")
        feat_mod = _load_features_module()
        df = feat_mod.build_features(write_to_db=False)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        log.info(f"  {len(df)} rows, {df['market_id'].nunique()} markets")
        return df

    def _time_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df["snapshot_date"].dt.year <= self.train_end_year].copy()
        test  = df[df["snapshot_date"].dt.year >= self.test_start_year].copy()
        log.info(f"  Train: {len(train)} rows, {train['market_id'].nunique()} markets "
                 f"(≤ {self.train_end_year})")
        log.info(f"  Test:  {len(test)} rows, {test['market_id'].nunique()} markets "
                 f"(≥ {self.test_start_year})")
        return train, test

    # ── Model training ────────────────────────────────────────────────────────

    def _train_model(self, train_df: pd.DataFrame, lm):
        """Train logistic model on train set. Returns model, scaler, feature_names, metrics."""
        X, y, df_clean, feature_names = lm.prepare_data(train_df)

        if len(y) == 0:
            raise ValueError("No training samples after dropping NaN rows. "
                             "Check that macro data covers the training period.")

        log.info(f"  Training on {len(y)} samples, {len(feature_names)} features...")
        model, scaler = lm.train(X, y)
        train_eval    = lm.evaluate(model, scaler, X, y, feature_names)
        return model, scaler, feature_names, train_eval

    def _evaluate_test(self, test_df: pd.DataFrame, model, scaler, feature_names: list, lm) -> dict:
        """Evaluate model on test set (no training — pure out-of-sample)."""
        available = [c for c in feature_names if c in test_df.columns]
        df_clean  = test_df[available + ["label", "market_id", "snapshot_date"]].dropna()
        X = df_clean[available].values
        y = df_clean["label"].values.astype(int)
        if len(y) == 0:
            log.warning("No test samples after dropping NaN. Check macro data coverage for test period.")
            return {"auc": np.nan, "gini": np.nan, "brier": np.nan, "ks_stat": np.nan}
        log.info(f"  Evaluating on {len(y)} test samples...")
        return lm.evaluate(model, scaler, X, y, feature_names)

    # ── Signal generation ─────────────────────────────────────────────────────

    def _select_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each market, pick the single snapshot closest to days_before_close.
        This is the entry point — the observation date for the trade.
        """
        target = self.days_before_close
        idx = (
            df.groupby("market_id")["days_to_close"]
            .apply(lambda x: (x - target).abs().idxmin())
        )
        return df.loc[idx].copy()

    def _generate_signals(
        self, snapshots: pd.DataFrame, model, scaler, feature_names: list, lm
    ) -> pd.DataFrame:
        """
        Apply model to one-snapshot-per-market df. Compute edge. Filter by threshold.
        """
        available = [c for c in feature_names if c in snapshots.columns]
        df_clean  = snapshots[available + ["market_id", "snapshot_date", "yes_price",
                                           "label", "days_to_close", "question"]].dropna(subset=available)

        X_scaled    = scaler.transform(df_clean[available].values)
        model_probs = model.predict_proba(X_scaled)[:, 1]

        df_clean = df_clean.copy()
        df_clean["model_prob"] = model_probs
        df_clean["edge"]       = model_probs - df_clean["yes_price"]

        # Only enter trades where we have a real edge
        signals = df_clean[df_clean["edge"].abs() >= self.edge_threshold].copy()
        signals["action"] = signals["edge"].apply(lambda e: "YES" if e > 0 else "NO")

        log.info(f"  {len(df_clean)} test markets → {len(signals)} signals "
                 f"(edge ≥ {self.edge_threshold:.0%})")
        return signals

    # ── Trade simulation ──────────────────────────────────────────────────────

    def _simulate_trades(
        self, signals: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Walk through signals in chronological order.
        Apply Kelly sizing. Record P&L. Track running bankroll.
        """
        signals = signals.sort_values("snapshot_date").reset_index(drop=True)
        bankroll = self.initial_bankroll
        trade_records = []

        for _, row in signals.iterrows():
            sizing = size_bet(
                model_prob      = row["model_prob"],
                yes_price       = row["yes_price"],
                bankroll        = bankroll,
                edge_threshold  = self.edge_threshold,
                kelly_multiplier= self.kelly_multiplier,
                max_fraction    = self.max_bet_fraction,
            )
            if sizing["action"] == "PASS" or sizing["bet_amount"] <= 0:
                continue

            pnl = compute_pnl(
                action     = sizing["action"],
                bet_amount = sizing["bet_amount"],
                yes_price  = row["yes_price"],
                label      = int(row["label"]),
            )

            bankroll_before = bankroll
            bankroll       += pnl

            trade_records.append({
                "market_id":      row["market_id"],
                "snapshot_date":  row["snapshot_date"],
                "question":       row.get("question", ""),
                "yes_price":      row["yes_price"],
                "model_prob":     row["model_prob"],
                "edge":           row["edge"],
                "action":         sizing["action"],
                "full_kelly":     sizing["full_kelly"],
                "kelly_fraction": sizing["kelly_fraction"],
                "capped_fraction":sizing["capped_fraction"],
                "bet_amount":     sizing["bet_amount"],
                "bankroll_before":bankroll_before,
                "pnl":            pnl,
                "bankroll_after": bankroll,
                "label":          int(row["label"]),
                "days_to_close":  row["days_to_close"],
            })

        trades       = pd.DataFrame(trade_records)
        equity_curve = pd.Series(
            [self.initial_bankroll] + list(trades["bankroll_after"]),
            name="bankroll"
        )
        return trades, equity_curve

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(self) -> BacktestResults:
        """Full backtest pipeline. Returns BacktestResults."""
        lm = _load_model_module()

        df                 = self._load_data()
        train_df, test_df  = self._time_split(df)

        log.info("Training model on train set...")
        model, scaler, feature_names, train_metrics = self._train_model(train_df, lm)

        log.info("Evaluating model on test set (out-of-sample)...")
        test_metrics = self._evaluate_test(test_df, model, scaler, feature_names, lm)

        log.info("Generating trading signals on test set...")
        test_snapshots = self._select_snapshots(
            test_df[
                [c for c in feature_names if c in test_df.columns]
                + ["market_id", "snapshot_date", "yes_price", "label",
                   "days_to_close", "question"]
            ].dropna(subset=[c for c in feature_names if c in test_df.columns])
        )
        signals = self._generate_signals(test_snapshots, model, scaler, feature_names, lm)

        log.info("Simulating trades...")
        trades, equity_curve = self._simulate_trades(signals)

        summary = bm.summarize(trades, equity_curve, self.initial_bankroll)
        self._print_summary(summary)

        return BacktestResults(
            trades        = trades,
            equity_curve  = equity_curve,
            train_metrics = train_metrics,
            test_metrics  = test_metrics,
            summary       = summary,
            model         = model,
            scaler        = scaler,
            feature_names = feature_names,
            config        = self.config,
        )

    def _print_summary(self, summary: dict):
        print("\n" + "=" * 55)
        print("BACKTEST SUMMARY")
        print("=" * 55)
        if "error" in summary:
            print(f"  {summary['error']}")
            return
        print(f"  Total trades:    {summary['total_trades']}")
        print(f"  YES bets:        {summary['yes_bets']}")
        print(f"  NO bets:         {summary['no_bets']}")
        print(f"  Win rate:        {summary['win_rate']:.1%}")
        print(f"  Total P&L:       ${summary['total_pnl']:.2f}")
        print(f"  ROI:             {summary['roi']:.1%}")
        print(f"  Final bankroll:  ${summary['final_bankroll']:.2f}")
        print(f"\n  Risk metrics:")
        print(f"    Sharpe ratio:  {summary['sharpe']:.3f}")
        print(f"    Sortino ratio: {summary['sortino']:.3f}")
        print(f"    Max drawdown:  {summary['max_drawdown']:.1%}")
        print(f"    Profit factor: {summary['profit_factor']:.2f}")
        print(f"\n  Calibration:")
        print(f"    Brier score:   {summary['brier_score']:.4f}")
        print(f"    Avg edge:      {summary['avg_edge']:.3f}")
        print(f"    Avg bet size:  {summary['avg_bet_pct']:.1%} of bankroll")
        print("=" * 55)


if __name__ == "__main__":
    engine = BacktestEngine(
        initial_bankroll  = 1000.0,
        edge_threshold    = 0.05,
        kelly_multiplier  = 0.5,
        max_bet_fraction  = 0.10,
        days_before_close = 30,
    )
    results = engine.run()
