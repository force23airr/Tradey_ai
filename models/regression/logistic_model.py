"""
Logistic Regression Model — Polymarket outcome prediction.

Predicts P(YES) for a binary prediction market given:
  - Market price features (implied probability, volatility, momentum)
  - Macro environment (VIX, yields, yield curve slope, S&P return)

This is structurally identical to a bank's PD (Probability of Default) model:
  - Binary label       → default (1) / no default (0)
  - Macro features     → CCAR stress variables
  - Probability output → used for capital allocation / bet sizing

Validation metrics follow SR 11-7 model risk standards:
  AUC-ROC, KS Statistic, Gini Coefficient, Brier Score
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    classification_report,
    RocCurveDisplay,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from db.connection import get_conn


# ── Feature columns used in the model ────────────────────────────────────────
# Grouped so you can easily add/remove feature groups and see the effect.

MARKET_FEATURES = [
    "yes_price",          # implied probability — the market's best guess
    "rolling_avg_7d",     # smoothed price (reduces noise)
    "rolling_vol_7d",     # price volatility — uncertain markets have high vol
    "momentum_7d",        # price trend — is market moving toward YES?
    "momentum_3d",        # short-term momentum
    "conviction",         # |price - 0.5| — how confident is the market?
    "days_to_close",      # time remaining (urgency)
]

MACRO_FEATURES = [
    "vix",                # fear index — high VIX = risk-off environment
    "treasury_10y",       # long-term rate level
    "treasury_5y",        # medium-term rate level
    "tbill_13w",          # short-term rate (Fed policy proxy)
    "yield_curve_slope",  # 10y - 3m: negative = inverted = recession signal
    "sp500_return_7d",    # recent equity performance (risk appetite)
    "gold_return_7d",     # safe haven demand
]

ALL_FEATURES = MARKET_FEATURES + MACRO_FEATURES


def load_features() -> pd.DataFrame:
    """Load the feature dataset by running the feature engineering SQL."""
    import importlib.util
    from pathlib import Path as _Path

    # Import build_features from data/features.py without it being a package
    spec = importlib.util.spec_from_file_location(
        "features",
        _Path(__file__).resolve().parent.parent.parent / "data" / "features.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_features(write_to_db=False)


def prepare_data(df: pd.DataFrame, feature_cols: list[str] = None):
    """
    Clean and split features/labels.
    Drops rows with any NaN in the selected feature columns.
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    available = [c for c in feature_cols if c in df.columns]
    df_clean = df[available + ["label", "market_id", "snapshot_date"]].dropna()

    X = df_clean[available].values
    y = df_clean["label"].values.astype(int)

    return X, y, df_clean, available


def train(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    """
    Fit a logistic regression with StandardScaler.

    Scaling is required because features are on very different scales
    (VIX ~15, treasury yields ~4, momentum ~0.01).

    C is the inverse regularization strength — lower C = more regularization.
    Ridge (L2) penalty by default to handle multicollinearity between macro vars.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=C,
        l1_ratio=0,   # L2 regularization (handles multicollinearity between macro vars)
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y)
    return model, scaler


def evaluate(model, scaler, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """
    Compute all validation metrics used in bank model risk management.

    Metrics:
      AUC-ROC      — discrimination (can the model rank YES above NO?)
      KS Statistic — max separation between YES and NO score distributions
      Gini         — 2*AUC - 1 (standard credit risk metric)
      Brier Score  — calibration quality (lower is better, 0 = perfect)
    """
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc   = roc_auc_score(y, probs)
    gini  = 2 * auc - 1
    brier = brier_score_loss(y, probs)

    # KS Statistic: max difference between YES CDF and NO CDF
    yes_scores = probs[y == 1]
    no_scores  = probs[y == 0]
    ks_stat, ks_pval = stats.ks_2samp(yes_scores, no_scores)

    print("\n" + "=" * 55)
    print("MODEL VALIDATION REPORT")
    print("=" * 55)
    print(f"  Samples:       {len(y)}  (YES={y.sum()}, NO={(y==0).sum()})")
    print(f"\n  Discrimination:")
    print(f"    AUC-ROC:     {auc:.4f}   (random=0.50, perfect=1.00)")
    print(f"    KS Statistic:{ks_stat:.4f}  (p={ks_pval:.4f})")
    print(f"    Gini Coeff:  {gini:.4f}   (standard credit risk metric)")
    print(f"\n  Calibration:")
    print(f"    Brier Score: {brier:.4f}   (perfect=0.00, random=0.25)")
    print(f"\n  Classification (threshold=0.50):")
    print(classification_report(y, preds, target_names=["NO", "YES"]))

    # Feature importance (standardized coefficients)
    coef = pd.Series(model.coef_[0], index=feature_names)
    print("  Feature coefficients (standardized):")
    print(coef.sort_values(ascending=False).round(4).to_string())
    print("=" * 55)

    return {
        "auc": auc, "gini": gini, "brier": brier,
        "ks_stat": ks_stat, "ks_pval": ks_pval,
        "probs": probs, "preds": preds,
        "coef": coef,
    }


def cross_validate(X: np.ndarray, y: np.ndarray, C: float = 1.0, n_splits: int = 5):
    """
    Stratified k-fold cross-validation.
    Stratified = each fold preserves the YES/NO ratio (important for imbalanced data).
    """
    scaler = StandardScaler()
    model  = LogisticRegression(C=C, l1_ratio=0, solver="lbfgs",
                                max_iter=1000, random_state=42)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("scaler", scaler), ("model", model)])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores   = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    brier_scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_brier_score")

    print(f"\n  {n_splits}-Fold Cross-Validation:")
    print(f"    AUC:   {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    print(f"    Brier: {(-brier_scores).mean():.4f} ± {(-brier_scores).std():.4f}")

    return auc_scores, -brier_scores


def plot_diagnostics(results: dict, y: np.ndarray, feature_names: list[str],
                     save_path: str = None):
    """
    4-panel diagnostic plot:
      1. ROC curve
      2. Calibration curve (reliability diagram)
      3. Score distribution by outcome
      4. Feature coefficients
    """
    probs = results["probs"]
    coef  = results["coef"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Logistic Regression — Model Diagnostics", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, probs)
    ax1.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"AUC = {results['auc']:.3f}")
    ax1.plot([0,1],[0,1], "k--", lw=1)
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC Curve")
    ax1.legend()

    # 2. Calibration Curve (Reliability Diagram)
    ax2 = fig.add_subplot(gs[0, 1])
    fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=8)
    ax2.plot(mean_pred, fraction_pos, "s-", color="darkorange", lw=2,
             label="Model")
    ax2.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
    ax2.set(xlabel="Mean Predicted Probability", ylabel="Fraction of YES outcomes",
            title=f"Calibration Curve  (Brier={results['brier']:.3f})")
    ax2.legend()

    # 3. Score Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(probs[y==0], bins=15, alpha=0.6, color="tomato",   label="NO  (0)")
    ax3.hist(probs[y==1], bins=15, alpha=0.6, color="seagreen", label="YES (1)")
    ax3.axvline(0.5, color="black", lw=1, linestyle="--")
    ax3.set(xlabel="Predicted P(YES)", ylabel="Count",
            title=f"Score Distribution  (KS={results['ks_stat']:.3f})")
    ax3.legend()

    # 4. Feature Coefficients
    ax4 = fig.add_subplot(gs[1, 1])
    coef_sorted = coef.sort_values()
    colors = ["seagreen" if v > 0 else "tomato" for v in coef_sorted]
    ax4.barh(coef_sorted.index, coef_sorted.values, color=colors)
    ax4.axvline(0, color="black", lw=0.8)
    ax4.set(xlabel="Coefficient (standardized)", title="Feature Importance")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()


def run(feature_group: str = "all", C: float = 1.0, plot: bool = True):
    """
    Full model training and evaluation pipeline.

    Args:
        feature_group: 'all' | 'market' | 'macro'
        C:             Regularization strength (inverse). Try 0.1, 1.0, 10.0.
        plot:          Show diagnostic plots.
    """
    print("Loading features...")
    df = load_features()

    if df.empty:
        print("No data in features table. Run: python data/features.py first.")
        return

    # Select feature group
    if feature_group == "market":
        cols = MARKET_FEATURES
    elif feature_group == "macro":
        cols = MACRO_FEATURES
    else:
        cols = ALL_FEATURES

    X, y, df_clean, used_features = prepare_data(df, cols)
    print(f"\nFeature group: '{feature_group}'  ({len(used_features)} features)")
    print(f"Training rows: {len(y)}  (YES={y.sum()}, NO={(y==0).sum()})")

    # Train
    model, scaler = train(X, y, C=C)

    # Evaluate (in-sample — small dataset, use CV for generalization estimate)
    results = evaluate(model, scaler, X, y, used_features)

    # Cross-validation
    cross_validate(X, y, C=C)

    # Plots
    if plot:
        save_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "notebooks" / "research" / "model_diagnostics.png"
        )
        plot_diagnostics(results, y, used_features, save_path=save_path)

    return model, scaler, results


if __name__ == "__main__":
    run(feature_group="all", C=1.0, plot=True)
