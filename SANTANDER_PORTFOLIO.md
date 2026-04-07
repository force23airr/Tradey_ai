# Santander Quant/Risk Portfolio — What to Build & Apply

## Why this project is relevant
Prediction market modeling (YES/NO outcome ~ probability) is structurally identical to
credit risk modeling (default/no-default ~ PD). The math, tools, and workflow are the same.
Santander quant teams do this on loan books. You're doing it on macro-driven market outcomes.

---

## Core skills to demonstrate

### 1. Logistic Regression (Credit Risk / PD Modeling)
- **What banks do:** Build PD (Probability of Default) models using logistic regression on
  borrower financials + macro variables. Required by Basel III.
- **What you're building:** Logistic regression on macro variables (VIX, yields, rates)
  to predict binary market outcomes.
- **How to frame it:** "This is structurally a PD model — binary label, macro features,
  probability output, calibrated against realized outcomes."

### 2. Macro Stress Testing (CCAR / DFAST scenarios)
- **What banks do:** Regulators hand them adverse macro scenarios. They run their models
  under those scenarios to estimate portfolio losses.
- **What you're building:** Stress test your model — plug in a +200bps rate spike,
  a VIX surge to 40, or a specific event (Bank of Japan YCC shift, Fed emergency cut)
  and see how predicted probabilities shift.
- **Specific scenarios to implement:**
  - Fed funds rate +200bps (2022 hiking cycle)
  - VIX spike to 40+ (COVID crash, Aug 2024)
  - Bank of Japan YCC policy shift (Oct 2023, Jan 2024)
  - USD/JPY reaction to BOJ decisions

### 3. FX Sensitivity Analysis
- **What banks do:** Measure how currency pairs react to central bank rate decisions.
- **What you can build:** Add USD/JPY, EUR/USD, GBP/USD to your macro dataset.
  Run regression: how much does USD/JPY move per 25bps Fed hike?
  This is literally what Santander's FX desk models.
- **Tickers to add:** USDJPY=X, EURUSD=X, GBPUSD=X (all free via yfinance)

### 4. Model Validation Metrics
Standard in every bank's model risk management (MRM) team:
- **AUC-ROC** — discrimination power of your model
- **KS Statistic** — separation between YES and NO distributions
- **Gini Coefficient** — 2*AUC - 1, standard in credit risk
- **Brier Score** — calibration quality (are your 70% predictions right 70% of the time?)
- **Hosmer-Lemeshow Test** — formal calibration hypothesis test

### 5. Feature Engineering with SQL
- **What banks do:** Data engineers build feature pipelines feeding into risk models.
  Window functions, rolling averages, macro joins.
- **What you're building:** Exactly this — rolling vol, momentum, days-to-close,
  macro variable joins at snapshot dates.

### 6. Time Series Forecasting
- **What banks do:** ARIMA/VAR models to forecast rates, FX, macro variables.
- **What you can build:** ARIMA on Fed funds rate or CPI to forecast next month's value.
  Use that forecast as a feature in your prediction model.

---

## Specific project deliverables to show a recruiter

| Deliverable | Why it impresses |
|---|---|
| Logistic regression model + calibration curve | Standard PD model workflow |
| Stress test notebook (rate spike / BOJ event) | Shows CCAR/stress testing awareness |
| FX reaction analysis (USD/JPY vs BOJ decisions) | Directly relevant to FX trading desk |
| AUC, KS, Gini, Brier score report | Model validation vocabulary |
| SQL feature engineering pipeline | Data engineering for risk models |
| ARIMA forecast on a macro variable | Time series / econometrics |

---

## Talking points for interviews

- "I built a binary probability model on macro variables — structurally equivalent to a PD model."
- "I implemented stress testing by shocking input variables and measuring prediction drift."
- "I validated my model using AUC-ROC, KS statistic, and Brier score — the same metrics
   used in SR 11-7 model validation."
- "I analyzed FX sensitivity to central bank decisions using OLS regression on rate differentials."

> SR 11-7 is the Fed's guidance on model risk management — every bank quant knows this document.
> Mentioning it signals you understand the regulatory context.

---

## What to add to this repo over time
- [ ] `models/regression/logistic_model.py` — logistic regression + calibration
- [ ] `models/time_series/arima_macro.py` — ARIMA on Fed funds / CPI
- [ ] `backtest/metrics.py` — AUC, KS, Gini, Brier score
- [ ] `notebooks/research/stress_test.ipynb` — macro shock scenarios
- [ ] `notebooks/research/fx_sensitivity.ipynb` — USD/JPY vs BOJ rate decisions
- [ ] `data/macro/fetcher.py` — add FX pairs (USDJPY, EURUSD, GBPUSD)
