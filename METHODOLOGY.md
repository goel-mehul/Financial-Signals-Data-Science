# FinSignal — Methodology & Findings

## Overview

FinSignal is an end-to-end financial data science pipeline that ingests
market and macroeconomic data, applies rigorous time series analysis,
engineers predictive features, and trains ML models to forecast 5-day
forward returns across S&P 500 constituents.

---

## Data

**Market data** — 5 years of daily OHLCV for 80 S&P 500 tickers and
sector ETFs via yfinance. Stored in Parquet format, loaded into DuckDB
for SQL querying.

**Macro data** — 6 FRED indicators from 2018 onwards:
- CPI (inflation)
- Fed Funds Rate (monetary policy)
- Unemployment Rate
- 10-year Treasury Yield (risk-free rate proxy)
- M2 Money Supply (liquidity)
- VIX (market fear index)

**Cleaning** — forward-filled gaps within tickers, dropped tickers with
fewer than 252 trading days of history, removed zero/null close prices.

---

## Time Series Analysis

### Stationarity testing
Batch ADF + KPSS tests across all 80 tickers on both raw close prices
and log returns. Raw prices universally failed both tests (non-stationary).
Log returns passed both tests (stationary). All downstream modeling uses
log returns, not raw prices.

### STL decomposition
Applied Seasonal-Trend decomposition using Loess (STL) to monthly macro
indicators with period=12. CPI decomposition clearly isolates the 2021-2023
inflation surge in the trend component, separate from monthly seasonality.

### GARCH(1,1) volatility modeling
Fit GARCH(1,1) on SPY, QQQ, XLF, XLK log returns.

Key finding — volatility persistence (α+β) of ~0.986 on SPY. This confirms
strong volatility clustering: shocks are extremely long-lived and markets
remain volatile for weeks after an initial shock. This is consistent with
decades of academic finance literature.

### Granger causality
Tested whether VIX, Fed Funds Rate, and 10yr Yield Granger-cause returns
for SPY, QQQ, and XLF at lags 1-5.

Key finding — VIX significantly Granger-causes SPY returns at lags 1-3
(p < 0.05). The fear index contains real predictive information about
near-term market direction beyond SPY's own history.

### VAR model
Fit Vector Autoregression on sector ETF returns (XLF, XLK, XLE, XLV,
XLI, XLP). AIC selected lag order 0, indicating that at daily frequency
sector returns do not strongly predict each other — consistent with
efficient market hypothesis at short horizons.

### Prophet forecasting
Applied Facebook Prophet to monthly macro indicators (CPI, Unemployment,
Yield10yr, FedFunds) with 3-month forward forecasts. Changepoint detection
identified 11-16 structural breaks per series, capturing Fed policy shifts.

---

## Feature Engineering

24 features engineered per ticker:

**Momentum** — 5, 10, 21, 63-day price momentum
**Volatility** — 5, 21, 63-day rolling standard deviation of log returns
**Trend** — RSI(14), MACD, MACD signal, MACD histogram
**Mean reversion** — Bollinger Band %B, 63-day rolling z-score
**Range** — ATR(14)
**Volume** — volume ratio vs 20-day average
**Macro (lagged 1 period)** — CPI, FedFunds, Unemployment, Yield10yr,
M2, VIX — all lagged by 1 business day to prevent lookahead bias
**Derived macro** — CPI MoM change, 21-day yield change, VIX regime flag

**Target variable** — 5-day forward return, computed as
`close.pct_change(5).shift(-5)` to avoid leakage.

---

## Modeling

### Walk-forward validation
All models evaluated using TimeSeriesSplit (5 folds). Each fold's test
set is strictly after its training set. StandardScaler fit on train only,
applied to test. This is the only correct evaluation methodology for
financial time series.

### XGBoost baseline
Parameters: 300 estimators, max_depth=4, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8.

### LightGBM + Optuna
30-trial Bayesian hyperparameter search using Optuna, optimizing
Information Coefficient (IC) on 3-fold inner CV. Best params then
evaluated on full 5-fold walk-forward.

---

## Results

| Ticker | Model | Sharpe | IC | Hit Rate |
|--------|-------|--------|----|----------|
| SPY | XGBoost | 0.09 | 0.09 | 48.1% |
| SPY | LightGBM | 0.42 | 0.12 | 50.6% |
| QQQ | XGBoost | 0.03 | 0.09 | 48.9% |
| QQQ | LightGBM | 0.12 | 0.11 | 49.5% |
| NVDA | XGBoost | 0.13 | 0.07 | 51.7% |
| NVDA | LightGBM | 0.22 | 0.07 | 53.1% |
| XLF | XGBoost | 0.03 | 0.08 | 48.1% |
| XLF | LightGBM | 0.22 | 0.10 | 48.8% |
| XLK | XGBoost | 0.23 | 0.07 | 49.7% |
| XLK | LightGBM | 0.32 | 0.08 | 51.5% |

### Key findings

**LightGBM + Optuna outperformed XGBoost** on average Sharpe across all
tickers (0.21 vs 0.06). Bayesian HPO contributed meaningfully — tuned
learning rate, num_leaves, and regularization parameters drove the gap.

**Sector ETFs more predictable than individual stocks** — XLF, XLK
consistently showed positive IC and Sharpe. Individual stocks (AAPL, MSFT)
showed near-zero or negative signal, suggesting idiosyncratic news dominates
at the single-stock level.

**Top SHAP features** — M2 money supply (lagged) was the strongest feature
for SPY and XLF, followed by long-window momentum (mom_63d) and volatility
(vol_63d). This suggests the model captured macro liquidity regimes more
than short-term technical patterns.

**NVDA highest hit rate** (53.1%) — momentum features drove NVDA predictions,
consistent with the stock's strong trend behavior during the AI boom period.

---

## Limitations & v2 Roadmap

**Data window** — 5 years / ~1,200 rows per ticker is insufficient for
complex models. v2 will expand to 10 years.

**Feature leakage risk** — M2 as a level feature is a slow-moving time
proxy. v2 will replace with M2 rate-of-change.

**Single regime model** — one model trained across bull and bear markets.
v2 will add VIX regime-conditioned models.

**No transaction cost modeling** — raw Sharpe overstates real-world
performance. v2 will add realistic cost assumptions.

**Target variable** — raw 5-day return is noisy. v2 will explore
risk-adjusted return as the target.