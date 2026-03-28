# FinSignal — Methodology & Findings (v2)

## Overview

FinSignal is an end-to-end financial data science pipeline that ingests
10 years of market and macroeconomic data, applies rigorous time series
analysis, engineers regime-aware features, and trains regime-conditioned
ML models to forecast 5-day forward returns across 30 S&P 500 constituents
and sector ETFs.

---

## Data

**Market data** — 10 years of daily OHLCV (2016-2026) for 30 tickers
covering every major sector. Stored in Parquet, loaded into DuckDB.

**Macro data** — 6 FRED indicators:
- CPI (inflation)
- Fed Funds Rate (monetary policy)
- Unemployment Rate
- 10-year Treasury Yield
- M2 Money Supply
- VIX (market fear index)

**Key data decisions:**
- 10 years chosen over 5 to double training rows (~2,500 vs ~1,200 per ticker)
- 2016 start captures two full bear markets (2020, 2022), two recoveries,
  COVID, and a full rate hike cycle
- Macro features lagged by 1 business day — prevents lookahead bias

---

## Time Series Analysis

### Stationarity
ADF + KPSS batch tests across all tickers. Raw prices universally
non-stationary. Log returns stationary. All modeling uses log returns.

### STL Decomposition
Applied to monthly macro indicators (period=12). CPI decomposition
isolates the 2021-2023 inflation surge from seasonal patterns.

### GARCH(1,1)
Volatility persistence (α+β) ~0.986 on SPY — confirms strong volatility
clustering. Shocks are long-lived, consistent with academic literature.

### Granger Causality
VIX significantly Granger-causes SPY returns at lags 1-3 (p<0.05).
Yield curve and Fed Funds Rate also show predictive causality for
rate-sensitive sectors.

### VAR Model
Sector ETF returns show lag-0 optimal order — at daily frequency,
sectors don't strongly predict each other. Consistent with EMH at
short horizons.

### Prophet
3-month forecasts on CPI, Unemployment, Yield10yr, FedFunds.
Changepoint detection identifies 11-16 structural breaks per series.

---

## Feature Engineering — v2

### Technical features (15)
RSI(14), ATR(14), MACD, MACD signal, MACD histogram, Bollinger Band %B,
rolling z-score (63d), momentum (5/10/21/63d), volatility (5/21/63d),
volume ratio

### Macro rate-of-change features (7)
CPI MoM change, yield 21d change, CPI 1m momentum, M2 1m momentum,
M2 3m momentum, Fed Funds 3m change, yield 1m change

**Key v2 decision** — replaced M2 level with M2 rate of change.
M2 as a level feature is a slow-moving time proxy (model learns
"later date = higher M2 = bull market"). Rate of change captures
genuine liquidity impulses.

### Regime features (13) — new in v2
- VIX regime flags (4 levels: low/normal/elevated/crisis)
- VIX raw level
- Yield curve slope (10yr minus Fed Funds)
- Yield curve inverted flag
- 52-week drawdown
- Volatility regime (current vol vs own 252d history)
- Volatility expanding flag
- Rolling 63d beta vs SPY
- 21d relative strength vs SPY
- Macro stress index (composite: 0.5×VIX + 0.3×curve stress + 0.2×vol stress)

---

## Modeling — v2

### Walk-forward validation
TimeSeriesSplit (5 folds). Each fold's test set strictly after training.
StandardScaler fit on train only. No lookahead bias.

### XGBoost baseline
300 estimators, max_depth=4, learning_rate=0.05.

### LightGBM + Optuna
30-trial Bayesian HPO per ticker, optimizing IC on 3-fold inner CV.

### Regime-conditioned LightGBM — new in v2
At each walk-forward fold:
1. Split training data into calm (VIX < 20) and stressed (VIX ≥ 20) subsets
2. Train separate Optuna-tuned LightGBM for each regime
3. At prediction time, route each row to the appropriate model

This allows the model to learn that financials behave differently when
the yield curve is inverted vs steep, and that defensive sectors respond
differently to high-VIX vs low-VIX environments.

---

## Results — v2

### Best performers

| Ticker | Best Model | Sharpe | IC | Hit Rate | Sector |
|--------|-----------|--------|----|----------|--------|
| XLU | LightGBM | 0.94 | 0.14 | 56.9% | Utilities |
| XLV | LightGBM | 0.74 | 0.14 | 53.6% | Healthcare |
| JNJ | XGBoost | 0.70 | 0.15 | 54.2% | Healthcare |
| XLK | LightGBM | 0.67 | 0.07 | 52.6% | Technology |
| XLI | LightGBM | 0.57 | 0.05 | 51.6% | Industrials |
| QQQ | LightGBM | 0.55 | 0.07 | 52.7% | Tech index |
| AVGO | LightGBM | 0.48 | 0.04 | 51.8% | Semiconductors |
| MSFT | LightGBM | 0.47 | 0.07 | 50.5% | Technology |
| WMT | XGBoost | 0.43 | 0.11 | 53.2% | Consumer staples |
| XLP | LightGBM | 0.42 | 0.08 | 52.3% | Consumer staples |

### V1 vs V2 comparison

| Ticker | V1 Sharpe | V2 Sharpe | Change |
|--------|-----------|-----------|--------|
| QQQ | 0.12 | 0.55 | +0.43 |
| MSFT | -0.02 | 0.47 | +0.49 |
| XLK | 0.32 | 0.67 | +0.35 |
| XLI | — | 0.57 | new |
| XLU | — | 0.94 | new |

### Key findings

**Defensive sectors dominate** — XLU, XLV, XLP, XLB all show Sharpe > 0.4
and IC > 0.08. These sectors respond predictably to interest rate regimes
which our yield curve slope and macro stress features capture directly.

**Regime conditioning works** — SPY improved from 0.09 to 0.24 Sharpe.
QQQ improved from 0.12 to 0.55. The separate calm/stressed models learned
different relationships between macro features and returns across market regimes.

**Growth stocks remain unpredictable** — NVDA, META, AMD show near-zero
or negative Sharpe. These are driven by earnings surprises and AI narrative
shifts that macro and technical features cannot anticipate.

**Energy is structurally unforecastable** — XOM (-0.33), XLE (-0.10).
Oil prices respond to geopolitical events which are inherently unpredictable
from market data alone.

**Top SHAP features across winning tickers:**
- yield_curve_slope — universal macro signal, top 3 for every winning ticker
- vol_regime — relative volatility vs own history, better than raw VIX
- macro_stress — composite regime indicator
- drawdown_52w — distance from 52-week high
- M2_mom_1m — liquidity impulse, not level

### What the model learned

The yield curve slope is the most universally important feature. When the
curve steepens, risk assets and defensive sectors do well. When it inverts,
the model correctly becomes more cautious. This relationship held across
the 2019 inversion, 2020 COVID shock, 2022 rate hike cycle, and 2024
normalization.

The vol_regime feature (current volatility relative to its own 252-day
history) outperformed raw VIX as a feature — relative volatility is more
informative than absolute VIX level because it captures whether conditions
are getting better or worse, not just where they are.

---

## Limitations & v3 Roadmap

**Energy and financials** — XOM, XLE, XLF remain difficult. Would require
commodity price data (oil futures, natural gas) and credit spread data
(investment grade vs high yield spreads) to improve.

**Growth stocks** — NVDA, META, AMD need earnings surprise data, options
implied volatility, and sentiment data (news/social) to be predictable.

**Transaction costs** — all results are pre-cost. A realistic simulation
would reduce Sharpe by 30-50% depending on turnover.

**Regime transitions** — the model struggles at regime change points
(e.g. when yield curve moved from inverted to steep in 2024). A
hidden Markov model layer for regime detection could help.

**Single-period target** — 5-day return is noisy. v3 could explore
longer horizons (21-day) for defensive sectors where signal persists longer.