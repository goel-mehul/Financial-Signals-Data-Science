# FinSignal — time-series driven market intelligence pipeline

> End-to-end financial data science pipeline: ingestion → time series analysis → ML signals → interactive dashboard.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What it does

FinSignal ingests 5 years of daily market data for 80 S&P 500 tickers and 6 FRED macro
indicators, applies rigorous time series analysis, engineers 24+ features, and trains
XGBoost and LightGBM models to predict 5-day forward returns — all evaluated with
walk-forward validation to prevent lookahead bias.

---

## Architecture
```
yfinance (OHLCV) ──┐
FRED API (macro)  ──┤──▶ DuckDB ──▶ Cleaning ──▶ Feature Engineering
                   │                                      │
                   │                                      ▼
                   │                          Time Series Analysis
                   │                     (ADF/KPSS, STL, GARCH, VAR)
                   │                                      │
                   └──────────────────────────────────────┤
                                                          ▼
                                              XGBoost + LightGBM
                                           (walk-forward validation)
                                                          │
                                                          ▼
                                              Streamlit Dashboard
```

---

## Key results

| Ticker | Model | Sharpe | IC | Hit Rate |
|--------|-------|--------|----|----------|
| SPY | LightGBM | 0.42 | 0.12 | 50.6% |
| QQQ | LightGBM | 0.12 | 0.11 | 49.5% |
| NVDA | LightGBM | 0.22 | 0.08 | 53.1% |
| XLF | LightGBM | 0.22 | 0.10 | 48.8% |
| XLK | LightGBM | 0.32 | 0.08 | 51.5% |

> Evaluated via 5-fold walk-forward validation. No lookahead bias.
> LightGBM + Optuna outperformed XGBoost baseline on average Sharpe (0.21 vs 0.06).

---

## Time series findings

- **Stationarity** — Raw prices non-stationary (ADF p=0.98), log returns stationary
  (ADF p<0.001). Confirmed with both ADF and KPSS tests across all 80 tickers.
- **Volatility clustering** — GARCH(1,1) persistence (α+β) of 0.986 on SPY confirms
  volatility shocks are long-lasting — calm periods cluster, volatile periods cluster.
- **Granger causality** — VIX significantly Granger-causes SPY returns at lags 1-3
  (p<0.05), confirming fear index contains predictive information about market direction.
- **Cross-sector dynamics** — VAR model on sector ETFs shows XLF shocks propagate
  into XLK and XLV within 3-5 trading days.
- **Top predictive feature** — SHAP analysis identifies M2 money supply (lagged) as
  the strongest feature for SPY, suggesting the model captured Fed liquidity dynamics.

---

## Techniques used

| Category | Techniques |
|----------|-----------|
| Data engineering | DuckDB, Parquet, FRED API, yfinance, forward-fill, lag features |
| Time series | ADF/KPSS stationarity, STL decomposition, ARIMA walk-forward, GARCH(1,1), Granger causality, VAR + IRF, Prophet |
| Feature engineering | RSI, MACD, Bollinger Bands, ATR, rolling z-score, momentum (5/10/21/63d), volatility (5/21/63d), macro lags |
| ML modeling | XGBoost, LightGBM, Optuna Bayesian HPO, TimeSeriesSplit walk-forward |
| Explainability | SHAP TreeExplainer, dependence plots, feature importance |
| Anomaly detection | Isolation Forest, rolling z-score (3σ threshold) |
| Visualization | Plotly, Streamlit |
| Engineering | GitHub Actions CI, Makefile, pytest, pyproject.toml |

---

## Setup
```bash
git clone https://github.com/goel-mehul/Financial-Signals-Data-Science.git
cd Financial-Signals-Data-Science
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env    # add your FRED_API_KEY
```

Run the full pipeline:
```bash
make all                          # ingest + clean + features
python -m models.xgboost_model    # train XGBoost
python -m models.lgbm_model       # train LightGBM + Optuna
python -m models.explain          # SHAP analysis
python -m models.anomaly          # anomaly detection
python -m models.evaluate         # benchmarking report
streamlit run dashboard/app.py    # launch dashboard
```

---

## Project structure
```
finsignal/
├── ingestion/          # Data ingestion (yfinance, FRED, DuckDB)
├── analysis/           # EDA, Granger causality
├── time_series/        # ARIMA, GARCH, STL, VAR, Prophet
├── features/           # Technical + macro feature engineering
├── models/             # XGBoost, LightGBM, SHAP, anomaly, evaluation
├── dashboard/          # Streamlit app
├── reports/            # Auto-generated insight report
├── data/
│   ├── raw/            # Raw Parquet files (gitignored)
│   ├── processed/      # Feature matrices (gitignored)
│   └── quality_report/ # HTML reports and charts
├── artifacts/          # Trained models + predictions (gitignored)
└── tests/              # pytest suite
```

---

## CI/CD

GitHub Actions runs the ingestion pipeline every weekday at 9am ET,
refreshes the data quality report, and commits the updated report to the repo.

---

## License

MIT