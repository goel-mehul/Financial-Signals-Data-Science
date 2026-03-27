import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def test_stationarity(series: pd.Series, name: str = "") -> dict:
    """
    Run ADF and KPSS on a series.
    ADF  null = unit root (non-stationary) → reject at p < 0.05 → stationary
    KPSS null = stationary               → reject at p < 0.05 → non-stationary
    Both agree stationary  → confident it's stationary
    Both agree non-stat    → confident it's non-stationary
    They disagree          → likely difference-stationary (needs differencing)
    """
    series = series.dropna()

    adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag="AIC")
    kpss_stat, kpss_p, _, _     = kpss(series, regression="c", nlags="auto")

    adf_stationary  = adf_p < 0.05
    kpss_stationary = kpss_p >= 0.05

    if adf_stationary and kpss_stationary:
        verdict = "stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non-stationary"
    else:
        verdict = "difference-stationary"

    return {
        "name":            name,
        "adf_stat":        round(adf_stat, 4),
        "adf_p":           round(adf_p, 4),
        "adf_stationary":  adf_stationary,
        "kpss_stat":       round(kpss_stat, 4),
        "kpss_p":          round(kpss_p, 4),
        "kpss_stationary": kpss_stationary,
        "verdict":         verdict,
    }

def batch_stationarity(df: pd.DataFrame, col: str = "log_return") -> pd.DataFrame:
    """Test stationarity for every ticker in the dataset."""
    results = []
    tickers = df["ticker"].unique()
    print(f"Testing stationarity for {len(tickers)} tickers on column '{col}'...")

    for i, ticker in enumerate(tickers):
        series = df[df["ticker"] == ticker][col].dropna()
        if len(series) < 50:
            continue
        result = test_stationarity(series, name=ticker)
        results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(tickers)}...")

    out = pd.DataFrame(results)

    print("\nStationarity summary:")
    print(out["verdict"].value_counts())

    out.to_parquet("data/processed/stationarity_results.parquet", index=False)
    print("\nSaved to data/processed/stationarity_results.parquet")
    return out

def compare_price_vs_returns(df: pd.DataFrame, ticker: str = "SPY") -> None:
    """
    Print side by side — raw prices are non-stationary,
    log returns are stationary. This is the key insight.
    """
    sub = df[df["ticker"] == ticker].sort_values("date")

    print(f"\n--- {ticker} raw close price ---")
    price_result = test_stationarity(sub["close"], name=f"{ticker}_price")
    print(f"ADF p-value:  {price_result['adf_p']}  ({'stationary' if price_result['adf_stationary'] else 'NON-STATIONARY'})")
    print(f"KPSS p-value: {price_result['kpss_p']}  ({'stationary' if price_result['kpss_stationary'] else 'NON-STATIONARY'})")
    print(f"Verdict: {price_result['verdict']}")

    print(f"\n--- {ticker} log returns ---")
    return_result = test_stationarity(sub["log_return"].dropna(), name=f"{ticker}_returns")
    print(f"ADF p-value:  {return_result['adf_p']}  ({'stationary' if return_result['adf_stationary'] else 'NON-STATIONARY'})")
    print(f"KPSS p-value: {return_result['kpss_p']}  ({'stationary' if return_result['kpss_stationary'] else 'NON-STATIONARY'})")
    print(f"Verdict: {return_result['verdict']}")

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/ohlcv_returns.parquet")

    # First show the key insight — prices vs returns
    compare_price_vs_returns(df, "SPY")

    # Then batch test all tickers on log returns
    results = batch_stationarity(df, col="log_return")
    print("\nFirst 10 results:")
    print(results.head(10).to_string(index=False))