import pandas as pd
import numpy as np
import xgboost as xgb
import json
import pickle
from pathlib import Path
from models.train import (
    walk_forward_train,
    save_predictions,
    FEATURE_COLS,
)

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

def run_xgboost(
    df: pd.DataFrame,
    ticker: str = "SPY",
    n_splits: int = 5,
) -> dict:
    print(f"\nTraining XGBoost on {ticker}...")
    print(f"Params: {XGB_PARAMS}")

    result = walk_forward_train(
        model_cls  = xgb.XGBRegressor,
        df         = df,
        ticker     = ticker,
        n_splits   = n_splits,
        model_name = "xgboost",
        **XGB_PARAMS,
    )

    metrics = result.summary()
    print(f"\nXGBoost results — {ticker}:")
    print(json.dumps(metrics, indent=2))

    # Save predictions
    save_predictions(result, output_dir="artifacts")

    # Save a final model trained on all data for the dashboard
    available = [c for c in FEATURE_COLS if c in df.columns]
    sub = (
        df[df["ticker"] == ticker]
        .sort_values("date")
        [available + ["fwd_5d_return"]]
        .dropna()
    )
    X = sub[available].values
    y = sub["fwd_5d_return"].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_scaled, y)

    Path("artifacts").mkdir(exist_ok=True)
    with open(f"artifacts/xgb_{ticker}_model.pkl", "wb") as f:
        pickle.dump({"model": final_model, "scaler": scaler, "features": available}, f)
    print(f"Final model saved to artifacts/xgb_{ticker}_model.pkl")

    return metrics

def run_all_tickers(
    df: pd.DataFrame,
    tickers: list[str] = None,
) -> pd.DataFrame:
    if tickers is None:
        tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "XLF", "XLK"]

    all_metrics = []
    for ticker in tickers:
        if ticker not in df["ticker"].values:
            print(f"Skipping {ticker} — not in dataset")
            continue
        metrics = run_xgboost(df, ticker=ticker)
        all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)
    print("\nXGBoost summary across tickers:")
    print(summary.to_string(index=False))

    summary.to_parquet("artifacts/xgboost_summary.parquet", index=False)
    return summary

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/full_features_v2.parquet")
    run_all_tickers(df)