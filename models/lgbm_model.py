import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import pickle
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from models.train import (
    walk_forward_train,
    save_predictions,
    prepare_dataset,
    FEATURE_COLS,
    WalkForwardResult,
)
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optuna_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
) -> float:
    """
    Optuna objective function.
    For each trial, Optuna suggests a set of hyperparameters.
    We evaluate them using walk-forward CV and return the IC.
    Optuna maximizes this — so it searches for params that give highest IC.
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 63),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "random_state":     42,
        "verbosity":        -1,
        "n_jobs":           -1,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    ics  = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_test)

        from scipy.stats import spearmanr
        ic, _ = spearmanr(preds, y_test)
        ics.append(ic if not np.isnan(ic) else 0.0)

    return float(np.mean(ics))

def tune_lgbm(
    df: pd.DataFrame,
    ticker: str,
    n_trials: int = 30,
) -> dict:
    """
    Run Optuna hyperparameter search for LightGBM.
    Returns best params found across n_trials.
    """
    print(f"\nRunning Optuna HPO for {ticker} ({n_trials} trials)...")

    data      = prepare_dataset(df, ticker)
    available = [c for c in FEATURE_COLS if c in data.columns]
    X         = data[available].values
    y         = data["fwd_5d_return"].values

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X, y, n_splits=3),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_ic     = study.best_value
    print(f"Best IC from HPO: {best_ic:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    best_params["random_state"] = 42
    best_params["verbosity"]    = -1
    best_params["n_jobs"]       = -1

    return best_params

def run_lgbm(
    df: pd.DataFrame,
    ticker: str = "SPY",
    n_splits: int = 5,
    n_trials: int = 30,
) -> dict:
    # Step 1 — find best hyperparameters with Optuna
    best_params = tune_lgbm(df, ticker, n_trials=n_trials)

    # Step 2 — evaluate with full walk-forward using best params
    print(f"\nWalk-forward evaluation with best params...")
    result = walk_forward_train(
        model_cls  = lgb.LGBMRegressor,
        df         = df,
        ticker     = ticker,
        n_splits   = n_splits,
        model_name = "lgbm",
        **best_params,
    )

    metrics = result.summary()
    print(f"\nLightGBM results — {ticker}:")
    print(json.dumps(metrics, indent=2))

    save_predictions(result, output_dir="artifacts")

    # Step 3 — save final model trained on all data
    data      = prepare_dataset(df, ticker)
    available = [c for c in FEATURE_COLS if c in data.columns]
    X         = data[available].values
    y         = data["fwd_5d_return"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_scaled, y)

    Path("artifacts").mkdir(exist_ok=True)
    with open(f"artifacts/lgbm_{ticker}_model.pkl", "wb") as f:
        pickle.dump({
            "model":    final_model,
            "scaler":   scaler,
            "features": available,
            "params":   best_params,
        }, f)
    print(f"Final model saved to artifacts/lgbm_{ticker}_model.pkl")

    return metrics

def run_all_tickers(
    df: pd.DataFrame,
    tickers: list[str] = None,
    n_trials: int = 30,
) -> pd.DataFrame:
    if tickers is None:
        tickers = [
    # Broad market
    "SPY", "QQQ", "IWM",
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD",
    # Finance
    "JPM", "BAC", "GS", "XLF",
    # Healthcare
    "JNJ", "UNH", "XLV",
    # Energy
    "XOM", "CVX", "XLE",
    # Consumer
    "AMZN", "COST", "WMT",
    # Sector ETFs
    "XLK", "XLI", "XLP", "XLU", "XLB",
    # Bonds / macro proxy
    "TLT", "GLD",
    ]

    all_metrics = []
    for ticker in tickers:
        if ticker not in df["ticker"].values:
            print(f"Skipping {ticker} — not in dataset")
            continue
        metrics = run_lgbm(df, ticker=ticker, n_trials=n_trials)
        all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)
    print("\nLightGBM summary across tickers:")
    print(summary.to_string(index=False))

    summary.to_parquet("artifacts/lgbm_summary.parquet", index=False)
    return summary

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/full_features_v2.parquet")
    run_all_tickers(df, n_trials=30)