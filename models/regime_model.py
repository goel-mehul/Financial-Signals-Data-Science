import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import pickle
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from models.train import prepare_dataset, FEATURE_COLS, WalkForwardResult
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def split_by_regime(
    data: pd.DataFrame,
    regime_col: str = "vix_regime_elevated",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into two regimes:
    - calm: VIX < 20 (low + normal)
    - stressed: VIX >= 20 (elevated + crisis)
    """
    calm = data[
        (data.get("vix_regime_low", 0) == 1) |
        (data.get("vix_regime_normal", 0) == 1) |
        (data["vix_regime_elevated"] == 0)
    ].copy() if "vix_regime_low" in data.columns else data[data[regime_col] == 0].copy()

    stressed = data[
        (data["vix_regime_elevated"] == 1) |
        (data["vix_regime_crisis"] == 1)
    ].copy() if "vix_regime_crisis" in data.columns else data[data[regime_col] == 1].copy()

    return calm, stressed

def optuna_objective_regime(trial, X, y, n_splits=3):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 7),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "random_state": 42, "verbosity": -1, "n_jobs": -1,
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ics  = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        model   = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                             lgb.log_evaluation(-1)])
        preds = model.predict(X_test)
        ic, _ = spearmanr(preds, y_test)
        ics.append(ic if not np.isnan(ic) else 0.0)
    return float(np.mean(ics))

def train_regime_model(
    data: pd.DataFrame,
    features: list[str],
    regime_name: str,
    n_trials: int = 20,
) -> dict:
    """Train a LightGBM model tuned for a specific regime."""
    if len(data) < 100:
        print(f"  {regime_name}: not enough data ({len(data)} rows) — skipping")
        return None

    available = [c for c in features if c in data.columns]
    X = data[available].values
    y = data["fwd_5d_return"].values

    print(f"  {regime_name}: {len(data)} rows, {len(available)} features")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective_regime(trial, X, y, n_splits=3),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    best_params.update({"random_state": 42, "verbosity": -1, "n_jobs": -1})
    print(f"  {regime_name} best IC: {study.best_value:.4f}")

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model   = lgb.LGBMRegressor(**best_params)
    model.fit(X_scaled, y)

    return {
        "model":    model,
        "scaler":   scaler,
        "features": available,
        "params":   best_params,
        "best_ic":  study.best_value,
        "n_rows":   len(data),
    }

def walk_forward_regime(
    df: pd.DataFrame,
    ticker: str,
    n_splits: int = 5,
    n_trials: int = 20,
) -> WalkForwardResult:
    """
    Walk-forward validation with regime-conditioned models.
    At each fold:
    1. Split training data into calm/stressed regimes
    2. Train separate model for each regime
    3. On test data, route each row to the appropriate model
    """
    data     = prepare_dataset(df, ticker)
    available = [c for c in FEATURE_COLS if c in data.columns]

    X     = data[available].values
    y     = data["fwd_5d_return"].values
    dates = data["date"].values

    # Get regime labels
    vix_elevated = data.get(
        "vix_regime_elevated",
        pd.Series(np.zeros(len(data)))
    ).values
    vix_crisis = data.get(
        "vix_regime_crisis",
        pd.Series(np.zeros(len(data)))
    ).values
    is_stressed = (vix_elevated == 1) | (vix_crisis == 1)

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    result = WalkForwardResult(ticker=ticker, model_name="regime_lgbm")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n  Fold {fold+1}/{n_splits}")

        train_data = data.iloc[train_idx].copy()
        test_data  = data.iloc[test_idx].copy()

        # Split training into regimes
        train_calm     = train_data[~is_stressed[train_idx]]
        train_stressed = train_data[is_stressed[train_idx]]

        print(f"    Train calm: {len(train_calm)} | stressed: {len(train_stressed)}")

        # Train regime-specific models
        calm_model     = train_regime_model(train_calm,     available, "calm",     n_trials)
        stressed_model = train_regime_model(train_stressed, available, "stressed", n_trials)

        # Fallback — if one regime has too little data use the other
        if calm_model is None and stressed_model is None:
            continue
        if calm_model is None:
            calm_model = stressed_model
        if stressed_model is None:
            stressed_model = calm_model

        # Predict on test — route to appropriate model
        preds = np.zeros(len(test_idx))
        test_stressed_mask = is_stressed[test_idx]

        for i, (idx, row) in enumerate(test_data.iterrows()):
            x = row[available].values.reshape(1, -1)
            if test_stressed_mask[i]:
                x_scaled = stressed_model["scaler"].transform(x)
                preds[i] = stressed_model["model"].predict(x_scaled)[0]
            else:
                x_scaled = calm_model["scaler"].transform(x)
                preds[i] = calm_model["model"].predict(x_scaled)[0]

        result.predictions.extend(preds.tolist())
        result.actuals.extend(y[test_idx].tolist())
        result.dates.extend(dates[test_idx].tolist())

        fold_ic = WalkForwardResult(
            predictions=preds.tolist(),
            actuals=y[test_idx].tolist()
        ).information_coefficient()
        print(f"    Fold IC: {fold_ic:.4f}")

    return result

def run_regime_models(
    df: pd.DataFrame,
    tickers: list[str] = None,
    n_trials: int = 20,
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
            continue
        print(f"\nRegime-conditioned model — {ticker}")
        result  = walk_forward_regime(df, ticker, n_splits=5, n_trials=n_trials)
        metrics = result.summary()
        print(f"\nResults — {ticker}:")
        print(json.dumps(metrics, indent=2))
        all_metrics.append(metrics)

        # Save predictions
        pd.DataFrame({
            "date":      result.dates,
            "actual":    result.actuals,
            "predicted": result.predictions,
        }).to_parquet(f"artifacts/regime_lgbm_{ticker}_preds.parquet", index=False)

    summary = pd.DataFrame(all_metrics)
    print("\nRegime model summary:")
    print(summary.to_string(index=False))
    summary.to_parquet("artifacts/regime_summary.parquet", index=False)
    return summary

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/full_features_v2.parquet")
    run_regime_models(df, n_trials=20)