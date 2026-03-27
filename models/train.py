import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "rsi_14", "atr_14", "zscore_63",
    "mom_5d", "mom_10d", "mom_21d", "mom_63d",
    "vol_5d", "vol_21d", "vol_63d",
    "macd", "macd_signal", "macd_hist",
    "bb_pct",
    "CPI_lag1", "FedFunds_lag1", "Unemployment_lag1",
    "Yield10yr_lag1", "M2_lag1", "VIX_lag1",
    "CPI_mom_change", "yield_21d_change", "vix_high_regime",
    "vol_ratio",
]
TARGET_COL = "fwd_5d_return"

@dataclass
class WalkForwardResult:
    predictions: list[float] = field(default_factory=list)
    actuals:     list[float] = field(default_factory=list)
    dates:       list        = field(default_factory=list)
    ticker:      str         = ""
    model_name:  str         = ""

    def sharpe(self, annualize: bool = True) -> float:
        """
        Sharpe ratio of a long/short strategy following the signal.
        Go long when prediction > 0, short when prediction < 0.
        Annualized by sqrt(252/5) since we're predicting 5-day returns.
        """
        preds = np.array(self.predictions)
        acts  = np.array(self.actuals)
        strategy_ret = np.sign(preds) * acts
        mean = strategy_ret.mean()
        std  = strategy_ret.std() + 1e-9
        raw  = mean / std
        return float(round(raw * np.sqrt(252 / 5) if annualize else raw, 4))

    def information_coefficient(self) -> float:
        """
        Rank correlation between predictions and actuals.
        IC > 0.05 is considered useful in practice.
        IC > 0.10 is considered strong.
        """
        from scipy.stats import spearmanr
        if len(self.predictions) < 10:
            return 0.0
        corr, _ = spearmanr(self.predictions, self.actuals)
        return round(float(corr), 4)

    def hit_rate(self) -> float:
        """
        Fraction of predictions where we got the direction right.
        50% = random, above 52% is generally useful.
        """
        preds = np.array(self.predictions)
        acts  = np.array(self.actuals)
        return round(float(np.mean(np.sign(preds) == np.sign(acts))), 4)

    def max_drawdown(self) -> float:
        """
        Largest peak-to-trough loss of the strategy.
        Important risk metric alongside Sharpe.
        """
        preds = np.array(self.predictions)
        acts  = np.array(self.actuals)
        cumret = np.cumsum(np.sign(preds) * acts)
        peak   = np.maximum.accumulate(cumret)
        dd     = cumret - peak
        return round(float(dd.min()), 4)

    def summary(self) -> dict:
        return {
            "ticker":    self.ticker,
            "model":     self.model_name,
            "n_preds":   len(self.predictions),
            "sharpe":    self.sharpe(),
            "ic":        self.information_coefficient(),
            "hit_rate":  self.hit_rate(),
            "max_dd":    self.max_drawdown(),
        }

def prepare_dataset(
    df: pd.DataFrame,
    ticker: str,
    feature_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Filter to one ticker, select features and target,
    drop rows with any nulls, sort by date.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Only use columns that exist in the dataframe
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns {missing}")

    sub = (
        df[df["ticker"] == ticker]
        .sort_values("date")
        .copy()
    )
    sub = sub[available + [TARGET_COL, "date"]].dropna()
    print(f"  {ticker}: {len(sub)} clean rows, {len(available)} features")
    return sub

def walk_forward_train(
    model_cls,
    df: pd.DataFrame,
    ticker: str,
    n_splits: int = 5,
    model_name: str = "",
    feature_cols: list[str] = None,
    **model_kwargs,
) -> WalkForwardResult:
    """
    Walk-forward validation using TimeSeriesSplit.

    TimeSeriesSplit creates folds where:
    - Fold 1: train on first 20%, test on next 20%
    - Fold 2: train on first 40%, test on next 20%
    - Fold 3: train on first 60%, test on next 20%
    etc.

    Each fold's test set is always strictly after its training set.
    This is the correct way to evaluate time series models.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    data = prepare_dataset(df, ticker, feature_cols)
    if len(data) < 200:
        print(f"  Not enough data for {ticker} — skipping")
        return WalkForwardResult(ticker=ticker, model_name=model_name)

    available_features = [c for c in feature_cols if c in data.columns]
    X     = data[available_features].values
    y     = data[TARGET_COL].values
    dates = data["date"].values

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    result = WalkForwardResult(ticker=ticker, model_name=model_name)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features — fit scaler on train only, apply to test
        # This prevents leaking test distribution into training
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        result.predictions.extend(preds.tolist())
        result.actuals.extend(y_test.tolist())
        result.dates.extend(dates[test_idx].tolist())

        fold_ic = WalkForwardResult(
            predictions=preds.tolist(),
            actuals=y_test.tolist(),
        ).information_coefficient()
        print(f"    Fold {fold+1}/{n_splits}: IC={fold_ic:.4f}  n={len(test_idx)}")

    return result

def save_predictions(result: WalkForwardResult, output_dir: str = "artifacts") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    pd.DataFrame({
        "date":      result.dates,
        "actual":    result.actuals,
        "predicted": result.predictions,
    }).to_parquet(
        f"{output_dir}/{result.model_name}_{result.ticker}_preds.parquet",
        index=False,
    )