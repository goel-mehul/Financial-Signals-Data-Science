import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPORT_DIR = Path("data/quality_report")

def select_arima_order(
    series: pd.Series,
    max_p: int = 4,
    max_q: int = 4,
) -> tuple:
    """
    Grid search over (p, d=1, q) orders.
    Pick the combination with the lowest AIC score.
    AIC balances model fit vs complexity — penalizes overfitting.
    """
    best_aic   = np.inf
    best_order = (1, 1, 0)

    print("Selecting ARIMA order via AIC grid search...")
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, 1, q)).fit()
                if model.aic < best_aic:
                    best_aic   = model.aic
                    best_order = (p, 1, q)
            except Exception:
                continue

    print(f"Best order: ARIMA{best_order}  AIC: {best_aic:.2f}")
    return best_order

def walk_forward_arima(
    series: pd.Series,
    order: tuple,
    train_size: int = 504,  # ~2 trading years
    step: int = 21,         # re-fit every month
) -> pd.DataFrame:
    """
    Walk-forward validation:
    1. Train on series[t - train_size : t]
    2. Predict next `step` values
    3. Move t forward by `step`
    4. Repeat — never look at future data

    This is the correct way to evaluate time series models.
    A simple train/test split would leak future data.
    """
    predictions, actuals, dates = [], [], []
    total_steps = (len(series) - train_size) // step

    print(f"Running walk-forward validation ({total_steps} folds)...")

    for i, start in enumerate(range(0, len(series) - train_size - step, step)):
        train = series.iloc[start : start + train_size]
        test  = series.iloc[start + train_size : start + train_size + step]

        try:
            model = ARIMA(train, order=order).fit()
            pred  = model.forecast(steps=step)
        except Exception as e:
            continue

        predictions.extend(pred.tolist())
        actuals.extend(test.tolist())
        dates.extend(test.index.tolist())

        if (i + 1) % 5 == 0:
            print(f"  Fold {i+1}/{total_steps} done")

    results = pd.DataFrame({
        "date":      dates,
        "actual":    actuals,
        "predicted": predictions,
    })

    # Evaluation metrics
    mae  = np.mean(np.abs(results["actual"] - results["predicted"]))
    rmse = np.sqrt(np.mean((results["actual"] - results["predicted"]) ** 2))
    hit_rate = np.mean(
        np.sign(results["actual"]) == np.sign(results["predicted"])
    )

    print(f"\nWalk-forward results:")
    print(f"  MAE:      {mae:.6f}")
    print(f"  RMSE:     {rmse:.6f}")
    print(f"  Hit rate: {hit_rate:.2%}  (did we get the direction right?)")

    return results

def plot_predictions(results: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["actual vs predicted returns",
                                        "prediction error"])
    fig.add_trace(go.Scatter(
        x=results["date"], y=results["actual"],
        name="actual", line=dict(color="#4fc3f7", width=1)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=results["date"], y=results["predicted"],
        name="predicted", line=dict(color="#ffb74d", width=1)
    ), row=1, col=1)

    error = results["actual"] - results["predicted"]
    fig.add_trace(go.Scatter(
        x=results["date"], y=error,
        name="error", line=dict(color="#e57373", width=1),
        fill="tozeroy", fillcolor="rgba(229,115,115,0.15)"
    ), row=2, col=1)

    fig.update_layout(
        title=f"ARIMA walk-forward — {ticker}",
        height=500,
    )
    return fig

def run_arima(ticker: str = "SPY") -> pd.DataFrame:
    df = pd.read_parquet("data/processed/ohlcv_returns.parquet")
    sub = df[df["ticker"] == ticker].sort_values("date")

    log_ret = sub.set_index("date")["log_return"].dropna()
    print(f"\nARIMA on {ticker} log returns ({len(log_ret)} observations)")

    order   = select_arima_order(log_ret)
    results = walk_forward_arima(log_ret, order)

    # Save results
    results.to_parquet(
        f"data/processed/arima_{ticker}_results.parquet",
        index=False
    )

    # Save chart
    fig = plot_predictions(results, ticker)
    fig.write_html(REPORT_DIR / f"arima_{ticker}.html")
    print(f"Chart saved: data/quality_report/arima_{ticker}.html")

    return results

if __name__ == "__main__":
    run_arima("SPY")