import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPORT_DIR = Path("data/quality_report")

def isolation_forest_anomalies(
    df: pd.DataFrame,
    ticker: str,
    contamination: float = 0.02,
) -> pd.DataFrame:
    """
    Isolation Forest anomaly detection.

    contamination = expected fraction of anomalies in the data.
    0.02 = we expect about 2% of days to be anomalous.

    Features used: return, volatility, volume ratio, RSI.
    Multi-dimensional anomaly — a day is flagged if it's
    unusual across multiple dimensions simultaneously.
    """
    sub = df[df["ticker"] == ticker].sort_values("date").copy()

    feature_cols = ["log_return", "vol_21d", "vol_ratio", "rsi_14"]
    available    = [c for c in feature_cols if c in sub.columns]

    data = sub[available].dropna()

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    # predict returns -1 for anomalies, 1 for normal
    labels = model.fit_predict(data)
    scores = model.score_samples(data)  # more negative = more anomalous

    data = data.copy()
    data["anomaly"]    = labels == -1
    data["iso_score"]  = scores
    data["date"]       = sub.loc[data.index, "date"].values
    data["close"]      = sub.loc[data.index, "close"].values
    data["log_return"] = sub.loc[data.index, "log_return"].values

    n_anomalies = data["anomaly"].sum()
    print(f"\nIsolation Forest — {ticker}")
    print(f"  Total days:  {len(data)}")
    print(f"  Anomalies:   {n_anomalies} ({n_anomalies/len(data)*100:.1f}%)")

    # Print most anomalous days
    top_anomalies = (
        data[data["anomaly"]]
        .sort_values("iso_score")
        .head(10)[["date", "close", "log_return", "iso_score"]]
    )
    print(f"\nMost anomalous days:")
    print(top_anomalies.to_string(index=False))

    return data

def zscore_anomalies(
    df: pd.DataFrame,
    ticker: str,
    window: int = 21,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Rolling z-score anomaly detection.
    Flag days where return is more than `threshold` std devs
    from the rolling mean — simple but interpretable.
    """
    sub = df[df["ticker"] == ticker].sort_values("date").copy()
    sub = sub.dropna(subset=["log_return"])

    roll_mean = sub["log_return"].rolling(window).mean()
    roll_std  = sub["log_return"].rolling(window).std()
    sub["zscore"] = (sub["log_return"] - roll_mean) / roll_std.replace(0, np.nan)
    sub["zscore_anomaly"] = sub["zscore"].abs() > threshold

    n_anomalies = sub["zscore_anomaly"].sum()
    print(f"\nZ-score anomalies — {ticker} (threshold={threshold})")
    print(f"  Anomalies: {n_anomalies} ({n_anomalies/len(sub)*100:.1f}%)")

    top = (
        sub[sub["zscore_anomaly"]]
        .sort_values("zscore", key=abs, ascending=False)
        .head(10)[["date", "close", "log_return", "zscore"]]
    )
    print(top.to_string(index=False))

    return sub

def plot_anomalies(
    iso_data: pd.DataFrame,
    zscore_data: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """
    Two panel chart:
    Top — price with anomalies highlighted
    Bottom — log returns with anomaly flags
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"{ticker} price — anomalies highlighted",
            "log returns — red=isolation forest, orange=z-score",
        ],
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=iso_data["date"],
        y=iso_data["close"],
        name="price",
        line=dict(color="#4fc3f7", width=1),
    ), row=1, col=1)

    # Isolation forest anomalies on price
    iso_anom = iso_data[iso_data["anomaly"]]
    fig.add_trace(go.Scatter(
        x=iso_anom["date"],
        y=iso_anom["close"],
        name="isolation forest anomaly",
        mode="markers",
        marker=dict(color="#e57373", size=6, symbol="x"),
    ), row=1, col=1)

    # Log returns
    fig.add_trace(go.Scatter(
        x=zscore_data["date"],
        y=zscore_data["log_return"],
        name="log return",
        line=dict(color="#81d4fa", width=0.8),
    ), row=2, col=1)

    # Z-score anomalies on returns
    z_anom = zscore_data[zscore_data["zscore_anomaly"]]
    fig.add_trace(go.Scatter(
        x=z_anom["date"],
        y=z_anom["log_return"],
        name="z-score anomaly",
        mode="markers",
        marker=dict(color="#ffb74d", size=7, symbol="diamond"),
    ), row=2, col=1)

    fig.update_layout(
        title=f"Anomaly detection — {ticker}",
        height=600,
        legend=dict(x=0.01, y=0.99),
    )
    return fig

def plot_anomaly_calendar(
    iso_data: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """
    Anomaly score over time — shows clustering of anomalous periods.
    Lower score = more anomalous.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iso_data["date"],
        y=iso_data["iso_score"],
        mode="lines",
        line=dict(color="#81d4fa", width=1),
        name="anomaly score",
    ))

    # Shade anomalous regions
    anom = iso_data[iso_data["anomaly"]]
    fig.add_trace(go.Scatter(
        x=anom["date"],
        y=anom["iso_score"],
        mode="markers",
        marker=dict(color="#e57373", size=5),
        name="flagged anomaly",
    ))

    fig.add_hline(
        y=iso_data["iso_score"].quantile(0.02),
        line_dash="dash",
        line_color="#ffb74d",
        annotation_text="2% threshold",
    )
    fig.update_layout(
        title=f"Isolation Forest anomaly score over time — {ticker}",
        xaxis_title="date",
        yaxis_title="anomaly score (lower = more anomalous)",
        height=380,
    )
    return fig

def run_anomaly_detection(
    df: pd.DataFrame,
    tickers: list[str] = None,
) -> pd.DataFrame:
    if tickers is None:
        tickers = ["SPY", "QQQ", "AAPL", "NVDA"]

    all_anomalies = []

    for ticker in tickers:
        if ticker not in df["ticker"].values:
            continue

        # Run both methods
        iso_data    = isolation_forest_anomalies(df, ticker)
        zscore_data = zscore_anomalies(df, ticker)

        # Save charts
        fig1 = plot_anomalies(iso_data, zscore_data, ticker)
        fig2 = plot_anomaly_calendar(iso_data, ticker)
        fig1.write_html(REPORT_DIR / f"anomalies_{ticker}.html")
        fig2.write_html(REPORT_DIR / f"anomaly_score_{ticker}.html")
        print(f"Charts saved for {ticker}")

        # Collect anomaly dates for summary
        iso_anom_dates = iso_data[iso_data["anomaly"]][["date", "log_return"]].copy()
        iso_anom_dates["ticker"] = ticker
        iso_anom_dates["method"] = "isolation_forest"
        all_anomalies.append(iso_anom_dates)

    if all_anomalies:
        combined = pd.concat(all_anomalies, ignore_index=True)
        combined.to_parquet(
            "data/processed/anomaly_dates.parquet",
            index=False,
        )
        print(f"\nTotal anomalies saved: {len(combined)}")
        return combined

    return pd.DataFrame()

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/full_features.parquet")
    anomalies = run_anomaly_detection(df)