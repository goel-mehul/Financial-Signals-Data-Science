import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

REPORT_DIR = Path("data/quality_report")

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"])
    df["log_return"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    return df

def plot_return_distribution(df: pd.DataFrame, ticker: str) -> go.Figure:
    sub = df[df["ticker"] == ticker]["log_return"].dropna()
    fig = px.histogram(
        sub, nbins=100,
        title=f"{ticker} — log return distribution",
        labels={"value": "log return", "count": "frequency"},
    )
    fig.add_vline(x=sub.mean(), line_dash="dash",
                  annotation_text=f"mean: {sub.mean():.4f}")
    fig.add_vline(x=sub.std(), line_dash="dot", line_color="red",
                  annotation_text=f"std: {sub.std():.4f}")
    return fig

def plot_rolling_stats(df: pd.DataFrame, ticker: str, window: int = 21) -> go.Figure:
    sub = df[df["ticker"] == ticker].copy().sort_values("date")
    sub["roll_mean"] = sub["close"].rolling(window).mean()
    sub["roll_std"]  = sub["close"].rolling(window).std()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["price + rolling mean", "rolling std (volatility)"])
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"],
                             name="close", line=dict(color="#4fc3f7", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["roll_mean"],
                             name=f"{window}d mean", line=dict(color="#ffb74d", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["roll_std"],
                             name=f"{window}d std", line=dict(color="#e57373", width=1.5)), row=2, col=1)
    fig.update_layout(title=f"{ticker} — rolling statistics (window={window})", height=500)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, tickers: list[str]) -> go.Figure:
    wide = df[df["ticker"].isin(tickers)].pivot(
        index="date", columns="ticker", values="log_return"
    ).dropna()
    corr = wide.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Return correlations across tickers",
    )
    return fig

def run_eda() -> None:
    df = pd.read_parquet("data/processed/ohlcv_clean.parquet")
    df = compute_log_returns(df)

    # Save enriched data with log returns
    df.to_parquet("data/processed/ohlcv_returns.parquet", index=False)
    print(f"Log returns computed. Shape: {df.shape}")

    # Key stats per ticker
    stats = df.groupby("ticker")["log_return"].agg(
        mean="mean", std="std", skew="skew"
    ).round(6)
    print("\nReturn stats (first 10 tickers):")
    print(stats.head(10))

    # Save charts for a few key tickers
    for ticker in ["SPY", "AAPL", "NVDA"]:
        if ticker not in df["ticker"].values:
            continue
        fig1 = plot_return_distribution(df, ticker)
        fig2 = plot_rolling_stats(df, ticker)
        fig1.write_html(REPORT_DIR / f"returns_{ticker}.html")
        fig2.write_html(REPORT_DIR / f"rolling_{ticker}.html")
        print(f"Charts saved for {ticker}")

    # Correlation heatmap across sector ETFs
    etfs = ["SPY", "QQQ", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP"]
    available = [t for t in etfs if t in df["ticker"].values]
    fig3 = plot_correlation_heatmap(df, available)
    fig3.write_html(REPORT_DIR / "correlation_heatmap.html")
    print("Correlation heatmap saved")

if __name__ == "__main__":
    run_eda()