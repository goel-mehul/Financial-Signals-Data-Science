import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPORT_DIR = Path("data/quality_report")
SECTOR_ETFS = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLP"]

def build_returns_matrix(
    df: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Pivot OHLCV to wide log-return matrix.
    Each column is one ticker, each row is one date.
    Drop any date where any ticker has missing data.
    """
    available = [t for t in tickers if t in df["ticker"].values]
    frames = {}
    for t in available:
        sub = (
            df[df["ticker"] == t]
            .set_index("date")["log_return"]
            .sort_index()
        )
        frames[t] = sub

    wide = pd.DataFrame(frames).dropna()
    print(f"Returns matrix: {wide.shape[0]} dates x {wide.shape[1]} tickers")
    print(f"Tickers included: {list(wide.columns)}")
    return wide

def select_var_lag(returns: pd.DataFrame, max_lags: int = 10) -> int:
    """
    Select optimal VAR lag order using AIC.
    More lags = more history captured but more parameters to estimate.
    AIC penalizes complexity to prevent overfitting.
    """
    model  = VAR(returns)
    result = model.select_order(max_lags)
    best   = result.selected_orders["aic"]
    print(f"\nVAR lag order selected by AIC: {best}")
    print(result.summary())
    return best

def fit_var(returns: pd.DataFrame, lag: int) -> dict:
    """Fit VAR model and print summary statistics."""
    model  = VAR(returns)
    fitted = model.fit(lag)

    print(f"\nVAR({lag}) fitted successfully")
    print(f"  AIC:  {fitted.aic:.4f}")
    print(f"  BIC:  {fitted.bic:.4f}")
    print(f"  HQIC: {fitted.hqic:.4f}")

    return {"model": fitted, "lag": lag, "tickers": list(returns.columns)}

def compute_irf(
    var_out: dict,
    periods: int = 20,
) -> dict:
    """
    Compute Impulse Response Functions.
    IRF answers: if ticker X gets a 1-std shock today,
    how does each other ticker respond over the next `periods` days?
    """
    irf     = var_out["model"].irf(periods)
    tickers = var_out["tickers"]
    return {"irf": irf, "tickers": tickers, "periods": periods}

def plot_irf(
    irf_out: dict,
    shock_ticker: str,
) -> go.Figure:
    """
    Plot the response of all tickers to a 1-std shock in shock_ticker.
    X axis = days after shock, Y axis = response magnitude.
    """
    tickers = irf_out["tickers"]
    periods = irf_out["periods"]
    irf     = irf_out["irf"]

    if shock_ticker not in tickers:
        print(f"{shock_ticker} not in model")
        return go.Figure()

    shock_idx = tickers.index(shock_ticker)
    colors    = ["#4fc3f7", "#81d4fa", "#ffb74d", "#e57373", "#81c784", "#ce93d8"]

    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        response = irf.irfs[:, i, shock_idx]
        fig.add_trace(go.Scatter(
            x=list(range(periods + 1)),
            y=response,
            name=ticker,
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
    fig.update_layout(
        title=f"Impulse response to 1-std shock in {shock_ticker}",
        xaxis_title="days after shock",
        yaxis_title="response (log return units)",
        height=450,
        legend=dict(x=0.01, y=0.99),
    )
    return fig

def plot_forecast_error_variance(var_out: dict, periods: int = 20) -> go.Figure:
    """
    FEVD — what fraction of each ticker's variance is explained
    by shocks from each other ticker?
    Shows how interconnected the sectors are.
    """
    fevd    = var_out["model"].fevd(periods)
    tickers = var_out["tickers"]

    fig = make_subplots(
        rows=1,
        cols=len(tickers),
        subplot_titles=tickers,
    )
    colors = ["#4fc3f7", "#ffb74d", "#e57373", "#81c784", "#ce93d8", "#81d4fa"]

    for col_idx, ticker in enumerate(tickers):
        ticker_idx    = tickers.index(ticker)
        max_periods = fevd.decomp.shape[0]
        safe_period = min(periods - 1, max_periods - 1)
        decomposition = fevd.decomp[safe_period, ticker_idx, :]


        fig.add_trace(go.Bar(
            x=tickers,
            y=decomposition,
            marker_color=colors,
            showlegend=False,
        ), row=1, col=col_idx + 1)

    fig.update_layout(
        title=f"Forecast error variance decomposition (at {periods} days)",
        height=400,
    )
    return fig

def run_var() -> None:
    df = pd.read_parquet("data/processed/ohlcv_returns.parquet")

    # Build returns matrix for sector ETFs
    returns = build_returns_matrix(df, SECTOR_ETFS)

    if returns.shape[1] < 2:
        print("Not enough tickers for VAR — need at least 2")
        return

    # Select lag order
    lag = select_var_lag(returns, max_lags=10)
    lag = max(lag, 1)  # ensure at least lag 1

    # Fit VAR
    var_out = fit_var(returns, lag)

    # Compute IRFs
    irf_out = compute_irf(var_out, periods=20)

    # Plot IRF for each ticker as the shock source
    for shock_ticker in returns.columns:
        fig = plot_irf(irf_out, shock_ticker=shock_ticker)
        fig.write_html(REPORT_DIR / f"var_irf_{shock_ticker}.html")
        print(f"IRF chart saved for shock: {shock_ticker}")

    # Forecast error variance decomposition
    fevd_periods = len(var_out["tickers"])
    fig_fevd = plot_forecast_error_variance(var_out, periods=fevd_periods)
    fig_fevd.write_html(REPORT_DIR / "var_fevd.html")
    print("FEVD chart saved")

    # Save fitted model summary
    summary_path = REPORT_DIR / "var_summary.txt"
    summary_path.write_text(str(var_out["model"].summary()))
    print(f"VAR summary saved to {summary_path}")

if __name__ == "__main__":
    run_var()