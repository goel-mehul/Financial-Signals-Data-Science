import pandas as pd
import numpy as np
from arch import arch_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPORT_DIR = Path("data/quality_report")

def fit_garch(
    log_returns: pd.Series,
    ticker: str = "",
    p: int = 1,
    q: int = 1,
) -> dict:
    """
    Fit GARCH(p, q) model on log returns.

    We scale returns by 100 for numerical stability —
    GARCH optimization works better on percentages than decimals.
    We unscale the output volatility afterward.

    Key output: conditional_volatility — the model's estimate of
    how volatile the market is at each point in time.
    """
    scaled = log_returns.dropna() * 100

    model  = arch_model(scaled, vol="Garch", p=p, q=q, dist="normal")
    result = model.fit(disp="off")

    # Unscale back to return units
    cond_vol     = result.conditional_volatility / 100
    realized_vol = log_returns.rolling(21).std()

    print(f"\nGARCH({p},{q}) — {ticker}")
    print(f"  AIC:   {result.aic:.2f}")
    print(f"  BIC:   {result.bic:.2f}")
    print(result.params.round(4).to_string())

    # Volatility persistence = alpha + beta
    # Close to 1 means shocks persist for a long time
    alpha = result.params.get("alpha[1]", 0)
    beta  = result.params.get("beta[1]", 0)
    persistence = alpha + beta
    print(f"  Volatility persistence (alpha+beta): {persistence:.4f}")
    if persistence > 0.95:
        print("  → High persistence: volatility shocks last a long time")
    else:
        print("  → Moderate persistence: volatility mean-reverts relatively quickly")

    return {
        "ticker":       ticker,
        "result":       result,
        "cond_vol":     cond_vol,
        "realized_vol": realized_vol,
        "aic":          result.aic,
        "bic":          result.bic,
        "persistence":  persistence,
    }

def plot_volatility(garch_out: dict) -> go.Figure:
    """
    Two panel chart:
    Top — conditional volatility from GARCH (forward-looking estimate)
    Bottom — realized volatility (21-day rolling std, backward-looking)

    Comparing these shows how well GARCH tracks actual market volatility.
    """
    ticker   = garch_out["ticker"]
    cond_vol = garch_out["cond_vol"]
    real_vol = garch_out["realized_vol"].dropna()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "GARCH conditional volatility (forward estimate)",
            "realized volatility — 21d rolling std (backward)",
        ],
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=cond_vol.index, y=cond_vol.values,
        name="GARCH vol", line=dict(color="#e57373", width=1.2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=real_vol.index, y=real_vol.values,
        name="realized vol", line=dict(color="#4fc3f7", width=1.2)
    ), row=2, col=1)

    # Shade the COVID crash period
    for row in [1, 2]:
        fig.add_vrect(
            x0="2020-02-20", x1="2020-04-01",
            fillcolor="rgba(255,183,77,0.15)",
            layer="below", line_width=0,
            row=row, col=1,
        )

    fig.update_layout(
        title=f"Volatility analysis — {ticker}",
        height=550,
        showlegend=True,
    )
    return fig

def plot_volatility_clustering(log_returns: pd.Series, ticker: str) -> go.Figure:
    """
    Show volatility clustering directly —
    plot |returns| over time. Clusters of large values = clustering.
    """
    abs_ret = log_returns.abs().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=abs_ret.index, y=abs_ret.values,
        name="|log return|",
        line=dict(color="#81d4fa", width=0.8),
    ))
    fig.update_layout(
        title=f"Volatility clustering — {ticker} |log returns|",
        xaxis_title="date",
        yaxis_title="|log return|",
        height=350,
    )
    return fig

def run_garch() -> None:
    df = pd.read_parquet("data/processed/ohlcv_returns.parquet")

    results_summary = []

    for ticker in ["SPY", "QQQ", "XLF", "XLK"]:
        if ticker not in df["ticker"].values:
            continue

        sub     = df[df["ticker"] == ticker].sort_values("date")
        log_ret = sub.set_index("date")["log_return"].dropna()

        # Fit GARCH
        out = fit_garch(log_ret, ticker=ticker)
        results_summary.append({
            "ticker":      ticker,
            "aic":         round(out["aic"], 2),
            "bic":         round(out["bic"], 2),
            "persistence": round(out["persistence"], 4),
            "mean_cond_vol": round(out["cond_vol"].mean(), 6),
        })

        # Save charts
        fig1 = plot_volatility(out)
        fig2 = plot_volatility_clustering(log_ret, ticker)
        fig1.write_html(REPORT_DIR / f"garch_{ticker}.html")
        fig2.write_html(REPORT_DIR / f"vol_clustering_{ticker}.html")
        print(f"Charts saved for {ticker}")

        # Save conditional volatility series for feature engineering later
        out["cond_vol"].to_frame(name="cond_vol").to_parquet(
            f"data/processed/garch_vol_{ticker}.parquet"
        )

    print("\nGARCH summary:")
    print(pd.DataFrame(results_summary).to_string(index=False))

if __name__ == "__main__":
    run_garch()