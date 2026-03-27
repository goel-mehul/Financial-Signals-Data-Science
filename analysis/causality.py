import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPORT_DIR = Path("data/quality_report")

def granger_test(
    y: pd.Series,
    x: pd.Series,
    max_lag: int = 5,
    name_y: str = "Y",
    name_x: str = "X",
) -> pd.DataFrame:
    """
    Test whether X Granger-causes Y at lags 1 through max_lag.

    At each lag k we ask: does adding X[t-k] to a model of Y[t]
    significantly improve prediction over using Y's own lags alone?

    Low p-value = yes, X helps predict Y = Granger causality.
    """
    combined = pd.DataFrame({"y": y, "x": x}).dropna()
    print(f"\nGranger causality: does {name_x} → {name_y}?")
    print(f"Sample size: {len(combined)} observations")

    raw = grangercausalitytests(
        combined[["y", "x"]],
        maxlag=max_lag,
        verbose=False,
    )

    rows = []
    for lag, res in raw.items():
        f_stat = res[0]["ssr_ftest"][0]
        p_val  = res[0]["ssr_ftest"][1]
        rows.append({
            "lag":         lag,
            "f_stat":      round(f_stat, 3),
            "p_value":     round(p_val, 4),
            "significant": p_val < 0.05,
        })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    sig_lags = df_out[df_out["significant"]]["lag"].tolist()
    if sig_lags:
        print(f"\nSignificant at lags: {sig_lags}")
        print(f"→ {name_x} Granger-causes {name_y} at these lags")
    else:
        print(f"\nNo significant lags found")
        print(f"→ {name_x} does NOT Granger-cause {name_y}")

    return df_out

def plot_granger_results(
    results: pd.DataFrame,
    name_y: str,
    name_x: str,
) -> go.Figure:
    """Bar chart of p-values across lags. Red line at 0.05 significance threshold."""
    colors = [
        "#e57373" if sig else "#4fc3f7"
        for sig in results["significant"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results["lag"],
        y=results["p_value"],
        marker_color=colors,
        name="p-value",
    ))
    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="#ffb74d",
        annotation_text="p = 0.05 threshold",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Granger causality: {name_x} → {name_y}",
        xaxis_title="lag (days)",
        yaxis_title="p-value",
        yaxis=dict(range=[0, max(results["p_value"].max() + 0.05, 0.15)]),
        height=380,
    )
    return fig

def plot_vix_vs_returns(
    vix: pd.Series,
    returns: pd.Series,
    ticker: str = "SPY",
) -> go.Figure:
    """Dual axis chart — VIX level vs next day return."""
    common = vix.index.intersection(returns.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=common,
        y=vix.loc[common],
        name="VIX",
        line=dict(color="#ffb74d", width=1.2),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=common,
        y=returns.loc[common],
        name=f"{ticker} log return",
        line=dict(color="#4fc3f7", width=0.8),
        yaxis="y2",
    ))
    fig.update_layout(
        title=f"VIX vs {ticker} returns",
        yaxis=dict(title="VIX", side="left"),
        yaxis2=dict(title="log return", side="right", overlaying="y"),
        height=400,
        legend=dict(x=0.01, y=0.99),
    )
    return fig

def run_causality() -> None:
    # Load data
    ohlcv  = pd.read_parquet("data/processed/ohlcv_returns.parquet")
    macro  = pd.read_parquet("data/raw/macro/macro_indicators.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    macro  = macro.set_index("date")

    vix = macro["VIX"].dropna()

    pairs = [
        ("SPY", "VIX"),
        ("QQQ", "VIX"),
        ("XLF", "VIX"),
        ("SPY", "FedFunds"),
        ("SPY", "Yield10yr"),
    ]

    all_results = {}

    for asset_ticker, macro_name in pairs:
        if asset_ticker not in ohlcv["ticker"].values:
            continue

        asset = (
            ohlcv[ohlcv["ticker"] == asset_ticker]
            .set_index("date")["log_return"]
            .dropna()
        )
        macro_series = macro[macro_name].dropna()

        # Align on common dates
        common = asset.index.intersection(macro_series.index)
        if len(common) < 100:
            continue

        results = granger_test(
            y=asset.loc[common],
            x=macro_series.loc[common],
            max_lag=5,
            name_y=f"{asset_ticker} returns",
            name_x=macro_name,
        )
        all_results[f"{asset_ticker}_{macro_name}"] = results

        # Save chart
        fig = plot_granger_results(
            results,
            name_y=f"{asset_ticker} returns",
            name_x=macro_name,
        )
        fig.write_html(
            REPORT_DIR / f"granger_{asset_ticker}_{macro_name}.html"
        )

    # VIX vs SPY visual
    spy = (
        ohlcv[ohlcv["ticker"] == "SPY"]
        .set_index("date")["log_return"]
        .dropna()
    )
    fig2 = plot_vix_vs_returns(vix, spy, "SPY")
    fig2.write_html(REPORT_DIR / "vix_vs_spy.html")

    print("\nAll Granger causality charts saved to data/quality_report/")

if __name__ == "__main__":
    run_causality()