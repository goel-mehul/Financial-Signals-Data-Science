import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPORT_DIR = Path("data/quality_report")

def decompose_series(
    series: pd.Series,
    period: int = 12,
    name: str = ""
) -> dict:
    """
    STL decomposition.
    period=12 for monthly data (12 months in a year).
    robust=True makes it resistant to outliers.
    """
    series = series.dropna().sort_index()

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    print(f"\nSTL decomposition — {name}")
    print(f"  Series length:     {len(series)}")
    print(f"  Trend range:       {result.trend.min():.2f} to {result.trend.max():.2f}")
    print(f"  Seasonal strength: {result.seasonal.std():.4f}")
    print(f"  Residual std:      {result.resid.std():.4f}")

    return {
        "name":     name,
        "original": series,
        "trend":    result.trend,
        "seasonal": result.seasonal,
        "resid":    result.resid,
    }

def plot_decomposition(decomp: dict) -> go.Figure:
    """4-panel plot: original, trend, seasonal, residual."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"{decomp['name']} — original",
            "trend",
            "seasonal",
            "residual",
        ],
        vertical_spacing=0.08,
    )

    components = [
        (decomp["original"], "#4fc3f7", "original"),
        (decomp["trend"],    "#81d4fa", "trend"),
        (decomp["seasonal"], "#ffb74d", "seasonal"),
        (decomp["resid"],    "#e57373", "residual"),
    ]

    for i, (series, color, name) in enumerate(components, 1):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=name,
                line=dict(color=color, width=1.2),
            ),
            row=i, col=1,
        )

    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

def run_decomposition() -> None:
    macro = pd.read_parquet("data/raw/macro/macro_indicators.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date")

    results = {}

    # Decompose each macro indicator
    for col in ["CPI", "FedFunds", "Unemployment", "Yield10yr"]:
        series = macro[col].dropna()

        # Monthly indicators use period=12
        decomp = decompose_series(series, period=12, name=col)
        results[col] = decomp

        # Save chart
        fig = plot_decomposition(decomp)
        fig.write_html(REPORT_DIR / f"stl_{col.lower()}.html")
        print(f"Chart saved: data/quality_report/stl_{col.lower()}.html")

    # VIX is daily — use period=252 (trading days in a year)
    vix = macro["VIX"].dropna()
    decomp_vix = decompose_series(vix, period=252, name="VIX")
    results["VIX"] = decomp_vix
    fig = plot_decomposition(decomp_vix)
    fig.write_html(REPORT_DIR / "stl_vix.html")
    print("Chart saved: data/quality_report/stl_vix.html")

    # Save trend components as a combined parquet for later use
    trends = pd.DataFrame({
        name: decomp["trend"]
        for name, decomp in results.items()
    })
    trends.to_parquet("data/processed/macro_trends.parquet")
    print(f"\nTrend components saved: {trends.shape}")

if __name__ == "__main__":
    run_decomposition()