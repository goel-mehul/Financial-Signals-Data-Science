import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

REPORT_DIR = Path("data/quality_report")

def prepare_prophet_df(series: pd.Series) -> pd.DataFrame:
    """
    Prophet requires a dataframe with exactly two columns:
    'ds' (datestamp) and 'y' (value).
    """
    df = series.dropna().reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def fit_prophet(
    series: pd.Series,
    name: str = "",
    forecast_periods: int = 90,
    yearly_seasonality: bool = True,
) -> dict:
    """
    Fit Prophet model and forecast forward.

    changepoint_prior_scale controls trend flexibility:
    - Higher = more flexible trend, risk of overfitting
    - Lower  = smoother trend, risk of missing real changes
    0.05 is a good default for macro indicators.
    """
    prophet_df = prepare_prophet_df(series)

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,  # macro data is monthly, no weekly pattern
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95,       # 95% confidence interval
    )
    model.fit(prophet_df)

    # Create future dataframe for forecasting
    future   = model.make_future_dataframe(
        periods=forecast_periods,
        freq="MS",  # month start for monthly macro data
    )
    forecast = model.predict(future)

    # Key stats
    last_actual   = prophet_df["y"].iloc[-1]
    last_forecast = forecast[forecast["ds"] > prophet_df["ds"].max()]["yhat"].iloc[-1]
    direction     = "up" if last_forecast > last_actual else "down"

    print(f"\nProphet forecast — {name}")
    print(f"  Last actual value:     {last_actual:.3f}")
    print(f"  Forecast in {forecast_periods} days:  {last_forecast:.3f}")
    print(f"  Direction:             {direction}")
    print(f"  Changepoints detected: {len(model.changepoints)}")

    return {
        "name":      name,
        "model":     model,
        "forecast":  forecast,
        "actual_df": prophet_df,
    }

def plot_prophet_forecast(prophet_out: dict) -> go.Figure:
    """
    Plot actual values, fitted trend, forecast, and confidence interval.
    """
    name     = prophet_out["name"]
    forecast = prophet_out["forecast"]
    actual   = prophet_out["actual_df"]

    # Split into historical fit and future forecast
    last_actual_date = actual["ds"].max()
    historical = forecast[forecast["ds"] <= last_actual_date]
    future     = forecast[forecast["ds"] > last_actual_date]

    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=actual["ds"], y=actual["y"],
        name="actual",
        line=dict(color="#4fc3f7", width=1.5),
        mode="lines",
    ))

    # Historical fitted trend
    fig.add_trace(go.Scatter(
        x=historical["ds"], y=historical["yhat"],
        name="fitted trend",
        line=dict(color="#81d4fa", width=1, dash="dot"),
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=future["ds"], y=future["yhat"],
        name="forecast",
        line=dict(color="#ffb74d", width=2),
    ))

    # Confidence interval — upper bound
    fig.add_trace(go.Scatter(
        x=future["ds"], y=future["yhat_upper"],
        name="95% upper",
        line=dict(width=0),
        showlegend=False,
    ))

    # Confidence interval — lower bound (filled to upper)
    fig.add_trace(go.Scatter(
        x=future["ds"], y=future["yhat_lower"],
        name="95% confidence interval",
        fill="tonexty",
        fillcolor="rgba(255,183,77,0.2)",
        line=dict(width=0),
    ))

    # Mark the forecast start
    fig.add_trace(go.Scatter(
        x=[last_actual_date, last_actual_date],
        y=[actual["y"].min(), actual["y"].max()],
        name="forecast start",
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
    ))

    fig.update_layout(
        title=f"Prophet forecast — {name} (90 days ahead)",
        xaxis_title="date",
        yaxis_title=name,
        height=450,
        legend=dict(x=0.01, y=0.99),
    )
    return fig

def plot_components(prophet_out: dict) -> go.Figure:
    """
    Plot trend and seasonal components separately.
    This shows what Prophet learned about the underlying structure.
    """
    name     = prophet_out["name"]
    forecast = prophet_out["forecast"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["trend component", "yearly seasonality"],
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["trend"],
        name="trend",
        line=dict(color="#4fc3f7", width=1.5),
    ), row=1, col=1)

    if "yearly" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yearly"],
            name="yearly seasonality",
            line=dict(color="#ffb74d", width=1.5),
        ), row=2, col=1)

    fig.update_layout(
        title=f"Prophet components — {name}",
        height=450,
    )
    return fig

def run_prophet() -> None:
    macro = pd.read_parquet("data/raw/macro/macro_indicators.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date")

    # Resample to monthly (Prophet works best with consistent frequency)
    macro_monthly = macro.resample("MS").first()

    indicators = {
        "CPI":          (macro_monthly["CPI"],          True),
        "Unemployment": (macro_monthly["Unemployment"], True),
        "Yield10yr":    (macro_monthly["Yield10yr"],    True),
        "FedFunds":     (macro_monthly["FedFunds"],     True),
    }

    forecasts = {}

    for name, (series, yearly) in indicators.items():
        out = fit_prophet(
            series,
            name=name,
            forecast_periods=3,   # 3 months ahead for monthly data
            yearly_seasonality=yearly,
        )
        forecasts[name] = out

        # Save charts
        fig1 = plot_prophet_forecast(out)
        fig2 = plot_components(out)
        fig1.write_html(REPORT_DIR / f"prophet_{name.lower()}.html")
        fig2.write_html(REPORT_DIR / f"prophet_{name.lower()}_components.html")
        print(f"Charts saved for {name}")

    # Save forecast values for later use in feature engineering
    forecast_rows = []
    for name, out in forecasts.items():
        future_fc = out["forecast"][
            out["forecast"]["ds"] > out["actual_df"]["ds"].max()
        ][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        future_fc["indicator"] = name
        forecast_rows.append(future_fc)

    if forecast_rows:
        all_forecasts = pd.concat(forecast_rows, ignore_index=True)
        all_forecasts.to_parquet(
            "data/processed/prophet_forecasts.parquet",
            index=False,
        )
        print(f"\nForecasts saved: {all_forecasts.shape}")
        print(all_forecasts.to_string(index=False))

if __name__ == "__main__":
    run_prophet()