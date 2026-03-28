import shap
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from models.train import prepare_dataset, FEATURE_COLS
import warnings
warnings.filterwarnings("ignore")

REPORT_DIR = Path("data/quality_report")

def compute_shap_values(
    df: pd.DataFrame,
    ticker: str = "SPY",
    model_path: str = None,
) -> dict:
    """
    Load the saved LightGBM model and compute SHAP values
    for every prediction in the dataset.
    """
    if model_path is None:
        model_path = f"artifacts/lgbm_{ticker}_model.pkl"

    if not Path(model_path).exists():
        print(f"No model found at {model_path} — train first")
        return {}

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model    = saved["model"]
    scaler   = saved["scaler"]
    features = saved["features"]

    data = prepare_dataset(df, ticker, feature_cols=features)
    X    = data[features].values
    X_scaled = scaler.transform(X)

    print(f"Computing SHAP values for {ticker}...")
    print(f"  Dataset: {X_scaled.shape[0]} rows x {X_scaled.shape[1]} features")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Mean absolute SHAP per feature = overall importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature":   features,
        "mean_shap": mean_abs_shap,
    }).sort_values("mean_shap", ascending=False)

    print(f"\nTop 10 features by SHAP importance ({ticker}):")
    print(importance.head(10).to_string(index=False))

    return {
        "ticker":      ticker,
        "shap_values": shap_values,
        "X":           X_scaled,
        "X_raw":       data[features],
        "features":    features,
        "importance":  importance,
        "dates":       data["date"].values,
    }

def plot_feature_importance(shap_out: dict) -> go.Figure:
    """
    Bar chart of mean absolute SHAP values.
    This shows which features matter most overall.
    """
    imp    = shap_out["importance"].head(15)
    ticker = shap_out["ticker"]

    colors = [
        "#4fc3f7" if i < 3 else "#81d4fa" if i < 7 else "#b0bec5"
        for i in range(len(imp))
    ]

    fig = go.Figure(go.Bar(
        x=imp["mean_shap"],
        y=imp["feature"],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"Feature importance — mean |SHAP| ({ticker})",
        xaxis_title="mean absolute SHAP value",
        yaxis=dict(autorange="reversed"),
        height=500,
    )
    return fig

def plot_shap_over_time(shap_out: dict, feature: str) -> go.Figure:
    """
    Show how a feature's SHAP contribution changes over time.
    Positive = feature is pushing prediction up on that day.
    Negative = feature is pushing prediction down on that day.
    """
    ticker  = shap_out["ticker"]
    features = shap_out["features"]

    if feature not in features:
        print(f"{feature} not in feature list")
        return go.Figure()

    feat_idx    = features.index(feature)
    shap_series = shap_out["shap_values"][:, feat_idx]
    dates       = shap_out["dates"]

    colors = ["#e57373" if v < 0 else "#81c784" for v in shap_series]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=shap_series,
        marker_color=colors,
        name=f"SHAP({feature})",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
    fig.update_layout(
        title=f"SHAP contribution of {feature} over time — {ticker}",
        xaxis_title="date",
        yaxis_title="SHAP value",
        height=380,
    )
    return fig

def plot_shap_vs_feature_value(shap_out: dict, feature: str) -> go.Figure:
    """
    Scatter plot of feature value vs its SHAP contribution.
    Shows the relationship the model learned —
    e.g. 'high VIX values push predictions negative'
    """
    ticker   = shap_out["ticker"]
    features = shap_out["features"]

    if feature not in features:
        return go.Figure()

    feat_idx   = features.index(feature)
    shap_vals  = shap_out["shap_values"][:, feat_idx]
    feat_vals  = shap_out["X_raw"][feature].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=feat_vals,
        y=shap_vals,
        mode="markers",
        marker=dict(
            color=shap_vals,
            colorscale="RdBu",
            size=4,
            opacity=0.6,
            colorbar=dict(title="SHAP value"),
        ),
        name=feature,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
    fig.update_layout(
        title=f"SHAP dependence — {feature} vs its contribution ({ticker})",
        xaxis_title=f"{feature} value",
        yaxis_title="SHAP contribution",
        height=400,
    )
    return fig

def run_shap_analysis(
    df: pd.DataFrame,
    tickers: list[str] = None,
) -> None:
    if tickers is None:
        tickers = ["SPY", "QQQ", "NVDA", "XLF"]

    for ticker in tickers:
        model_path = f"artifacts/lgbm_{ticker}_model.pkl"
        if not Path(model_path).exists():
            print(f"Skipping {ticker} — no model found")
            continue

        shap_out = compute_shap_values(df, ticker)
        if not shap_out:
            continue

        # Feature importance chart
        fig1 = plot_feature_importance(shap_out)
        fig1.write_html(REPORT_DIR / f"shap_importance_{ticker}.html")

        # Top 3 features — SHAP over time
        top_features = shap_out["importance"]["feature"].head(3).tolist()
        for feat in top_features:
            fig2 = plot_shap_over_time(shap_out, feat)
            fig2.write_html(
                REPORT_DIR / f"shap_time_{ticker}_{feat}.html"
            )

            fig3 = plot_shap_vs_feature_value(shap_out, feat)
            fig3.write_html(
                REPORT_DIR / f"shap_dependence_{ticker}_{feat}.html"
            )

        print(f"SHAP charts saved for {ticker}")
        print(f"Top feature: {top_features[0]}\n")

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/full_features_v2.parquet")
    run_shap_analysis(df)