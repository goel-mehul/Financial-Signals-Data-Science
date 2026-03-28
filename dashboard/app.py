import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FinSignal",
    page_icon="📈",
    layout="wide",
)

st.title("FinSignal — market intelligence dashboard")
st.caption("Time-series driven signal pipeline · XGBoost + LightGBM · Walk-forward validated")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/full_features_v2.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_comparison():
    path = Path("artifacts/model_comparison.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

@st.cache_resource
def load_model(ticker: str, model_name: str):
    path = Path(f"artifacts/{model_name}_{ticker}_model.pkl")
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

df         = load_data()
comparison = load_comparison()
tickers    = sorted(df["ticker"].unique())

# ── Sidebar ──────────────────────────────────────────────

st.sidebar.header("Controls")

# Group tickers by sector
sector_tickers = {
    "Broad market": ["SPY", "QQQ", "IWM"],
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "XLK"],
    "Healthcare": ["JNJ", "UNH", "XLV"],
    "Finance": ["JPM", "BAC", "GS", "XLF"],
    "Consumer": ["AMZN", "COST", "WMT", "XLP"],
    "Energy": ["XOM", "CVX", "XLE"],
    "Other sectors": ["XLI", "XLU", "XLB"]
}
all_tickers = [t for group in sector_tickers.values() for t in group]
available_tickers = [t for t in all_tickers if t in tickers]

sector = st.sidebar.selectbox("Sector", list(sector_tickers.keys()))
sector_list = [t for t in sector_tickers[sector] if t in tickers]
ticker = st.sidebar.selectbox("Ticker", sector_list if sector_list else available_tickers)

start_date = st.sidebar.date_input(
    "From",
    value=pd.Timestamp("2022-01-01"),
)
end_date = st.sidebar.date_input(
    "To",
    value=pd.Timestamp.today(),
)
st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.caption(
    "FinSignal is an end-to-end financial DS pipeline. "
    "Models trained on 5yr OHLCV + FRED macro data using "
    "walk-forward validation."
)

# ── Filter data ───────────────────────────────────────────
sub = df[
    (df["ticker"] == ticker) &
    (df["date"] >= pd.Timestamp(start_date)) &
    (df["date"] <= pd.Timestamp(end_date))
].sort_values("date")

if sub.empty:
    st.warning("No data for this selection.")
    st.stop()

# ── Metrics row ───────────────────────────────────────────
log_ret    = np.log(sub["close"] / sub["close"].shift(1)).dropna()
period_ret = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100
ann_vol    = log_ret.std() * np.sqrt(252) * 100
rsi_now    = sub["rsi_14"].iloc[-1]
vix_now    = sub["VIX_lag1"].iloc[-1] if "VIX_lag1" in sub.columns else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest close",    f"${sub['close'].iloc[-1]:.2f}")
m2.metric("Period return",   f"{period_ret:.1f}%")
m3.metric("Ann. volatility", f"{ann_vol:.1f}%")
m4.metric("RSI (14)",        f"{rsi_now:.1f}")

st.markdown("---")

# ── Price chart ───────────────────────────────────────────
st.subheader("Price & volatility")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "price + Bollinger Bands",
        "21-day rolling volatility",
        "RSI (14)",
    ],
    row_heights=[0.55, 0.25, 0.20],
    vertical_spacing=0.06,
)

# Price
fig.add_trace(go.Scatter(
    x=sub["date"], y=sub["close"],
    name="close",
    line=dict(color="#4fc3f7", width=1.5),
), row=1, col=1)

# Bollinger Bands
if "bb_upper" in sub.columns:
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["bb_upper"],
        name="BB upper",
        line=dict(color="#555", dash="dot", width=0.8),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["bb_lower"],
        name="BB lower",
        line=dict(color="#555", dash="dot", width=0.8),
        fill="tonexty",
        fillcolor="rgba(100,100,100,0.08)",
        showlegend=False,
    ), row=1, col=1)

# Volatility
fig.add_trace(go.Scatter(
    x=sub["date"], y=sub["vol_21d"],
    name="vol 21d",
    line=dict(color="#e57373", width=1.2),
), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=sub["date"], y=sub["rsi_14"],
    name="RSI",
    line=dict(color="#ffb74d", width=1.2),
), row=3, col=1)
fig.add_hline(y=70, line_dash="dot",
              line_color="#e57373", line_width=0.8, row=3, col=1)
fig.add_hline(y=30, line_dash="dot",
              line_color="#81c784", line_width=0.8, row=3, col=1)

fig.update_layout(
    height=600,
    showlegend=True,
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# ── Model predictions ─────────────────────────────────────
st.markdown("---")
st.subheader("Model signal")

model_options = {
    "LightGBM": f"lgbm_{ticker}_preds.parquet",
    "XGBoost": f"xgboost_{ticker}_preds.parquet",
    "Regime LightGBM": f"regime_lgbm_{ticker}_preds.parquet",
}

col_m1, col_m2 = st.columns([2, 1])
with col_m1:
    selected_model = st.selectbox(
        "Model",
        list(model_options.keys()),
    )

pred_path = Path(f"artifacts/{model_options[selected_model]}")
if pred_path.exists():
    preds = pd.read_parquet(pred_path)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[
        (preds["date"] >= pd.Timestamp(start_date)) &
        (preds["date"] <= pd.Timestamp(end_date))
    ]

    if not preds.empty:
        strategy_ret = np.sign(preds["predicted"].values) * preds["actual"].values
        bh_ret       = preds["actual"].values
        cum_strategy = np.cumsum(strategy_ret)
        cum_bh       = np.cumsum(bh_ret)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=preds["date"], y=cum_strategy,
            name=f"{selected_model} strategy",
            line=dict(color="#ffb74d", width=1.5),
        ))
        fig2.add_trace(go.Scatter(
            x=preds["date"], y=cum_bh,
            name="buy & hold",
            line=dict(color="#4fc3f7", width=1.5, dash="dot"),
        ))
        fig2.add_hline(y=0, line_dash="dash",
                       line_color="gray", line_width=0.5)
        fig2.update_layout(
            title=f"Cumulative returns — {selected_model} vs buy & hold",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        from scipy.stats import spearmanr
        ic, _    = spearmanr(preds["predicted"], preds["actual"])
        hit_rate = np.mean(
            np.sign(preds["predicted"]) == np.sign(preds["actual"])
        )
        sharpe = (
            strategy_ret.mean() / (strategy_ret.std() + 1e-9)
        ) * np.sqrt(252 / 5)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe",   f"{sharpe:.3f}")
        c2.metric("IC",       f"{ic:.3f}")
        c3.metric("Hit rate", f"{hit_rate:.1%}")
else:
    st.info(f"No predictions found for {selected_model} on {ticker}.")
    
# ── Model comparison table ────────────────────────────────
if not comparison.empty:
    st.markdown("---")
    st.subheader("Model comparison — all tickers")
    ticker_comp = comparison[comparison["ticker"] == ticker]
    if not ticker_comp.empty:
        st.dataframe(
            ticker_comp.style.format({
                "sharpe":   "{:.3f}",
                "ic":       "{:.3f}",
                "hit_rate": "{:.1%}",
                "max_dd":   "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ── Feature snapshot ──────────────────────────────────────
st.markdown("---")
st.subheader("Latest feature snapshot")
display_cols = [
    "date", "close", "rsi_14", "macd", "bb_pct",
    "vol_21d", "mom_21d", "VIX_lag1", "Yield10yr_lag1",
]
available = [c for c in display_cols if c in sub.columns]
st.dataframe(
    sub[available].tail(20).reset_index(drop=True),
    use_container_width=True,
)