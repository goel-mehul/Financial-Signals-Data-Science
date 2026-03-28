import pandas as pd
import numpy as np
from pathlib import Path

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market regime and crisis features.
    These tell the model WHAT KIND of market environment it is in
    rather than just what prices are doing.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── VIX regime (4 levels) ─────────────────────────────
    # Binary flag isn't enough — markets behave differently
    # across the full spectrum of fear
    vix = df["VIX_lag1"]
    df["vix_regime_low"]      = (vix < 15).astype(int)   # complacency
    df["vix_regime_normal"]   = ((vix >= 15) & (vix < 20)).astype(int)
    df["vix_regime_elevated"] = ((vix >= 20) & (vix < 30)).astype(int)
    df["vix_regime_crisis"]   = (vix >= 30).astype(int)  # crisis
    df["vix_level"]           = vix  # raw level also useful

    # ── Yield curve slope ─────────────────────────────────
    # 10yr minus Fed Funds = term premium
    # Negative = inverted curve = recession signal
    # This was deeply inverted in 2022-2023
    df["yield_curve_slope"] = df["Yield10yr_lag1"] - df["FedFunds_lag1"]
    df["yield_curve_inverted"] = (df["yield_curve_slope"] < 0).astype(int)

    # ── Rate of change features ───────────────────────────
    # Levels are time proxies — changes are real signals
    # Replace M2 level with M2 momentum
    df["M2_mom_1m"]  = df.groupby("ticker")["M2_lag1"].transform(
        lambda x: x.pct_change(21)
    )
    df["M2_mom_3m"]  = df.groupby("ticker")["M2_lag1"].transform(
        lambda x: x.pct_change(63)
    )
    df["CPI_mom_1m"] = df.groupby("ticker")["CPI_lag1"].transform(
        lambda x: x.pct_change(21)
    )
    df["fed_funds_change_3m"] = df.groupby("ticker")["FedFunds_lag1"].transform(
        lambda x: x.diff(63)
    )
    df["yield_change_1m"] = df.groupby("ticker")["Yield10yr_lag1"].transform(
        lambda x: x.diff(21)
    )

    # ── Market drawdown ───────────────────────────────────
    # How far is each ticker from its 52-week high?
    # Deep drawdown = distress, near high = momentum
    def rolling_drawdown(grp):
        roll_max = grp["close"].rolling(252, min_periods=21).max()
        return (grp["close"] - roll_max) / roll_max

    df["drawdown_52w"] = df.groupby("ticker", group_keys=False).apply(
        rolling_drawdown
    )

    # ── Volatility regime ─────────────────────────────────
    # Is current vol higher or lower than its own history?
    # vol expanding = uncertainty increasing
    df["vol_regime"] = df.groupby("ticker")["vol_21d"].transform(
        lambda x: (x - x.rolling(252).mean()) / (x.rolling(252).std() + 1e-9)
    )
    df["vol_expanding"] = (df["vol_regime"] > 0.5).astype(int)

    # ── Rolling beta vs market ────────────────────────────
    # How much does this stock move relative to SPY?
    # High beta in crisis = amplified losses
    spy_returns = (
        df[df["ticker"] == "SPY"]
        .set_index("date")["log_return"]
        .rename("spy_return")
    )
    df = df.join(spy_returns, on="date")

    def rolling_beta(grp, window=63):
        cov = grp["log_return"].rolling(window).cov(grp["spy_return"])
        var = grp["spy_return"].rolling(window).var()
        return cov / (var + 1e-9)

    df["rolling_beta"] = df.groupby("ticker", group_keys=False).apply(
        rolling_beta
    )
    df = df.drop(columns=["spy_return"])

    # ── Relative strength vs market ───────────────────────
    # Is this stock outperforming or underperforming SPY?
    # Relative strength is one of the most robust signals in finance
    spy_mom = (
        df[df["ticker"] == "SPY"]
        .set_index("date")["mom_21d"]
        .rename("spy_mom_21d")
    )
    df = df.join(spy_mom, on="date")
    df["rel_strength_21d"] = df["mom_21d"] - df["spy_mom_21d"]
    df = df.drop(columns=["spy_mom_21d"])

    # ── Macro stress index ────────────────────────────────
    # Composite score combining VIX, yield curve, vol regime
    # High = stressed market environment
    vix_norm = (df["VIX_lag1"] - 15) / 15
    curve_stress = (-df["yield_curve_slope"]).clip(lower=0)
    vol_stress = df["vol_regime"].clip(lower=0)
    df["macro_stress"] = (
        0.5 * vix_norm +
        0.3 * curve_stress +
        0.2 * vol_stress
    ).clip(lower=0)

    print(f"Regime features added. New shape: {df.shape}")
    print(f"New columns added: {[c for c in df.columns if c not in ['date','open','high','low','close','volume','ticker']][-15:]}")

    return df

def fill_regime_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill nulls in regime features intelligently.
    - Rate of change features: fill with 0 (no change)
    - Regime flags: fill with 0 (unknown = neutral)
    - Continuous features: forward fill within ticker then fill with median
    """
    # Rate of change features — null means no data yet, treat as no change
    roc_cols = [
        "M2_mom_1m", "M2_mom_3m", "CPI_mom_1m",
        "fed_funds_change_3m", "yield_change_1m",
        "CPI_mom_change", "yield_21d_change",
    ]
    for col in roc_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Regime flags — fill with neutral (0)
    flag_cols = [
        "vix_regime_low", "vix_regime_normal",
        "vix_regime_elevated", "vix_regime_crisis",
        "yield_curve_inverted", "vol_expanding",
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Continuous regime features — forward fill within ticker then median
    continuous_cols = [
        "yield_curve_slope", "macro_stress",
        "drawdown_52w", "vol_regime",
        "rolling_beta", "rel_strength_21d",
    ]
    for col in continuous_cols:
        if col in df.columns:
            df[col] = df.groupby("ticker")[col].transform(
                lambda x: x.ffill().fillna(x.median())
            )

    return df

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/full_features_v2.parquet")
    df_regime = add_regime_features(df)
    df_regime = fill_regime_nulls(df_regime)

    Path("data/processed").mkdir(exist_ok=True)
    df_regime.to_parquet(
        "data/processed/full_features_v2.parquet",
        index=False,
    )
    print("\nSaved to data/processed/full_features_v2.parquet")

    # Quick sanity check on SPY
    spy = df_regime[df_regime["ticker"] == "SPY"].sort_values("date")
    print("\nSPY regime snapshot (last 5 rows):")
    regime_cols = [
        "date", "close", "vix_regime_crisis", "yield_curve_inverted",
        "drawdown_52w", "rolling_beta", "rel_strength_21d", "macro_stress"
    ]
    available = [c for c in regime_cols if c in spy.columns]
    print(spy[available].tail(5).to_string(index=False))

