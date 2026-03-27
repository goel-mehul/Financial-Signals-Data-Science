import pandas as pd
import numpy as np
from pathlib import Path

MACRO_COLS = ["CPI", "FedFunds", "Unemployment", "Yield10yr", "M2", "VIX"]

def merge_macro_features(
    ohlcv: pd.DataFrame,
    macro: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Merge macro indicators into per-ticker dataset.

    Steps:
    1. Resample macro to daily frequency (forward fill within month)
       — macro data is monthly, stock data is daily
    2. Lag by `lag` periods to prevent lookahead bias
       — today's row uses yesterday's macro data
    3. Merge on date

    Why lag=1?
    CPI for January is released in mid-February.
    If we use January CPI on January dates, we're pretending
    a trader in January had data they couldn't have had.
    Lagging ensures we only use already-published data.
    """
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date")

    # Forward fill within original index first, then resample to daily
    macro_filled = macro[MACRO_COLS].ffill()
    # Extend index by 5 extra business days to cover today's stock dates
    extended_idx = pd.bdate_range(
        start=macro_filled.index.min(),
        end=macro_filled.index.max() + pd.offsets.BDay(5),
    )
    macro_daily = macro_filled.reindex(extended_idx).ffill()

    # Lag by 1 day to prevent lookahead bias
    macro_lagged = macro_daily.shift(lag)
    macro_lagged.columns = [f"{c}_lag{lag}" for c in macro_lagged.columns]
    macro_lagged = macro_lagged.reset_index()
    macro_lagged = macro_lagged.rename(columns={macro_lagged.columns[0]: "date"})
    macro_lagged["date"] = pd.to_datetime(macro_lagged["date"])

    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    merged = ohlcv.merge(macro_lagged, on="date", how="left")

    # Check merge quality
    macro_lag_cols = [f"{c}_lag{lag}" for c in MACRO_COLS]
    null_pct = merged[macro_lag_cols].isnull().mean().mean() * 100
    print(f"Macro merge null rate: {null_pct:.2f}%")
    print(f"Merged shape: {merged.shape}")

    return merged

def add_macro_derived_features(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Add derived macro features on top of the raw lagged values.
    These capture rate of change rather than level — often more predictive.
    """
    # Month-over-month change in CPI = inflation acceleration
    df["CPI_mom_change"] = df.groupby("ticker")[f"CPI_lag{lag}"].transform(
        lambda x: x.pct_change(1)
    )

    # Yield curve proxy — how far rates have moved recently
    df["yield_21d_change"] = df.groupby("ticker")[f"Yield10yr_lag{lag}"].transform(
        lambda x: x.diff(21)
    )

    # VIX regime — is the market in high or low fear mode
    df["vix_high_regime"] = (df[f"VIX_lag{lag}"] > 25).astype(int)

    return df

if __name__ == "__main__":
    ohlcv = pd.read_parquet("data/processed/ohlcv_features.parquet")
    macro = pd.read_parquet("data/raw/macro/macro_indicators.parquet")

    print(f"OHLCV shape before merge: {ohlcv.shape}")

    merged = merge_macro_features(ohlcv, macro, lag=1)
    merged = add_macro_derived_features(merged, lag=1)

    print(f"Final shape after merge:  {merged.shape}")
    print(f"\nAll columns:")
    print(merged.columns.tolist())

    # Sanity check — show one row for SPY
    spy_row = merged[merged["ticker"] == "SPY"].iloc[-1]
    print(f"\nLatest SPY row (key features):")
    key_cols = [
        "date", "close", "rsi_14", "vol_21d",
        "CPI_lag1", "FedFunds_lag1", "VIX_lag1", "Yield10yr_lag1",
        "vix_high_regime", "fwd_5d_return",
    ]
    available = [c for c in key_cols if c in spy_row.index]
    print(spy_row[available].to_string())

    # Save
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    merged.to_parquet("data/processed/full_features.parquet", index=False)
    print(f"\nSaved to data/processed/full_features.parquet")