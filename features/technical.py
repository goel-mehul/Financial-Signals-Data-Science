import pandas as pd
import numpy as np
from pathlib import Path

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index — ranges 0 to 100.
    Above 70 = overbought (potential sell signal).
    Below 30 = oversold (potential buy signal).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence.
    macd line crossing above signal = bullish momentum.
    macd line crossing below signal = bearish momentum.
    """
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd":       macd_line,
        "macd_signal": signal_line,
        "macd_hist":  macd_line - signal_line,
    })

def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands — price relative to its own volatility.
    bb_pct = 0 means price is at lower band (cheap relative to recent history).
    bb_pct = 1 means price is at upper band (expensive relative to recent history).
    """
    mid   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (series - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid":   mid,
        "bb_lower": lower,
        "bb_pct":   pct_b,
    })

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Average True Range — measures daily price volatility.
    High ATR = big daily swings, low ATR = quiet market.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def rolling_zscore(series: pd.Series, window: int = 63) -> pd.Series:
    """
    How many standard deviations above or below the rolling mean.
    Useful for mean-reversion signals — extreme values tend to revert.
    63 days = approx 1 trading quarter.
    """
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)

def momentum(series: pd.Series, periods: list[int] = None) -> pd.DataFrame:
    """
    Price momentum at multiple lookback windows.
    Positive = price higher than N days ago (uptrend).
    Negative = price lower than N days ago (downtrend).
    """
    if periods is None:
        periods = [5, 10, 21, 63]
    result = {}
    for p in periods:
        result[f"mom_{p}d"] = series.pct_change(p)
    return pd.DataFrame(result)

def volume_features(
    close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """
    Volume-based features — high volume on up days = bullish conviction.
    """
    dollar_vol    = close * volume
    vol_ma_20     = volume.rolling(20).mean()
    vol_ratio     = volume / vol_ma_20.replace(0, np.nan)
    return pd.DataFrame({
        "dollar_vol":  dollar_vol,
        "vol_ratio":   vol_ratio,   # today's volume vs 20d average
    })

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all features to every ticker in the dataset.
    Returns the full dataframe with feature columns added.
    """
    out = []
    tickers = df["ticker"].unique()
    print(f"Computing features for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        grp = df[df["ticker"] == ticker].sort_values("date").copy()

        # Log return
        grp["log_return"] = np.log(grp["close"] / grp["close"].shift(1))

        # Momentum features
        mom_df = momentum(grp["close"])
        grp    = pd.concat([grp, mom_df], axis=1)

        # RSI
        grp["rsi_14"] = rsi(grp["close"], window=14)

        # ATR
        grp["atr_14"] = atr(grp["high"], grp["low"], grp["close"], window=14)

        # Rolling z-score
        grp["zscore_63"] = rolling_zscore(grp["close"], window=63)

        # Rolling volatility
        grp["vol_5d"]  = grp["log_return"].rolling(5).std()
        grp["vol_21d"] = grp["log_return"].rolling(21).std()
        grp["vol_63d"] = grp["log_return"].rolling(63).std()

        # MACD
        macd_df = macd(grp["close"])
        grp     = pd.concat([grp, macd_df], axis=1)

        # Bollinger Bands
        bb_df = bollinger_bands(grp["close"])
        grp   = pd.concat([grp, bb_df], axis=1)

        # Volume features
        vol_df = volume_features(grp["close"], grp["volume"])
        grp    = pd.concat([grp, vol_df], axis=1)

        # Target variable — 5-day forward return
        # Shift -5 so today's row contains what happens in the next 5 days
        grp["fwd_5d_return"] = grp["close"].pct_change(5).shift(-5)

        out.append(grp)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(tickers)} tickers...")

    result = pd.concat(out, ignore_index=True)
    print(f"\nFeature matrix shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")
    return result

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/ohlcv_clean.parquet")
    df_feat = add_all_features(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet("data/processed/ohlcv_features.parquet", index=False)
    print(f"\nSaved to data/processed/ohlcv_features.parquet")

    # Quick sanity check
    print("\nNull counts for key features (SPY only):")
    spy = df_feat[df_feat["ticker"] == "SPY"]
    feat_cols = ["rsi_14", "macd", "bb_pct", "vol_21d", "mom_21d", "zscore_63"]
    print(spy[feat_cols].isnull().sum())