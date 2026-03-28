import pandas as pd
import numpy as np
import pytest
from features.technical import (
    rsi,
    macd,
    bollinger_bands,
    atr,
    rolling_zscore,
    momentum,
)

@pytest.fixture
def price_series():
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(252) * 0.5)
    return pd.Series(prices, name="close")

@pytest.fixture
def ohlcv_series(price_series):
    high   = price_series * 1.01
    low    = price_series * 0.99
    volume = pd.Series(np.random.randint(1_000_000, 5_000_000, 252))
    return price_series, high, low, volume

def test_rsi_range(price_series):
    r = rsi(price_series, window=14).dropna()
    assert (r >= 0).all() and (r <= 100).all()

def test_rsi_length(price_series):
    r = rsi(price_series, window=14)
    assert len(r) == len(price_series)

def test_rsi_no_inf(price_series):
    r = rsi(price_series)
    assert not np.isinf(r).any()

def test_macd_columns(price_series):
    m = macd(price_series)
    assert set(m.columns) == {"macd", "macd_signal", "macd_hist"}

def test_macd_hist_is_diff(price_series):
    m = macd(price_series).dropna()
    diff = (m["macd"] - m["macd_signal"] - m["macd_hist"]).abs()
    assert diff.max() < 1e-8

def test_macd_length(price_series):
    m = macd(price_series)
    assert len(m) == len(price_series)

def test_bb_columns(price_series):
    bb = bollinger_bands(price_series)
    assert set(bb.columns) == {"bb_upper", "bb_mid", "bb_lower", "bb_pct"}

def test_bb_upper_above_lower(price_series):
    bb = bollinger_bands(price_series).dropna()
    assert (bb["bb_upper"] >= bb["bb_lower"]).all()

def test_bb_pct_mostly_bounded(price_series):
    bb   = bollinger_bands(price_series).dropna()
    frac = ((bb["bb_pct"] >= 0) & (bb["bb_pct"] <= 1)).mean()
    assert frac > 0.8

def test_atr_positive(ohlcv_series):
    close, high, low, _ = ohlcv_series
    a = atr(high, low, close).dropna()
    assert (a > 0).all()

def test_atr_length(ohlcv_series):
    close, high, low, _ = ohlcv_series
    a = atr(high, low, close)
    assert len(a) == len(close)

def test_zscore_mean_near_zero(price_series):
    z = rolling_zscore(price_series, window=63).dropna()
    assert abs(z.mean()) < 0.5

def test_zscore_no_inf(price_series):
    z = rolling_zscore(price_series)
    assert not np.isinf(z).any()

def test_momentum_columns(price_series):
    mom = momentum(price_series, periods=[5, 21, 63])
    assert set(mom.columns) == {"mom_5d", "mom_21d", "mom_63d"}

def test_momentum_length(price_series):
    mom = momentum(price_series)
    assert len(mom) == len(price_series)

def test_full_feature_pipeline():
    from features.technical import add_all_features
    np.random.seed(0)
    n      = 300
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date":   pd.date_range("2020-01-01", periods=n, freq="B"),
        "open":   prices * 0.999,
        "high":   prices * 1.01,
        "low":    prices * 0.99,
        "close":  prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
        "ticker": "TEST",
    })
    result = add_all_features(df)
    assert len(result) == n
    assert "rsi_14"        in result.columns
    assert "macd"          in result.columns
    assert "bb_pct"        in result.columns
    assert "vol_21d"       in result.columns
    assert "fwd_5d_return" in result.columns
    core      = ["rsi_14", "macd", "vol_21d", "bb_pct"]
    tail_null = result[core].tail(200).isnull().mean().mean()
    assert tail_null < 0.01