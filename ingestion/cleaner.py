import pandas as pd
import numpy as np
import duckdb
from pathlib import Path

DB_PATH = "finsignal.duckdb"

def clean_ohlcv() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    df = con.execute("""
        SELECT * FROM ohlcv
        ORDER BY ticker, date
    """).df()
    con.close()

    initial_rows = len(df)
    print(f"Rows before cleaning: {initial_rows:,}")

    # Drop rows where close is null or zero
    df = df[df["close"].notna() & (df["close"] > 0)]

    # Forward fill gaps within each ticker
    df = (
        df.sort_values(["ticker", "date"])
          .groupby("ticker", group_keys=False)
          .apply(lambda g: g.ffill())
    )

    # Drop tickers with fewer than 252 trading days
    counts = df.groupby("ticker").size()
    thin = counts[counts < 252].index.tolist()
    if thin:
        print(f"Dropping thin tickers: {thin}")
        df = df[~df["ticker"].isin(thin)]

    print(f"Rows after cleaning:  {len(df):,}")
    print(f"Tickers remaining:    {df['ticker'].nunique()}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/processed/ohlcv_clean.parquet", index=False)
    print("Saved to data/processed/ohlcv_clean.parquet")
    return df

if __name__ == "__main__":
    clean_ohlcv()