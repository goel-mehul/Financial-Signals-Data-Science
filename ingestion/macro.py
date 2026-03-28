from fredapi import Fred
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

FRED_DIR = Path("data/raw/macro")

SERIES = {
    "CPI":          "CPIAUCSL",
    "FedFunds":     "FEDFUNDS",
    "Unemployment": "UNRATE",
    "Yield10yr":    "GS10",
    "M2":           "M2SL",
    "VIX":          "VIXCLS",
}

def ingest_macro() -> None:
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    frames = {}
    for name, series_id in SERIES.items():
        print(f"Fetching {name} ({series_id})...")
        s = fred.get_series(series_id, observation_start="2018-01-01")
        frames[name] = s.rename(name)
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.reset_index()
    df.to_parquet(FRED_DIR / "macro_indicators.parquet", index=False)
    print(f"Macro data saved: {df.shape}")
    print(df.tail())

if __name__ == "__main__":
    ingest_macro()