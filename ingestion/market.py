import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/ohlcv")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "UNH", "V",
    "XOM", "LLY", "JNJ", "MA", "PG", "AVGO", "HD", "MRK", "CVX", "ABBV",
    "COST", "PEP", "ADBE", "WMT", "BAC", "KO", "CRM", "MCD", "ACN", "NFLX",
    "TMO", "CSCO", "ABT", "LIN", "DHR", "AMD", "INTC", "NEE", "PFE", "TXN",
    "WFC", "PM", "RTX", "INTU", "AMGN", "SPGI", "HON", "CAT", "IBM", "GE",
    "QCOM", "GS", "MS", "BLK", "ISRG", "SYK", "ELV", "LOW", "AXP", "DE",
    "MDLZ", "BKNG", "ADI", "VRTX", "REGN", "NOW", "PLD", "CI", "CB", "BRK-B",
    "SPY", "QQQ", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB"
]

def fetch_ticker(ticker: str, period: str = "5y") -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty or len(df) < 252:
            log.warning(f"Skipping {ticker}: only {len(df)} rows")
            return None
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        return df.reset_index().rename(columns={"index": "date"})
    except Exception as e:
        log.error(f"Failed {ticker}: {e}")
        return None

def ingest_all(period: str = "5y") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Fetching {len(TICKERS)} tickers...")
    for ticker in TICKERS:
        out = RAW_DIR / f"{ticker}.parquet"
        if out.exists():
            log.info(f"Already exists, skipping: {ticker}")
            continue
        df = fetch_ticker(ticker, period)
        if df is not None:
            df.to_parquet(out, index=False)
            log.info(f"Saved {ticker} ({len(df)} rows)")
        time.sleep(0.5)  # small delay to avoid rate limiting

def main():
    ingest_all()

if __name__ == "__main__":
    main()