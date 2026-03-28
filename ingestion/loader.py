import duckdb
from pathlib import Path

DB_PATH = "finsignal.duckdb"
RAW_OHLCV = Path("data/raw/ohlcv")
RAW_MACRO = Path("data/raw/macro/macro_indicators.parquet")

def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            date    DATE NOT NULL,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  BIGINT,
            ticker  VARCHAR NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS macro (
            date         DATE NOT NULL,
            CPI          DOUBLE,
            FedFunds     DOUBLE,
            Unemployment DOUBLE,
            Yield10yr    DOUBLE,
            M2           DOUBLE,
            VIX          DOUBLE
        )
    """)

def load_ohlcv(con: duckdb.DuckDBPyConnection) -> None:
    pattern = str(RAW_OHLCV / "*.parquet")
    con.execute("DELETE FROM ohlcv")
    con.execute(f"""
        INSERT INTO ohlcv
        SELECT date, open, high, low, close, volume, ticker
        FROM read_parquet('{pattern}')
    """)
    count = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    print(f"ohlcv rows loaded: {count:,}")

def load_macro(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DELETE FROM macro")
    con.execute(f"""
        INSERT INTO macro
        SELECT * FROM read_parquet('{str(RAW_MACRO)}')
    """)
    count = con.execute("SELECT COUNT(*) FROM macro").fetchone()[0]
    print(f"macro rows loaded: {count:,}")

def build_db() -> None:
    con = duckdb.connect(DB_PATH)
    create_schema(con)
    load_ohlcv(con)
    load_macro(con)
    con.close()
    print(f"Database written to {DB_PATH}")

if __name__ == "__main__":
    build_db()