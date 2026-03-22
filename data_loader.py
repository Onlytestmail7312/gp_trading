"""
Data loading from DuckDB.

Loads 1-minute OHLCV bars for any symbol from the `min_1m` table.
Handles timezone-aware timestamps and basic validation.
"""

import pandas as pd
import duckdb
from typing import Optional, List

from gp_system_complete.config import DB_PATH, ALL_SYMBOLS, BARS_PER_DAY
from gp_system_complete.utils import get_logger


def load_symbol(
    symbol: str,
    db_path: str = DB_PATH,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load 1-minute bars for a single symbol.

    Returns DataFrame with columns:
        ts (datetime index), open, high, low, close, volume
    """
    log = get_logger()

    con = duckdb.connect(db_path, read_only=True)

    query = """
        SELECT ts, open, high, low, close, volume
        FROM   min_1m
        WHERE  symbol = ?
    """
    params = [symbol]

    if start_ts:
        query += " AND ts >= ?"
        params.append(start_ts)
    if end_ts:
        query += " AND ts <= ?"
        params.append(end_ts)

    query += " ORDER BY ts"

    df = con.execute(query, params).df()
    con.close()

    if df.empty:
        log.warning(f"No data loaded for {symbol}")
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    # Remove exact duplicate timestamps
    dupes = df.index.duplicated(keep="first")
    if dupes.any():
        n_dupes = dupes.sum()
        log.warning(f"{symbol}: dropped {n_dupes} duplicate timestamps")
        df = df[~dupes]

    log.info(
        f"Loaded {symbol}: {len(df):,} bars, "
        f"{df.index[0].date()} → {df.index[-1].date()}, "
        f"~{len(df) / BARS_PER_DAY:.0f} days"
    )
    return df


def load_all_symbols(
    symbols: Optional[List[str]] = None,
    db_path: str = DB_PATH,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> dict:
    """
    Load data for multiple symbols.

    Returns dict: {symbol: DataFrame}
    """
    if symbols is None:
        symbols = ALL_SYMBOLS

    data = {}
    for sym in symbols:
        df = load_symbol(sym, db_path, start_ts, end_ts)
        if not df.empty:
            data[sym] = df

    return data


def get_database_summary(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Return a summary of the database contents.
    """
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("""
        SELECT symbol,
               COUNT(*)  AS row_count,
               MIN(ts)   AS first_ts,
               MAX(ts)   AS last_ts
        FROM   min_1m
        GROUP  BY symbol
        ORDER  BY row_count DESC
    """).df()
    con.close()

    if not df.empty:
        df["days"] = df["row_count"] / BARS_PER_DAY
        df["years"] = df["days"] / 250.0
        df["first_ts"] = pd.to_datetime(df["first_ts"]).dt.date
        df["last_ts"] = pd.to_datetime(df["last_ts"]).dt.date

    return df


def validate_data(df: pd.DataFrame, symbol: str) -> dict:
    """
    Run data quality checks on a single symbol's DataFrame.
    Returns dict of check results.
    """
    checks = {}
    ohlc = ["open", "high", "low", "close"]

    checks["rows"] = len(df)
    checks["nan_pct"] = df[ohlc].isnull().sum().sum() / (len(df) * 4) * 100
    checks["zero_prices"] = (df[ohlc] == 0).sum().sum()
    checks["negative_prices"] = (df[ohlc] < 0).sum().sum()
    checks["high_lt_low"] = (df["high"] < df["low"]).sum()
    checks["duplicate_ts"] = df.index.duplicated().sum()

    log = get_logger()
    status = "✅" if all(
        checks[k] == 0 for k in ["zero_prices", "negative_prices", "high_lt_low"]
    ) else "⚠️"
    log.info(
        f"{status} {symbol}: {checks['rows']:,} rows, "
        f"NaN={checks['nan_pct']:.2f}%, "
        f"zeros={checks['zero_prices']}, "
        f"neg={checks['negative_prices']}, "
        f"H<L={checks['high_lt_low']}"
    )
    return checks