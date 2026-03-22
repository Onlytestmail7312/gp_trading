"""
Dataset builder: loads raw data, computes features for all stocks,
joins NIFTY context, and creates train/val/test splits.
"""

import pandas as pd
import duckdb
from typing import Dict, Tuple, Optional

from gp_system_complete.config import (
    DB_PATH,
    TRADABLE_STOCKS,
    MARKET_CONTEXT,
    TRAIN_END,
    VAL_END,
    FEATURE_TABLE,
    FEATURE_PARQUET,
    V1_GP_FEATURES,
    OUTPUT_DIR,
)
from gp_system_complete.data_loader import load_symbol, load_all_symbols, validate_data
from gp_system_complete.feature_engineering import (
    compute_nifty_features,
    build_features_for_stock,
)
from gp_system_complete.utils import get_logger, ensure_output_dirs, Timer, print_banner


def build_full_feature_dataset(
    db_path: str = DB_PATH,
    symbols: Optional[list] = None,
    save_parquet: bool = True,
    save_duckdb: bool = True,
) -> pd.DataFrame:
    """
    Build the complete feature dataset for all tradable stocks.

    Steps:
    1. Load NIFTY data → compute NIFTY features
    2. For each tradable stock → compute all features
    3. Concatenate into one DataFrame
    4. Save to parquet and/or DuckDB

    Returns the combined DataFrame.
    """
    log = get_logger()
    ensure_output_dirs()

    if symbols is None:
        symbols = TRADABLE_STOCKS

    print_banner("BUILDING FEATURE DATASET")

    # Step 1: Load and process NIFTY
    with Timer("NIFTY feature computation"):
        nifty_df = load_symbol(MARKET_CONTEXT, db_path)
        if nifty_df.empty:
            raise ValueError("No NIFTY data found in database")
        validate_data(nifty_df, MARKET_CONTEXT)
        nifty_features = compute_nifty_features(nifty_df)
        log.info(f"NIFTY features: {len(nifty_features):,} rows, "
                 f"{len(nifty_features.columns)} columns")

    # Step 2: Process each tradable stock
    all_frames = []
    for sym in symbols:
        with Timer(f"{sym} feature computation"):
            stock_df = load_symbol(sym, db_path)
            if stock_df.empty:
                log.warning(f"Skipping {sym}: no data")
                continue
            validate_data(stock_df, sym)
            feat_df = build_features_for_stock(stock_df, nifty_features, sym)
            all_frames.append(feat_df)

    if not all_frames:
        raise ValueError("No stock data processed")

    # Step 3: Concatenate
    combined = pd.concat(all_frames, axis=0).sort_index()
    log.info(f"Combined dataset: {len(combined):,} rows, {len(combined.columns)} columns")

    # Show per-symbol summary
    for sym in combined["symbol"].unique():
        mask = combined["symbol"] == sym
        n = mask.sum()
        log.info(f"  {sym}: {n:,} rows")

    # Step 4: Save
    if save_parquet:
        combined.to_parquet(str(FEATURE_PARQUET), index=True)
        log.info(f"Saved parquet → {FEATURE_PARQUET}")

    if save_duckdb:
        _save_to_duckdb(combined, db_path)

    return combined


def _save_to_duckdb(df: pd.DataFrame, db_path: str) -> None:
    """Save the feature DataFrame to DuckDB."""
    log = get_logger()
    # Reset index so 'ts' becomes a column
    df_save = df.reset_index()
    df_save = df_save.rename(columns={df_save.columns[0]: "ts"})

    con = duckdb.connect(db_path)
    con.execute(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")
    con.execute(f"CREATE TABLE {FEATURE_TABLE} AS SELECT * FROM df_save")
    count = con.execute(f"SELECT COUNT(*) FROM {FEATURE_TABLE}").fetchone()[0]
    con.close()
    log.info(f"Saved DuckDB table '{FEATURE_TABLE}': {count:,} rows")


def load_feature_dataset(
    db_path: str = DB_PATH,
    from_parquet: bool = True,
) -> pd.DataFrame:
    """
    Load the precomputed feature dataset.
    """
    log = get_logger()

    if from_parquet and FEATURE_PARQUET.exists():
        log.info(f"Loading features from {FEATURE_PARQUET}")
        df = pd.read_parquet(str(FEATURE_PARQUET))
        log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df

    # Fallback to DuckDB
    log.info(f"Loading features from DuckDB table '{FEATURE_TABLE}'")
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {FEATURE_TABLE} ORDER BY ts, symbol").df()
    con.close()

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def split_by_time(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split feature dataset into train/validation/test by date.

    Train: everything up to train_end
    Validation: train_end < ts <= val_end
    Test: everything after val_end
    """
    log = get_logger()

    # Handle timezone-aware index
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        train_end_ts = pd.Timestamp(train_end, tz=idx.tz)
        val_end_ts = pd.Timestamp(val_end, tz=idx.tz)
    else:
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)

    train = df[df.index <= train_end_ts].copy()
    val = df[(df.index > train_end_ts) & (df.index <= val_end_ts)].copy()
    test = df[df.index > val_end_ts].copy()

    log.info(f"Time-based split:")
    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        if len(split_df) > 0:
            symbols = split_df["symbol"].nunique()
            log.info(
                f"  {name:5s}: {len(split_df):>10,} rows, "
                f"{symbols} symbols, "
                f"{split_df.index[0].date()} → {split_df.index[-1].date()}"
            )
        else:
            log.warning(f"  {name:5s}: EMPTY")

    return train, val, test


def get_stock_data(
    df: pd.DataFrame,
    symbol: str,
    feature_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Extract data for a single stock from the combined dataset.
    """
    if feature_cols is None:
        feature_cols = V1_GP_FEATURES

    mask = df["symbol"] == symbol
    stock_df = df.loc[mask].copy()

    # Verify all required feature columns exist
    missing = [c for c in feature_cols if c not in stock_df.columns]
    if missing:
        log = get_logger()
        log.warning(f"{symbol}: missing features: {missing}")

    return stock_df