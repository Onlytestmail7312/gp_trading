"""
dataset_builder.py -- build daily feature dataset from 1m DuckDB data.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import get_logger, print_banner, Timer
from data_loader import load_symbol, validate_data
from feature_engineering import build_daily_features
from config import (
    ALL_SYMBOLS, NIFTY_SYMBOL, DB_PATH, OUTPUT_DIR,
    TRAIN_END, VAL_START, VAL_END, TEST_START,
    DAILY_FEATURES,
)

log = get_logger()


def build_full_feature_dataset(
    symbols: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    if symbols is None:
        symbols = list(ALL_SYMBOLS)
    if db_path is None:
        db_path = DB_PATH

    print_banner("BUILDING DAILY FEATURE DATASET")

    log.info("Loading NIFTY (market index)...")
    try:
        nifty_1m = load_symbol(NIFTY_SYMBOL, db_path)
        validate_data(nifty_1m, NIFTY_SYMBOL)
    except Exception as e:
        log.warning(f"Could not load NIFTY: {e} -- market features will be zero")
        nifty_1m = None

    all_dfs = []

    for sym in symbols:
        log.info(f"Processing {sym}...")
        try:
            with Timer(f"{sym} feature computation"):
                raw = load_symbol(sym, db_path)
                validate_data(raw, sym)
                feat_df = build_daily_features(raw, nifty_1m, sym)
                all_dfs.append(feat_df)
        except Exception as e:
            log.error(f"Failed to process {sym}: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No data loaded for any symbol.")

    df = pd.concat(all_dfs, axis=0).sort_index()

    log.info(f"Combined daily dataset: {len(df):,} rows, {len(df.columns)} columns")
    log.info(f"  Symbols : {sorted(df['symbol'].unique())}")
    log.info(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df.index <  TRAIN_END].copy()
    val   = df[(df.index >= VAL_START) & (df.index < VAL_END)].copy()
    test  = df[df.index >= TEST_START].copy()

    log.info(f"Train: {len(train):,} rows  ({train.index.min().date()} to {train.index.max().date()})")
    log.info(f"Val  : {len(val):,} rows  ({val.index.min().date()} to {val.index.max().date()})")
    log.info(f"Test : {len(test):,} rows  ({test.index.min().date()} to {test.index.max().date()})")

    return train, val, test


def prepare_stock_data(
    df: pd.DataFrame,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if symbols is None:
        symbols = list(ALL_SYMBOLS)

    result = {}
    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_index()
        if len(sym_df) < 50:
            log.warning(f"  {sym}: only {len(sym_df)} rows -- skipping")
            continue
        features = sym_df[DAILY_FEATURES].values.astype(np.float64)
        prices   = sym_df["close"].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        result[sym] = (features, prices)
        log.info(f"  {sym}: {len(sym_df)} daily bars ready")

    return result


def save_features(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    if path is None:
        path = OUTPUT_DIR / "gp_features_daily.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path))
    size_mb = path.stat().st_size / 1e6
    log.info(f"[SAVE] Features saved to {path}  ({size_mb:.1f} MB)")
    return path


def load_features(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = OUTPUT_DIR / "gp_features_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_parquet(str(path))
    log.info(f"[DATA] Loaded {len(df):,} rows from {path}")
    return df
