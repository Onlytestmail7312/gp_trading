"""
dataset_builder_v8.py
=====================
Load and split data by regime for V8 GP evolution.
Each regime gets its own 80/20 chronological train/val split.
This ensures every regime has validation data regardless of calendar year.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

log = logging.getLogger('gp')

from config_v8 import (
    PARQUET_FILE, ALL_SYMBOLS, DAILY_FEATURES,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START,
    REGIME_COLUMN, REGIMES, REGIME_MIN_DAYS,
)


# =============================================================================
# LOAD
# =============================================================================
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(str(path))
    log.info(f"[DATA] Loaded {len(df):,} rows from {path}")
    return df


# =============================================================================
# CALENDAR SPLIT (for test set only)
# =============================================================================
def split_dataset(df: pd.DataFrame):
    """Standard calendar split for test set evaluation."""
    if df.index.tz is not None:
        t_end   = pd.Timestamp(TRAIN_END,  tz=df.index.tz)
        v_start = pd.Timestamp(VAL_START,  tz=df.index.tz)
        v_end   = pd.Timestamp(VAL_END,    tz=df.index.tz)
        t_start = pd.Timestamp(TEST_START, tz=df.index.tz)
    else:
        t_end   = pd.Timestamp(TRAIN_END)
        v_start = pd.Timestamp(VAL_START)
        v_end   = pd.Timestamp(VAL_END)
        t_start = pd.Timestamp(TEST_START)

    train = df[df.index <= t_end].copy()
    val   = df[(df.index >= v_start) & (df.index <= v_end)].copy()
    test  = df[df.index >= t_start].copy()

    log.info(f"Train: {len(train):,} rows  "
             f"({train.index.min().date()} to {train.index.max().date()})")
    log.info(f"Val  : {len(val):,} rows  "
             f"({val.index.min().date()} to {val.index.max().date()})")
    log.info(f"Test : {len(test):,} rows  "
             f"({test.index.min().date()} to {test.index.max().date()})")
    return train, val, test


# =============================================================================
# REGIME FILTER
# =============================================================================
def filter_by_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Filter dataframe to only rows matching regime."""
    if REGIME_COLUMN not in df.columns:
        raise ValueError(f"Column '{REGIME_COLUMN}' not in dataset")
    return df[df[REGIME_COLUMN] == regime].copy()


# =============================================================================
# STOCK DATA PREPARATION
# =============================================================================
def prepare_stock_data(
    df: pd.DataFrame,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Prepare per-stock feature/price arrays from dataframe."""
    if symbols is None:
        symbols = list(ALL_SYMBOLS)

    result = {}
    for sym in symbols:
        sym_df = df[df['symbol'].astype(str) == str(sym)].sort_index()
        if len(sym_df) < 20:
            log.warning(f"  {sym}: only {len(sym_df)} rows -- skipping")
            continue

        features = sym_df[DAILY_FEATURES].values.astype(np.float64)
        prices   = sym_df['close'].values.astype(np.float64)
        opens    = sym_df['open'].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        result[sym] = (features, prices, opens)
        log.info(f"    {sym}: {len(sym_df)} bars")

    return result


# =============================================================================
# REGIME DATA -- 80/20 AUTOMATIC SPLIT
# =============================================================================
def prepare_regime_data(
    df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Prepare train + val data for each regime.

    Uses 80/20 chronological split per regime automatically.
    This ensures every regime has validation data regardless
    of which calendar years the regime appears in.

    Input:
        df = full dataset (train + val combined)

    Returns:
        {
            'BULL_QUIET': {
                'train':        {sym: (features, prices, opens)},
                'val':          {sym: (features, prices, opens)},
                'n_train_days': int,
                'n_val_days':   int,
                'train_start':  date,
                'train_end':    date,
                'val_start':    date,
                'val_end':      date,
            },
            ...
        }
    """
    regime_data = {}

    for regime in REGIMES:
        log.info(f"\n  [{regime}]")

        # Filter to this regime only
        regime_df    = filter_by_regime(df, regime)
        unique_dates = sorted(regime_df.index.unique())
        n_total      = len(unique_dates)

        if n_total < REGIME_MIN_DAYS:
            log.warning(f"  {regime}: only {n_total} days "
                        f"(min={REGIME_MIN_DAYS}) -- skipping")
            continue

        # 80/20 chronological split
        n_train     = int(n_total * 0.80)
        n_val       = n_total - n_train
        train_dates = set(unique_dates[:n_train])
        val_dates   = set(unique_dates[n_train:])

        train_regime = regime_df[regime_df.index.isin(train_dates)]
        val_regime   = regime_df[regime_df.index.isin(val_dates)]

        log.info(f"  Total : {n_total} days")
        log.info(f"  Train : {n_train} days  "
                 f"({min(train_dates).date()} to {max(train_dates).date()})")
        log.info(f"  Val   : {n_val} days  "
                 f"({min(val_dates).date()} to {max(val_dates).date()})")

        # Prepare stock arrays
        log.info(f"  Preparing train stocks...")
        train_stocks = prepare_stock_data(train_regime, ALL_SYMBOLS)

        log.info(f"  Preparing val stocks...")
        val_stocks   = prepare_stock_data(val_regime, ALL_SYMBOLS)

        log.info(f"  Stocks: train={len(train_stocks)}, "
                 f"val={len(val_stocks)}")

        regime_data[regime] = {
            'train':        train_stocks,
            'val':          val_stocks,
            'n_train_days': n_train,
            'n_val_days':   n_val,
            'train_start':  min(train_dates).date(),
            'train_end':    max(train_dates).date(),
            'val_start':    min(val_dates).date(),
            'val_end':      max(val_dates).date(),
        }

    return regime_data


# =============================================================================
# REGIME DISTRIBUTION
# =============================================================================
def get_regime_distribution(df: pd.DataFrame) -> pd.Series:
    if REGIME_COLUMN not in df.columns:
        return pd.Series()
    dates = df.groupby(df.index)[REGIME_COLUMN].first()
    return dates.value_counts()


# =============================================================================
# TEST
# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing dataset_builder_v8")
    print("=" * 60)

    # Load
    df = load_features(PARQUET_FILE)

    # Calendar split
    train_df, val_df, test_df = split_dataset(df)

    # Regime distribution
    print("\nRegime distribution in train+val:")
    combined = pd.concat([train_df, val_df])
    print(get_regime_distribution(combined).to_string())

    # Regime data with 80/20 split
    print("\nPreparing regime data (80/20 split)...")
    regime_data = prepare_regime_data(combined)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for regime, data in regime_data.items():
        print(f"\n  {regime}:")
        print(f"    Train: {data['n_train_days']} days  "
              f"({data['train_start']} to {data['train_end']})")
        print(f"    Val  : {data['n_val_days']} days  "
              f"({data['val_start']} to {data['val_end']})")
        print(f"    Stocks train: {list(data['train'].keys())}")
        print(f"    Stocks val  : {list(data['val'].keys())}")
