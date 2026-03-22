"""
Main Feature Build Script — compute all features and save to parquet/DuckDB.

Usage:
    python -m gp_system_complete.main_build_features

Workflow:
1. Connect to gp_research.duckdb
2. Load raw OHLCV data for all symbols
3. Compute normalised features (returns, trend, vol, volume, time, NIFTY context)
4. Merge NIFTY context features and relative strength
5. Save combined feature dataset to parquet + DuckDB table
"""

import sys
import time
from datetime import datetime

import pandas as pd

from .config import (
    DB_PATH, OUTPUT_DIR, FEATURE_PARQUET, FEATURE_TABLE,
    TRADABLE_STOCKS, MARKET_CONTEXT, ALL_SYMBOLS,
    V1_GP_FEATURES,
)
from .data_loader import load_all_symbols, get_database_summary, validate_data
from .dataset_builder import build_full_feature_dataset, load_feature_dataset, split_by_time
from .regime_analysis import add_regime_column, print_regime_distribution
from .utils import get_logger, print_banner, ensure_output_dirs, Timer

logger = get_logger("main_build_features")


def main():
    """Run the full feature engineering pipeline."""

    ensure_output_dirs()
    print_banner("FEATURE ENGINEERING PIPELINE")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Database: {DB_PATH}")
    logger.info(f"  Output: {OUTPUT_DIR}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Database summary
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n📂 Checking database...")

    try:
        summary = get_database_summary()
    except Exception as exc:
        logger.error(f"❌ Cannot connect to database: {exc}")
        logger.error(f"   Make sure '{DB_PATH}' exists in the working directory.")
        sys.exit(1)

    logger.info(f"  Found {len(summary)} symbols:")
    for _, row in summary.iterrows():
        sym = row['symbol']
        count = row['row_count']
        min_date = row['first_ts']
        max_date = row['last_ts']
        logger.info(
            f"    {sym:>12}: {count:>10,} rows  "
            f"({min_date} → {max_date})"
        )

    # Verify required symbols exist
    missing = [s for s in ALL_SYMBOLS if s not in summary['symbol'].values]
    if missing:
        logger.warning(f"  ⚠️  Missing symbols: {missing}")
        if MARKET_CONTEXT in missing:
            logger.error(f"  ❌ Market context '{MARKET_CONTEXT}' is required!")
            sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Build features
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n🔧 Building features for all stocks...")

    with Timer("Feature engineering"):
        df = build_full_feature_dataset()

    if df is None or df.empty:
        logger.error("❌ Feature build failed — no data produced.")
        sys.exit(1)

    logger.info(f"\n  ✅ Feature dataset built:")
    logger.info(f"     Total rows: {len(df):,}")
    logger.info(f"     Columns: {len(df.columns)}")
    logger.info(f"     Symbols: {sorted(df['symbol'].unique())}")
    logger.info(f"     Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Verify features
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n🔍 Verifying feature columns...")

    missing_features = [f for f in V1_GP_FEATURES if f not in df.columns]
    if missing_features:
        logger.error(f"  ❌ Missing V1 features: {missing_features}")
        sys.exit(1)

    logger.info(f"  ✅ All {len(V1_GP_FEATURES)} V1 GP features present")

    # Feature statistics
    logger.info("\n  Feature statistics (mean / std / NaN%):")
    for feat in V1_GP_FEATURES:
        col = df[feat]
        nan_pct = col.isna().mean() * 100
        logger.info(
            f"    {feat:>25}: "
            f"mean={col.mean():>8.4f}  "
            f"std={col.std():>8.4f}  "
            f"NaN={nan_pct:>5.1f}%"
        )

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Time splits preview
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n📅 Time split preview:")

    train_df, val_df, test_df = split_by_time(df)
    logger.info(f"  Train: {len(train_df):>10,} rows")
    logger.info(f"  Val:   {len(val_df):>10,} rows")
    logger.info(f"  Test:  {len(test_df):>10,} rows")

    # Per-stock split sizes
    logger.info("\n  Per-stock split sizes:")
    for sym in TRADABLE_STOCKS:
        n_train = len(train_df[train_df["symbol"] == sym])
        n_val = len(val_df[val_df["symbol"] == sym])
        n_test = len(test_df[test_df["symbol"] == sym])
        logger.info(
            f"    {sym:>12}: "
            f"train={n_train:>8,}  "
            f"val={n_val:>8,}  "
            f"test={n_test:>8,}"
        )

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Regime distribution preview
    # ══════════════════════════════════════════════════════════════════
    if "trend_50d_return" in df.columns:
        logger.info("\n📈 Regime distribution:")
        df_with_regime = add_regime_column(df.copy())
        print_regime_distribution(df_with_regime, title="ALL DATA")

    # ══════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════
    print_banner("FEATURE BUILD COMPLETE")
    logger.info(f"  Parquet: {FEATURE_PARQUET}")
    logger.info(f"  DuckDB table: {FEATURE_TABLE}")
    logger.info(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()