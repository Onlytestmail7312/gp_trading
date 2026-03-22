"""
dataset_builder2.py
Build GP feature dataset from 10-year daily + hourly data.
Output: gp_output/gp_features_daily2.parquet
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
SNAPSHOT_DB  = r"gp_output\gp_snapshot2.duckdb"
OUTPUT_FILE  = r"gp_output\gp_features_daily2.parquet"

GP_STOCKS    = ["ICICIBANK", "HDFCBANK", "RELIANCE", "INFY", "TCS", "WIPRO"]
INDEX        = "NIFTY"

# New 10-year split
TRAIN_START  = "2015-01-01"
TRAIN_END    = "2022-12-31"
VAL_START    = "2023-01-01"
VAL_END      = "2023-12-31"
TEST_START   = "2024-01-01"
TEST_END     = "2026-12-31"

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_features(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    """Compute all daily features for one stock."""
    df = df.sort_values('date').copy()
    df = df.set_index('date')

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']
    open_  = df['open']

    # --- Returns ---
    df['ret_1d']  = close.pct_change(1)
    df['ret_3d']  = close.pct_change(3)
    df['ret_5d']  = close.pct_change(5)
    df['ret_10d'] = close.pct_change(10)
    df['ret_20d'] = close.pct_change(20)

    # --- Moving averages ---
    df['sma_5']   = close.rolling(5).mean()
    df['sma_10']  = close.rolling(10).mean()
    df['sma_20']  = close.rolling(20).mean()
    df['sma_50']  = close.rolling(50).mean()
    df['sma_200'] = close.rolling(200).mean()

    df['close_vs_sma5']   = (close - df['sma_5'])   / (df['sma_5']   + 1e-9)
    df['close_vs_sma10']  = (close - df['sma_10'])  / (df['sma_10']  + 1e-9)
    df['close_vs_sma20']  = (close - df['sma_20'])  / (df['sma_20']  + 1e-9)
    df['close_vs_sma50']  = (close - df['sma_50'])  / (df['sma_50']  + 1e-9)
    df['close_vs_sma200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-9)

    # --- EMA ---
    df['ema_12'] = close.ewm(span=12).mean()
    df['ema_26'] = close.ewm(span=26).mean()
    df['macd']   = (df['ema_12'] - df['ema_26']) / (close + 1e-9)

    # --- RSI ---
    for period in [7, 14, 21]:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{period}'] = df[f'rsi_{period}'] / 100.0  # normalize 0-1

    # --- Bollinger Bands ---
    for period in [20]:
        mid  = close.rolling(period).mean()
        std  = close.rolling(period).std()
        df[f'bb_upper_{period}'] = (mid + 2*std - close) / (close + 1e-9)
        df[f'bb_lower_{period}'] = (close - (mid - 2*std)) / (close + 1e-9)
        df[f'bb_width_{period}'] = (4*std) / (mid + 1e-9)
        df[f'bb_pos_{period}']   = (close - (mid - 2*std)) / (4*std + 1e-9)

    # --- ATR ---
    for period in [7, 14]:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df[f'atr_{period}']     = tr.rolling(period).mean()
        df[f'atr_pct_{period}'] = df[f'atr_{period}'] / (close + 1e-9)

    # --- Volume features ---
    df['vol_sma20']    = volume.rolling(20).mean()
    df['vol_ratio']    = volume / (df['vol_sma20'] + 1e-9)
    df['vol_ret']      = volume.pct_change(1).clip(-5, 5)

    # --- Momentum ---
    df['mom_5']  = close / (close.shift(5)  + 1e-9) - 1
    df['mom_10'] = close / (close.shift(10) + 1e-9) - 1
    df['mom_20'] = close / (close.shift(20) + 1e-9) - 1

    # --- High/Low features ---
    df['high_20d'] = high.rolling(20).max()
    df['low_20d']  = low.rolling(20).min()
    df['pct_from_high20'] = (close - df['high_20d']) / (df['high_20d'] + 1e-9)
    df['pct_from_low20']  = (close - df['low_20d'])  / (df['low_20d']  + 1e-9)

    df['high_52w'] = high.rolling(252).max()
    df['low_52w']  = low.rolling(252).min()
    df['pct_from_high52w'] = (close - df['high_52w']) / (df['high_52w'] + 1e-9)
    df['pct_from_low52w']  = (close - df['low_52w'])  / (df['low_52w']  + 1e-9)

    # --- Candle features ---
    df['body']       = (close - open_) / (open_ + 1e-9)
    df['upper_wick'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (close + 1e-9)
    df['lower_wick'] = (pd.concat([close, open_], axis=1).min(axis=1) - low)  / (close + 1e-9)

    # --- Volatility ---
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    df['vol_60d'] = df['ret_1d'].rolling(60).std()

    # --- Nifty features ---
    if nifty is not None and len(nifty) > 0:
        nifty_close = nifty.set_index('date')['close']
        nifty_close = nifty_close.reindex(df.index, method='ffill')
        df['nifty_ret_1d']  = nifty_close.pct_change(1)
        df['nifty_ret_5d']  = nifty_close.pct_change(5)
        df['nifty_ret_20d'] = nifty_close.pct_change(20)
        df['nifty_vs_sma20'] = (nifty_close - nifty_close.rolling(20).mean()) / (nifty_close.rolling(20).mean() + 1e-9)
        df['nifty_rsi14']   = _rsi(nifty_close, 14) / 100.0
        df['rel_strength']  = df['ret_20d'] - df['nifty_ret_20d']
    else:
        df['nifty_ret_1d']   = 0.0
        df['nifty_ret_5d']   = 0.0
        df['nifty_ret_20d']  = 0.0
        df['nifty_vs_sma20'] = 0.0
        df['nifty_rsi14']    = 0.5
        df['rel_strength']   = 0.0

    df = df.reset_index()
    return df


def _rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# =============================================================================
# MAIN
# =============================================================================
def build_features():
    print("=" * 60)
    print("DATASET BUILDER 2 -- 10 YEAR FEATURES")
    print("=" * 60)

    con = duckdb.connect(SNAPSHOT_DB, read_only=True)

    # Load Nifty
    print("\n[1] Loading Nifty data...")
    nifty = con.execute(
        "SELECT date, close FROM daily_1d WHERE symbol='NIFTY' ORDER BY date"
    ).fetchdf()
    nifty['date'] = pd.to_datetime(nifty['date'])
    print(f"    Nifty: {len(nifty)} rows  ({nifty['date'].min()} to {nifty['date'].max()})")

    # Build features per stock
    print("\n[2] Building features per stock...")
    all_frames = []

    for sym in GP_STOCKS:
        print(f"    {sym}...", end=" ")
        df = con.execute(
            f"SELECT date, open, high, low, close, volume "
            f"FROM daily_1d WHERE symbol='{sym}' ORDER BY date"
        ).fetchdf()
        df['date'] = pd.to_datetime(df['date'])

        feat = compute_features(df, nifty)
        feat['symbol'] = sym

        # Drop rows with too many NaNs (first 200 days for SMA200)
        feat = feat.dropna(subset=['sma_200', 'rsi_14', 'atr_14'])
        all_frames.append(feat)
        print(f"{len(feat)} rows")

    con.close()

    # Combine
    print("\n[3] Combining all stocks...")
    combined = pd.concat(all_frames, ignore_index=True)
    # Match old dataset format: DatetimeIndex named 'ts'
    combined['ts'] = pd.to_datetime(combined['date'], utc=True)
    combined = combined.drop(columns=['date'])
    combined = combined.sort_values(['symbol', 'ts'])
    combined = combined.set_index('ts')

    print(f"    Total rows: {len(combined):,}")
    print(f"    Date range: {combined.index.min()} to {combined.index.max()}")
    print(f"    Symbols: {combined['symbol'].unique().tolist()}")
    print(f"    Features: {len(combined.columns)} columns")

    # Show split sizes
    train = combined[combined.index <= pd.Timestamp(TRAIN_END, tz='UTC')]
    val   = combined[(combined.index >= pd.Timestamp(VAL_START, tz='UTC')) & (combined.index <= pd.Timestamp(VAL_END, tz='UTC'))]
    test  = combined[combined.index >= pd.Timestamp(TEST_START, tz='UTC')]

    print(f"\n[4] Split sizes:")
    print(f"    Train: {len(train):,} rows  ({TRAIN_START} to {TRAIN_END})")
    print(f"    Val:   {len(val):,}   rows  ({VAL_START} to {VAL_END})")
    print(f"    Test:  {len(test):,}  rows  ({TEST_START} to {TEST_END})")

    # Save
    Path(OUTPUT_FILE).parent.mkdir(exist_ok=True)
    # Ensure ts is the index before saving
    if 'ts' in combined.columns:
        combined = combined.set_index('ts')
    elif combined.index.name != 'ts':
        print(f'WARNING: index is {combined.index.name}, columns: {combined.columns.tolist()[:5]}')

    print(f'    Index before save: {combined.index.name}, type: {type(combined.index)}')
    combined.to_parquet(OUTPUT_FILE, index=True)
    print(f"\n[DONE] Saved to: {OUTPUT_FILE}")
    print(f"       File size: {Path(OUTPUT_FILE).stat().st_size / 1024 / 1024:.1f} MB")

    # Show feature list
    feature_cols = [c for c in combined.columns if c not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    print(f"\n[5] Feature list ({len(feature_cols)} features):")
    for i, f in enumerate(feature_cols, 1):
        print(f"    {i:3}. {f}")


if __name__ == "__main__":
    build_features()
