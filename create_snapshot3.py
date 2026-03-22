import duckdb
import os
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
SOURCE_DB   = r"D:\2. Anshul\openalgo\openalgo\openalgo_db\db\historify.duckdb"
SNAPSHOT_DB = r"gp_output\gp_snapshot2.duckdb"   # raw string fixes backslash warning

# GP trading stocks + Nifty index
GP_STOCKS = [
    "ICICIBANK", "HDFCBANK", "RELIANCE",
    "INFY", "TCS", "WIPRO"
]
INDEX    = "NIFTY"
ALL_SYMS = GP_STOCKS + [INDEX]

# Date range — 10 years
DATE_FROM = "2015-01-01"
DATE_TO   = "2026-12-31"

# =============================================================================
# MAIN
# =============================================================================
def make_snapshot2():

    # Clean old snapshot
    if os.path.exists(SNAPSHOT_DB):
        try:
            os.remove(SNAPSHOT_DB)
            print(f"[OK] Deleted old {SNAPSHOT_DB}")
        except PermissionError:
            print(f"[FAIL] Cannot delete {SNAPSHOT_DB} -- close it first")
            return

    sym_str = ", ".join([f"'{s}'" for s in ALL_SYMS])
    print(f"[INFO] Symbols: {', '.join(ALL_SYMS)}")
    print(f"[INFO] Date range: {DATE_FROM} to {DATE_TO}")

    # Connect to source with read_only=True to avoid lock conflict
    print(f"\n[INFO] Connecting to source DB...")
    try:
        src_con = duckdb.connect(SOURCE_DB, read_only=True)
        print(f"[OK] Connected to source DB")
    except Exception as e:
        print(f"[FAIL] Cannot connect to source DB: {e}")
        print("\n[TIP] Close Historify / OpenAlgo application first, then retry.")
        return

    res_con = duckdb.connect(SNAPSHOT_DB)

    # =========================================================================
    # 1. Extract 1D daily data
    # =========================================================================
    print("\n[STEP 1] Extracting 1D daily data...")
    df_daily = src_con.execute(f"""
        SELECT DISTINCT
            CAST(to_timestamp(CAST(timestamp AS BIGINT)) AS DATE) AS date,
            symbol,
            open, high, low, close, volume
        FROM market_data
        WHERE interval = 'D'
          AND symbol IN ({sym_str})
          AND to_timestamp(CAST(timestamp AS BIGINT)) >= '{DATE_FROM}'
          AND to_timestamp(CAST(timestamp AS BIGINT)) <= '{DATE_TO}'
        ORDER BY symbol, date
    """).fetchdf()

    print(f"   Loaded {len(df_daily):,} daily rows")

    if len(df_daily) == 0:
        print("[FAIL] No daily data found -- check symbol names in source DB")
        # Show available symbols
        try:
            syms = src_con.execute(
                "SELECT DISTINCT symbol FROM market_data WHERE interval='D' LIMIT 20"
            ).fetchdf()
            print(f"Available symbols in DB: {syms['symbol'].tolist()}")
        except Exception as e:
            print(f"Could not query symbols: {e}")
        src_con.close()
        res_con.close()
        return

    res_con.execute("CREATE TABLE daily_1d AS SELECT * FROM df_daily")

    d1_stats = res_con.execute(
        "SELECT symbol, COUNT(*) as rows, MIN(date) as from_dt, MAX(date) as to_dt "
        "FROM daily_1d GROUP BY symbol ORDER BY symbol"
    ).fetchall()

    print("[OK] Daily 1D rows:")
    for sym, cnt, f, t in d1_stats:
        print(f"   {sym:12}: {cnt:5,} rows  ({f} to {t})")

    # =========================================================================
    # 2. Extract 1M data and resample to 1H
    # =========================================================================
    print("\n[STEP 2] Extracting 1M data and resampling to 1H...")
    print("   [NOTE] This may take a few minutes for 10 years of 1M data...")

    df_1m = src_con.execute(f"""
        SELECT DISTINCT
            to_timestamp(CAST(timestamp AS BIGINT)) AS ts,
            symbol, open, high, low, close, volume
        FROM market_data
        WHERE interval = '1m'
          AND symbol IN ({sym_str})
          AND to_timestamp(CAST(timestamp AS BIGINT)) >= '{DATE_FROM}'
          AND to_timestamp(CAST(timestamp AS BIGINT)) <= '{DATE_TO}'
        ORDER BY symbol, ts
    """).fetchdf()

    print(f"   1M rows loaded: {len(df_1m):,}")

    if len(df_1m) == 0:
        print("[WARN] No 1M data found -- skipping 1H table")
        hourly_frames = []
    else:
        df_1m['ts'] = pd.to_datetime(df_1m['ts'])
        df_1m = df_1m.set_index('ts')

        hourly_frames = []
        for sym in ALL_SYMS:
            sym_df = df_1m[df_1m['symbol'] == sym].copy()
            if len(sym_df) == 0:
                print(f"   [WARN] No 1M data for {sym}")
                continue

            sym_1h = sym_df[['open','high','low','close','volume']].resample('1h').agg({
                'open':   'first',
                'high':   'max',
                'low':    'min',
                'close':  'last',
                'volume': 'sum'
            }).dropna(subset=['close'])

            sym_1h['symbol'] = sym
            sym_1h = sym_1h.reset_index()
            sym_1h = sym_1h.rename(columns={'ts': 'datetime'})
            hourly_frames.append(sym_1h)
            print(f"   {sym:12}: {len(sym_1h):6,} hourly bars")

    if hourly_frames:
        df_1h = pd.concat(hourly_frames, ignore_index=True)
        res_con.execute("CREATE TABLE hourly_1h AS SELECT * FROM df_1h")
        print(f"\n[OK] hourly_1h table: {len(df_1h):,} total rows")

    # =========================================================================
    # 3. Create indices
    # =========================================================================
    print("\n[STEP 3] Creating indices...")
    res_con.execute("CREATE INDEX idx_daily ON daily_1d (symbol, date)")
    if hourly_frames:
        res_con.execute("CREATE INDEX idx_hourly ON hourly_1h (symbol, datetime)")
    print("[OK] Indices created")

    # =========================================================================
    # 4. Verify
    # =========================================================================
    src_con.close()

    print("\n--- FINAL VERIFICATION ---")
    tables = res_con.execute("SHOW TABLES").fetchdf()
    print(f"Tables: {tables['name'].tolist()}")

    d1_total = res_con.execute("SELECT COUNT(*) FROM daily_1d").fetchone()[0]
    print(f"daily_1d  total rows: {d1_total:,}")

    if hourly_frames:
        h1_total = res_con.execute("SELECT COUNT(*) FROM hourly_1h").fetchone()[0]
        print(f"hourly_1h total rows: {h1_total:,}")

    print("\nDate ranges in daily_1d:")
    ranges = res_con.execute(
        "SELECT symbol, MIN(date), MAX(date), COUNT(*) "
        "FROM daily_1d GROUP BY symbol ORDER BY symbol"
    ).fetchall()
    for sym, mn, mx, cnt in ranges:
        print(f"   {sym:12}: {mn} to {mx}  ({cnt} days)")

    res_con.close()
    print(f"\n[DONE] Snapshot saved to: {SNAPSHOT_DB}")
    print("Next step: run dataset_builder2.py to rebuild features with 10 year data")


if __name__ == "__main__":
    make_snapshot2()
