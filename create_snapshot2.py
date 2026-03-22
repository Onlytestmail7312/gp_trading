import duckdb
import os

# =============================================================================
# CONFIGURATION: ADDING VALIDATION STOCKS
# =============================================================================
SOURCE_DB = r"D:\2. Anshul\openalgo\openalgo\openalgo_db\db\historify.duckdb"
RESEARCH_DB = "gp_research.duckdb"

# Define all symbols needed for Target, Index, and Multi-Stock Validation
TARGET_STOCK = "ICICIBANK"
INDEX = "NIFTY"
# These match the validation_stocks in your Ninja script
VALIDATION_STOCKS = ["HDFCBANK", "RELIANCE", "TCS", "INFY", "WIPRO", "BHARTIARTL", "IDEA", "HINDUNILVR", "SBIN", "SUNPHARMA"]

# Combine into one unique list
ALL_SYMBOLS = list(set([TARGET_STOCK, INDEX] + VALIDATION_STOCKS))

def make_snapshot():
    # Delete the old research DB if it exists to ensure a clean slate
    if os.path.exists(RESEARCH_DB):
        try:
            os.remove(RESEARCH_DB)
            print(f"[?] Deleted old {RESEARCH_DB}")
        except PermissionError:
            print(f"[FAIL] Could not delete {RESEARCH_DB}. Is it open in another program?")
            return

    # 1. Connect and Attach
    res_con = duckdb.connect(RESEARCH_DB)
    res_con.execute(f"ATTACH '{SOURCE_DB}' AS source (READ_ONLY TRUE)")

    # Convert list to SQL format: 'STOCK1', 'STOCK2', 'STOCK3'
    symbol_str = ", ".join([f"'{s}'" for s in ALL_SYMBOLS])

    print(f"[?] Extracting 1m and Daily data for: {', '.join(ALL_SYMBOLS)}")

    # 2. Extract 1-Minute Data (DISTINCT prevents duplicates)
    # Using DISTINCT ensures that if the source has double entries, the snapshot won't.
    res_con.execute(f"""
        CREATE TABLE min_1m AS 
        SELECT DISTINCT
            to_timestamp(cast(timestamp as BIGINT)) as ts, 
            symbol, open, high, low, close, volume
        FROM source.market_data 
        WHERE interval = '1m' 
        AND symbol IN ({symbol_str})
    """)

    # 3. Extract Daily Data (DISTINCT prevents duplicates)
    res_con.execute(f"""
        CREATE TABLE daily_ctx AS 
        SELECT DISTINCT
            to_timestamp(cast(timestamp as BIGINT)) as ts, 
            symbol,
            close as prev_close, 
            high as prev_high, 
            low as prev_low
        FROM source.market_data
        WHERE interval = 'D'
        AND symbol IN ({symbol_str})
    """)

    # 4. Create Performance Indices
    print("[?] Creating Search Indices for ASOF JOIN speed...")
    res_con.execute("CREATE INDEX idx_m1 ON min_1m (symbol, ts)")
    res_con.execute("CREATE INDEX idx_d1 ON daily_ctx (symbol, ts)")

    res_con.execute("DETACH source")

    # 5. FINAL VERIFICATION: Show breakdown per symbol
    print("\n--- SNAPSHOT VERIFICATION ---")
    
    # 1-Minute Breakdown
    m1_stats = res_con.execute("SELECT symbol, count(*) FROM min_1m GROUP BY symbol ORDER BY symbol").fetchall()
    print("[OK] 1-Minute Row Counts:")
    for sym, count in m1_stats:
        print(f"   - {sym:10}: {count:,} rows")

    # Daily Breakdown
    d1_stats = res_con.execute("SELECT symbol, count(*) FROM daily_ctx GROUP BY symbol ORDER BY symbol").fetchall()
    print("\n[OK] Daily Row Counts:")
    for sym, count in d1_stats:
        print(f"   - {sym:10}: {count:,} rows")

    total_m1 = sum(count for sym, count in m1_stats)
    if total_m1 == 0:
        print("\n[FAIL] WARNING: Snapshot is empty. Verify symbol names in source DB!")
    else:
        print(f"\n[?] Success! '{RESEARCH_DB}' is ready for Multi-Stock Genetic Programming.")

    res_con.close()

if __name__ == "__main__":
    make_snapshot()