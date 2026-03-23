"""
config.py -- all configuration for the GP positional trading system.

TRADING RULES:
- Long/Short positional trading
- Entry  : signal crosses zero (pos -> long, neg -> short)
- Exit   : signal crosses back through zero
- Size   : signal-strength based, max 30% per stock, min 5%
- Stocks : all 6 simultaneously
- Hold   : 3 days minimum, 60 days maximum
"""

from pathlib import Path
import numpy as np

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "gp_output/gp_snapshot2.duckdb"
OUTPUT_DIR = BASE_DIR / "gp_output"

# ============================================================================
# UNIVERSE
# ============================================================================
ALL_SYMBOLS    = ["ICICIBANK", "HDFCBANK", "RELIANCE", "INFY", "TCS", "WIPRO"]
NIFTY_SYMBOL   = "NIFTY"
BARS_PER_DAY   = 375   # 1-minute bars per trading day

# ============================================================================
# TIME SPLITS
# ============================================================================
TRAIN_START = "2019-01-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"

# ============================================================================
# DAILY FEATURE LIST  (GP terminals)
# ============================================================================
DAILY_FEATURES = [
    "vol_60d",
    "nifty_rsi14",
    "pct_from_high52w",
    "nifty_ret_20d",
    "nifty_ret_5d",
    "rel_strength",
    "vol_20d",
    "pct_from_low52w",
    "rsi_7",
    "close_vs_sma200",
    "bb_width_20",
    "atr_pct_7",
    "ret_20d",
    "ret_5d",
    "bb_upper_20",
    "upper_wick",
]

N_FEATURES = len(DAILY_FEATURES)

# ============================================================================
# POSITIONAL TRADING RULES
# ============================================================================
MIN_HOLD_DAYS   = 5
MAX_HOLD_DAYS   = 60
ENTRY_THRESHOLD = 0.0
EXIT_THRESHOLD  = 0.0

# Position sizing
MAX_POSITION_WEIGHT = 0.30
MIN_POSITION_WEIGHT = 0.05
TOTAL_CAPITAL       = 1_000_000.0

# Transaction costs
COMMISSION_PCT  = 0.0003
SLIPPAGE_PCT    = 0.0002
STOP_LOSS_PCT      = 0.05   # 5% hard stop loss
TRAILING_STOP_PCT  = 0.07   # 7% trailing stop
TOTAL_COST_PCT  = COMMISSION_PCT + SLIPPAGE_PCT

# ============================================================================
# GP PARAMETERS
# ============================================================================
GP_POPULATION    = 5000
GP_GENERATIONS   = 100
GP_CROSSOVER     = 0.70
GP_MUTATION      = 0.25
GP_ELITE         = 30
GP_TOURNAMENT    = 4
GP_MIN_DEPTH     = 2
GP_MAX_DEPTH     = 5
GP_MAX_NODES     = 20
GP_EARLY_STOP    = 25
GP_CHUNK_SIZE    = 350
GP_WORKERS       = 8

# ============================================================================
# FITNESS WEIGHTS
# ============================================================================
W_ANNUAL_RETURN   = 3.0
W_SHARPE          = 2.0
W_SORTINO         = 1.0
W_WIN_RATE        = 1.0
W_MAX_DRAWDOWN    = 2.0
W_TRADE_COUNT     = 0.5
W_CONSISTENCY     = 1.5
COMPLEXITY_PENALTY = 0.05

MIN_TRADES_TOTAL     = 10
MIN_TRADES_PER_STOCK = 2

EPSILON = 1e-8


# ===========================================================================
# CONFIG VALIDATION
# ===========================================================================

def validate_config():
    """Validate configuration values to catch misconfiguration early."""
    assert GP_POPULATION > 0,            "GP_POPULATION must be positive"
    assert GP_GENERATIONS > 0,           "GP_GENERATIONS must be positive"
    assert 0 < GP_CROSSOVER < 1,         "GP_CROSSOVER must be in (0, 1)"
    assert 0 < GP_MUTATION < 1,          "GP_MUTATION must be in (0, 1)"
    assert GP_CROSSOVER + GP_MUTATION <= 1.0, "GP_CROSSOVER + GP_MUTATION should be <= 1.0"
    assert MIN_HOLD_DAYS < MAX_HOLD_DAYS, "MIN_HOLD_DAYS must be less than MAX_HOLD_DAYS"
    assert GP_MIN_DEPTH < GP_MAX_DEPTH,  "GP_MIN_DEPTH must be less than GP_MAX_DEPTH"
    assert STOP_LOSS_PCT < TRAILING_STOP_PCT, "STOP_LOSS_PCT should be tighter than TRAILING_STOP_PCT"
    print("  [OK] Config validation passed")


if __name__ == "__main__":
    validate_config()
