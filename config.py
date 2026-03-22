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
DB_PATH    = BASE_DIR / "gp_research.duckdb"
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
TRAIN_START = "2021-08-16"
TRAIN_END   = "2023-12-31"
VAL_START   = "2024-01-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"

# ============================================================================
# DAILY FEATURE LIST  (GP terminals)
# ============================================================================
DAILY_FEATURES = [
    # Price returns
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    # Trend
    "close_vs_sma10",
    "close_vs_sma20",
    "close_vs_sma50",
    "close_vs_sma200",
    # Momentum
    "rsi_14",
    "macd_signal",
    "macd_hist",
    # Volatility
    "atr_pct_14",
    "bb_pct",
    # Volume
    "volume_rel_20",
    # Market (Nifty)
    "nifty_ret_5d",
    "nifty_ret_20d",
    "nifty_vs_sma50",
    # Relative strength vs Nifty
    "rs_5d",
    "rs_20d",
]

N_FEATURES = len(DAILY_FEATURES)

# ============================================================================
# POSITIONAL TRADING RULES
# ============================================================================
MIN_HOLD_DAYS   = 3
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
TOTAL_COST_PCT  = COMMISSION_PCT + SLIPPAGE_PCT

# ============================================================================
# GP PARAMETERS
# ============================================================================
GP_POPULATION    = 3000
GP_GENERATIONS   = 50
GP_CROSSOVER     = 0.75
GP_MUTATION      = 0.15
GP_ELITE         = 20
GP_TOURNAMENT    = 7
GP_MIN_DEPTH     = 2
GP_MAX_DEPTH     = 8
GP_MAX_NODES     = 50
GP_EARLY_STOP    = 20
GP_CHUNK_SIZE    = 300
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
COMPLEXITY_PENALTY = 0.1

MIN_TRADES_TOTAL     = 10
MIN_TRADES_PER_STOCK = 2

EPSILON = 1e-8
