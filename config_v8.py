"""
config_v8.py
============
V8 configuration -- regime-aware GP evolution.
3 separate GP runs, one per regime.
Live signal = soft-weighted blend of 3 formulas.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
OUTPUT_DIR   = Path("gp_output")
PARQUET_FILE = OUTPUT_DIR / "gp_features_daily2.parquet"

# =============================================================================
# SYMBOLS
# =============================================================================
ALL_SYMBOLS = ['ICICIBANK', 'HDFCBANK', 'RELIANCE', 'INFY', 'TCS', 'WIPRO']

# =============================================================================
# DATE RANGES
# =============================================================================
TRAIN_START = "2015-10-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"

# =============================================================================
# REGIMES
# =============================================================================
REGIMES = ['BULL_QUIET', 'SIDEWAYS_BEAR', 'VOLATILE']
# CRISIS excluded -- too few days, go flat

REGIME_COLUMN       = 'regime_smooth'   # use smoothed labels
REGIME_MIN_DAYS     = 100               # minimum days per regime for GP
CRISIS_FLAT         = True              # no trades in CRISIS regime

# Smooth probability columns for signal blending
PROB_COLS = {
    'BULL_QUIET':    'smooth_bull_quiet',
    'SIDEWAYS_BEAR': 'smooth_sideways_bear',
    'VOLATILE':      'smooth_volatile',
    'CRISIS':        'smooth_crisis',
}

# =============================================================================
# FEATURES -- 16 RF selected features
# =============================================================================
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

# =============================================================================
# GP PARAMETERS -- per regime run
# =============================================================================
GP_POPULATION    = 4000    # smaller -- less data per regime
GP_GENERATIONS   = 80
GP_CROSSOVER     = 0.70
GP_MUTATION      = 0.25
GP_ELITE         = 30
GP_TOURNAMENT    = 4
GP_MAX_DEPTH     = 5
GP_MAX_NODES     = 20
GP_EARLY_STOP    = 20
GP_MIN_DEPTH     = 2
GP_WORKERS       = 8
GP_CHUNK_SIZE    = 350

# =============================================================================
# FITNESS WEIGHTS
# =============================================================================
W_ANNUAL_RETURN   = 3.0
W_SHARPE          = 2.0
W_SORTINO         = 1.0
W_WIN_RATE        = 1.0
W_MAX_DRAWDOWN    = 2.0
W_TRADE_COUNT     = 0.5
W_CONSISTENCY     = 1.5
LAMBDA_PARSIMONY  = 0.02   # linear parsimony

MIN_TRADES_TOTAL     = 5    # fewer -- less data per regime
MIN_TRADES_PER_STOCK = 1

EPSILON = 1e-8

# =============================================================================
# BACKTESTER
# =============================================================================
MIN_HOLD_DAYS  = 3
MAX_HOLD_DAYS  = 20
TOTAL_COST_PCT = 0.001
STOP_LOSS_PCT  = 0.05

# =============================================================================
# SIGNAL BLENDING
# =============================================================================
BLEND_MODE = 'soft'   # 'soft' = weighted blend, 'hard' = single formula
CRISIS_SCALE = 0.0    # scale all signals to 0 in crisis

# =============================================================================
# VALIDATION
# =============================================================================
def validate_config():
    assert GP_CROSSOVER + GP_MUTATION <= 1.0
    assert len(DAILY_FEATURES) > 0
    assert len(REGIMES) > 0
    assert GP_MIN_DEPTH < GP_MAX_DEPTH

import os as _os
if _os.environ.get("WORKER_PROCESS") != "1":
    validate_config()
