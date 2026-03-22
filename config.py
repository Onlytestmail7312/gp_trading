"""
Configuration for Unified Multi-Stock GP Trading System.

All tuneable constants, file paths, feature lists, and GP parameters
live here so every other module imports from one place.
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════
DB_PATH = "gp_research.duckdb"
OUTPUT_DIR = Path("gp_output")
FEATURE_TABLE = "gp_features_1m"
FEATURE_PARQUET = OUTPUT_DIR / "gp_features_1m.parquet"

# ═══════════════════════════════════════════════════════════════════
# SYMBOLS
# ═══════════════════════════════════════════════════════════════════
TRADABLE_STOCKS = ["ICICIBANK", "HDFCBANK", "RELIANCE", "INFY", "TCS", "WIPRO"]
MARKET_CONTEXT = "NIFTY"
ALL_SYMBOLS = TRADABLE_STOCKS + [MARKET_CONTEXT]

# ═══════════════════════════════════════════════════════════════════
# MARKET SESSION
# ═══════════════════════════════════════════════════════════════════
BARS_PER_DAY = 375
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30
SESSION_MINUTES = 375  # 9:15 to 15:30

# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING PARAMETERS
# ═══════════════════════════════════════════════════════════════════
RETURN_PERIODS = [1, 5, 15, 30, 60]
SMA_PERIODS_BARS = {
    "sma20": 20 * BARS_PER_DAY,
    "sma50": 50 * BARS_PER_DAY,
    "sma100": 100 * BARS_PER_DAY,
    "sma200": 200 * BARS_PER_DAY,
}
ATR_PERIOD_BARS = 14 * BARS_PER_DAY
VOLATILITY_PERIOD = 20 * BARS_PER_DAY
VOLUME_REL_PERIODS = [5, 20]
RANGE_PCT_PERIODS = [5, 20]
TREND_50D_BARS = 50 * BARS_PER_DAY
TREND_20D_BARS = 20 * BARS_PER_DAY

FORWARD_RETURN_PERIODS = [5, 15, 30, 60]

# ═══════════════════════════════════════════════════════════════════
# V1 FEATURE SET — compact GP terminal inputs
# ═══════════════════════════════════════════════════════════════════
V1_STOCK_FEATURES = [
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "ret_60m",
    "close_vs_sma20",
    "close_vs_sma50",
    "close_vs_sma100",
    "close_vs_sma200",
    "atr_pct",
    "volume_rel_20",
]

V1_MARKET_FEATURES = [
    "nifty_ret_5m",
    "nifty_ret_15m",
    "nifty_ret_60m",
    "nifty_close_vs_sma200",
]

V1_RELATIVE_STRENGTH = [
    "rs_15m",
    "rs_60m",
    "rs_20d",
]

V1_TIME_FEATURES = [
    "minutes_from_open_norm",
]

V1_GP_FEATURES = (
    V1_STOCK_FEATURES
    + V1_MARKET_FEATURES
    + V1_RELATIVE_STRENGTH
    + V1_TIME_FEATURES
)

# ═══════════════════════════════════════════════════════════════════
# TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════
TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"
# Everything after VAL_END is test

# ═══════════════════════════════════════════════════════════════════
# REGIME EVALUATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════
REGIME_BULL_THRESHOLD = 0.05   # trend_50d_return > +5%
REGIME_BEAR_THRESHOLD = -0.05  # trend_50d_return < -5%

# ═══════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════
STARTING_CASH = 1_000_000
COMMISSION_RATE = 0.0003   # 3 bps round-trip
SLIPPAGE_PCT = 0.0001      # 1 bp slippage
ENTRY_THRESHOLD = 0.7
EXIT_THRESHOLD = -0.7


# ═══════════════════════════════════════════════════════════════════
# GP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
@dataclass
class GPConfig:
    """All GP evolution hyperparameters."""
    # Population
    pop_size: int = 5000
    n_generations: int = 50
    tournament_size: int = 5
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    elite_size: int = 10

    # Tree constraints
    max_tree_depth: int = 8
    max_tree_nodes: int = 50
    min_tree_depth: int = 2
    init_min_depth: int = 2
    init_max_depth: int = 5

    # Fitness
    min_trades: int = 1
    max_trades_per_day: float = 20.0
    max_drawdown_pct: float = 50.0
    complexity_penalty_weight: float = 0.05

    # Multi-stock aggregation
    lambda_robustness: float = 0.5

    # Fitness weights
    w_sharpe: float = 1.0
    w_sortino: float = 0.5
    w_annual_return: float = 0.3
    w_max_drawdown: float = 0.5
    w_trades_per_day: float = 0.1
    w_low_trades: float = 0.5
    w_trade_bonus: float = 0.2

    # Misc
    random_seed: int = 42
    n_workers: int = -1  # -1 = all CPUs
    chunk_size: int = 200
    checkpoint_every: int = 5  # save every N generations

    # Files
    checkpoint_file: str = str(OUTPUT_DIR / "checkpoint.pkl")
    best_model_file: str = str(OUTPUT_DIR / "best_model.pkl")
    stats_file: str = str(OUTPUT_DIR / "evolution_stats.csv")
    log_file: str = str(OUTPUT_DIR / "training.log")

    # Backtest
    starting_cash: float = float(STARTING_CASH)
    commission: float = COMMISSION_RATE
    slippage: float = SLIPPAGE_PCT
    entry_threshold: float = ENTRY_THRESHOLD
    exit_threshold: float = EXIT_THRESHOLD


DEFAULT_GP_CONFIG = GPConfig()


# ═══════════════════════════════════════════════════════════════════
# NUMERIC SAFETY
# ═══════════════════════════════════════════════════════════════════
EPSILON = 1e-10