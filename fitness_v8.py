"""
fitness_v8.py
=============
Regime-aware fitness function for V8 GP evolution.

Key differences from V7:
    - Evaluated on regime-specific data only
    - Linear parsimony pressure
    - Overfit guard: train vs val consistency
    - Regime-specific trade frequency targets
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from backtester import backtest_portfolio, BacktestResult
from config_v8 import (
    W_ANNUAL_RETURN, W_SHARPE, W_SORTINO, W_WIN_RATE,
    W_MAX_DRAWDOWN, W_TRADE_COUNT, W_CONSISTENCY,
    LAMBDA_PARSIMONY, MIN_TRADES_TOTAL, MIN_TRADES_PER_STOCK,
    EPSILON,
)

# =============================================================================
# REGIME-SPECIFIC TRADE FREQUENCY TARGETS
# =============================================================================
# Each regime has different expected trade frequency
REGIME_TRADE_TARGETS = {
    'BULL_QUIET':    (4,  30),   # min 4, max 30 trades/year
    'SIDEWAYS_BEAR': (4,  25),   # fewer trades in sideways
    'VOLATILE':      (6,  40),   # more trades in volatile
}

DEFAULT_TRADE_TARGET = (4, 35)

# =============================================================================
# SINGLE STOCK FITNESS
# =============================================================================
def single_stock_fitness(
    result:  BacktestResult,
    tree_size: int,
    regime:  str = 'BULL_QUIET',
) -> float:
    """
    Compute fitness for a single stock backtest result.
    Regime-aware trade frequency targets.
    """
    # Minimum trades gate
    if result.n_trades < MIN_TRADES_PER_STOCK:
        return -50.0

    ann_ret  = result.annual_return_pct / 100.0
    sharpe   = result.sharpe_ratio
    sortino  = result.sortino_ratio
    win_rate = result.win_rate / 100.0
    max_dd   = abs(result.max_drawdown_pct) / 100.0

    # Regime-specific trade frequency bonus
    min_trades, max_trades = REGIME_TRADE_TARGETS.get(
        regime, DEFAULT_TRADE_TARGET
    )
    tpy = result.trades_per_year
    if tpy < min_trades:
        trade_bonus = -1.0       # too few trades
    elif tpy <= max_trades:
        trade_bonus = 0.5        # ideal range
    else:
        trade_bonus = -0.5       # too many trades = overtrading

    # Linear parsimony
    complexity = LAMBDA_PARSIMONY * tree_size

    fitness = (
        W_ANNUAL_RETURN * ann_ret
        + W_SHARPE       * sharpe
        + W_SORTINO      * sortino
        + W_WIN_RATE     * win_rate
        - W_MAX_DRAWDOWN * max_dd
        + W_TRADE_COUNT  * trade_bonus
        - complexity
    )

    return float(fitness)


# =============================================================================
# PORTFOLIO FITNESS
# =============================================================================
def portfolio_fitness(
    results:   Dict[str, BacktestResult],
    tree_size: int,
    regime:    str = 'BULL_QUIET',
) -> float:
    """
    Aggregate per-stock fitness into portfolio fitness.
    Penalizes inconsistency across stocks.
    """
    if not results:
        return -50.0

    per_stock = {
        sym: single_stock_fitness(r, tree_size, regime)
        for sym, r in results.items()
    }
    scores = list(per_stock.values())
    valid  = [s for s in scores if s > -50.0]

    if len(valid) < 3:
        return -50.0 + len(valid) * 5.0

    mean_fit = np.mean(valid)
    std_fit  = np.std(valid)

    # Penalize inconsistency across stocks
    portfolio_fit = mean_fit - W_CONSISTENCY * std_fit

    return float(portfolio_fit)


# =============================================================================
# REGIME-AWARE FULL FITNESS
# =============================================================================
def evaluate_individual(
    func,
    train_data: Dict,
    val_data:   Dict,
    tree_size:  int,
    regime:     str,
) -> float:
    """
    Full fitness evaluation for one individual.

    Steps:
        1. Backtest on train data
        2. Backtest on val data
        3. Compute train + val fitness
        4. Apply overfit penalty
        5. Return final fitness
    """
    # --- Train fitness ---
    train_results = backtest_portfolio(func, train_data)
    train_fit     = portfolio_fitness(train_results, tree_size, regime)

    # Early exit if train is terrible
    if train_fit <= -40.0:
        return -50.0

    # --- Val fitness ---
    val_results = backtest_portfolio(func, val_data)
    val_fit     = portfolio_fitness(val_results, tree_size, regime)

    # --- Overfit guard ---
    overfit_penalty = 1.0   # default: no penalty

    if train_fit > EPSILON:
        if val_fit <= 0:
            # Loses money on val -- strong overfit signal
            overfit_penalty = 0.1
        else:
            consistency = val_fit / (train_fit + EPSILON)
            if consistency >= 0.8:
                overfit_penalty = 1.0    # great generalization
            elif consistency >= 0.5:
                overfit_penalty = 0.8    # acceptable
            elif consistency >= 0.2:
                overfit_penalty = 0.5    # weak generalization
            else:
                overfit_penalty = 0.2    # poor generalization
    else:
        # Train itself is bad
        overfit_penalty = 0.5

    # --- Final fitness ---
    # Weight val more than train (60/40)
    final_fit = (0.4 * train_fit + 0.6 * val_fit) * overfit_penalty

    return float(final_fit)


# =============================================================================
# MULTIPROCESSING WORKER
# =============================================================================
# Global state for workers
_GLOBAL_TOOLBOX     = None
_GLOBAL_TRAIN_DATA  = None
_GLOBAL_VAL_DATA    = None
_GLOBAL_REGIME      = None


def init_worker(toolbox, train_data, val_data, regime):
    """Initialize worker process with shared data."""
    global _GLOBAL_TOOLBOX, _GLOBAL_TRAIN_DATA
    global _GLOBAL_VAL_DATA, _GLOBAL_REGIME
    _GLOBAL_TOOLBOX    = toolbox
    _GLOBAL_TRAIN_DATA = train_data
    _GLOBAL_VAL_DATA   = val_data
    _GLOBAL_REGIME     = regime


def evaluate_worker(individual):
    """Worker function for multiprocessing evaluation."""
    global _GLOBAL_TOOLBOX, _GLOBAL_TRAIN_DATA
    global _GLOBAL_VAL_DATA, _GLOBAL_REGIME

    try:
        func = _GLOBAL_TOOLBOX.compile(expr=individual)
        fit  = evaluate_individual(
            func,
            _GLOBAL_TRAIN_DATA,
            _GLOBAL_VAL_DATA,
            len(individual),
            _GLOBAL_REGIME,
        )
        individual.fitness.values = (fit,)
    except Exception:
        individual.fitness.values = (-50.0,)

    return individual
