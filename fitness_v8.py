"""
fitness_v8.py
=============
Regime-aware fitness function for V8 GP evolution.
Uses dill for multiprocessing serialization.
Toolbox initialized inside each worker process.
"""

import numpy as np
from typing import Dict
import dill

from backtester import backtest_portfolio, BacktestResult
from config_v8 import (
    W_ANNUAL_RETURN, W_SHARPE, W_SORTINO, W_WIN_RATE,
    W_MAX_DRAWDOWN, W_TRADE_COUNT, W_CONSISTENCY,
    LAMBDA_PARSIMONY, MIN_TRADES_PER_STOCK,
    EPSILON, DAILY_FEATURES,
)

# =============================================================================
# REGIME-SPECIFIC TRADE FREQUENCY TARGETS
# =============================================================================
REGIME_TRADE_TARGETS = {
    'BULL_QUIET':    (4,  30),
    'SIDEWAYS_BEAR': (4,  25),
    'VOLATILE':      (6,  40),
}
DEFAULT_TRADE_TARGET = (4, 35)

# =============================================================================
# GLOBAL WORKER STATE
# =============================================================================
_WORKER_STATE = {}


def init_worker(pset_dill, train_data, val_data, regime):
    """Initialize worker with dill-serialized pset."""
    import os
    os.environ["WORKER_PROCESS"] = "1"
    global _WORKER_STATE
    import dill as _dill
    from deap import gp, creator, base, tools
    from gp_individual import setup_gp_toolbox

    # Rebuild toolbox inside worker
    toolbox, pset = setup_gp_toolbox(DAILY_FEATURES)

    _WORKER_STATE['toolbox']     = toolbox
    _WORKER_STATE['train_data']  = train_data
    _WORKER_STATE['val_data']    = val_data
    _WORKER_STATE['regime']      = regime


# =============================================================================
# SINGLE STOCK FITNESS
# =============================================================================
def single_stock_fitness(
    result:    BacktestResult,
    tree_size: int,
    regime:    str = 'BULL_QUIET',
) -> float:
    if result.n_trades < MIN_TRADES_PER_STOCK:
        return -50.0

    ann_ret  = result.annual_return_pct / 100.0
    sharpe   = result.sharpe_ratio
    sortino  = result.sortino_ratio
    win_rate = result.win_rate / 100.0
    max_dd   = abs(result.max_drawdown_pct) / 100.0

    min_trades, max_trades = REGIME_TRADE_TARGETS.get(
        regime, DEFAULT_TRADE_TARGET
    )
    tpy = result.trades_per_year
    if tpy < min_trades:
        trade_bonus = -1.0
    elif tpy <= max_trades:
        trade_bonus = 0.5
    else:
        trade_bonus = -0.5

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

    return float(mean_fit - W_CONSISTENCY * std_fit)


# =============================================================================
# FULL EVALUATION
# =============================================================================
def evaluate_individual(
    func,
    train_data: Dict,
    val_data:   Dict,
    tree_size:  int,
    regime:     str,
) -> float:
    # Train fitness
    train_results = backtest_portfolio(func, train_data)
    train_fit     = portfolio_fitness(train_results, tree_size, regime)

    if train_fit <= -40.0:
        return -50.0

    # Val fitness
    val_results = backtest_portfolio(func, val_data)
    val_fit     = portfolio_fitness(val_results, tree_size, regime)

    # Overfit guard
    overfit_penalty = 1.0
    if train_fit > EPSILON:
        if val_fit <= 0:
            overfit_penalty = 0.1
        else:
            consistency = val_fit / (train_fit + EPSILON)
            if consistency >= 0.8:
                overfit_penalty = 1.0
            elif consistency >= 0.5:
                overfit_penalty = 0.8
            elif consistency >= 0.2:
                overfit_penalty = 0.5
            else:
                overfit_penalty = 0.2
    else:
        overfit_penalty = 0.5

    final_fit = (0.4 * train_fit + 0.6 * val_fit) * overfit_penalty
    return float(final_fit)


# =============================================================================
# WORKER FUNCTION
# =============================================================================
def evaluate_worker(ind_dill):
    """Worker function — receives dill-serialized individual."""
    global _WORKER_STATE
    try:
        individual = dill.loads(ind_dill)
        toolbox    = _WORKER_STATE['toolbox']
        train_data = _WORKER_STATE['train_data']
        val_data   = _WORKER_STATE['val_data']
        regime     = _WORKER_STATE['regime']

        func = toolbox.compile(expr=individual)
        fit  = evaluate_individual(
            func, train_data, val_data,
            len(individual), regime,
        )
        return (fit,)
    except Exception:
        return (-50.0,)
