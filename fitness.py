"""
fitness.py -- fitness for positional GP trading.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional

from config import (
    W_ANNUAL_RETURN, W_SHARPE, W_SORTINO, W_WIN_RATE,
    W_MAX_DRAWDOWN, W_TRADE_COUNT, W_CONSISTENCY,
    COMPLEXITY_PENALTY, GP_MAX_NODES,
    MIN_TRADES_TOTAL, MIN_TRADES_PER_STOCK,
    EPSILON,
)
from backtester import backtest_portfolio, BacktestResult


def single_stock_fitness(result: BacktestResult, tree_size: int) -> float:
    if result.n_trades < MIN_TRADES_PER_STOCK:
        return -50.0

    ann_ret  = result.annual_return_pct / 100.0
    sharpe   = result.sharpe_ratio
    sortino  = result.sortino_ratio
    win_rate = result.win_rate / 100.0
    max_dd   = abs(result.max_drawdown_pct) / 100.0

    trades_per_yr = result.trades_per_year
    if trades_per_yr < 4:
        trade_bonus = -1.0
    elif trades_per_yr <= 50:
        trade_bonus = 0.5
    else:
        trade_bonus = -0.5

    complexity = COMPLEXITY_PENALTY * (tree_size / GP_MAX_NODES)

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


def portfolio_fitness(results: Dict[str, BacktestResult], tree_size: int) -> float:
    if not results:
        return -50.0

    per_stock = {sym: single_stock_fitness(r, tree_size) for sym, r in results.items()}
    scores = list(per_stock.values())

    valid = [s for s in scores if s > -50.0]
    if len(valid) < 2:
        return -50.0 + len(valid) * 5.0

    mean_fit = np.mean(valid)
    std_fit  = np.std(valid)

    return float(mean_fit - W_CONSISTENCY * std_fit)


# Global state for multiprocessing workers
_GLOBAL_TOOLBOX    = None
_GLOBAL_TRAIN_DATA = None
_GLOBAL_VAL_DATA   = None

def init_worker(toolbox, train_data, val_data):
    """Initialise worker process with shared state."""
    global _GLOBAL_TOOLBOX, _GLOBAL_TRAIN_DATA, _GLOBAL_VAL_DATA
    _GLOBAL_TOOLBOX    = toolbox
    _GLOBAL_TRAIN_DATA = train_data
    _GLOBAL_VAL_DATA   = val_data


def evaluate_individual(
    individual,
    toolbox=None,
    train_data: Optional[Dict] = None,
    val_data: Optional[Dict] = None,
    cfg=None,
    generation: int = 0,
) -> Tuple[float, ...]:
    BASE_FAIL = -50.0

    try:
        # Use globals if not passed directly (multiprocessing workers)
        if toolbox is None:
            toolbox = _GLOBAL_TOOLBOX
        if train_data is None:
            train_data = _GLOBAL_TRAIN_DATA
        if val_data is None:
            val_data = _GLOBAL_VAL_DATA

        if toolbox is None or train_data is None or val_data is None:
            return (BASE_FAIL,)

        if generation == 0 and hasattr(toolbox, "current_generation"):
            generation = toolbox.current_generation

        func      = toolbox.compile(expr=individual)
        tree_size = len(individual)

        train_results = backtest_portfolio(func, train_data)
        train_fit     = portfolio_fitness(train_results, tree_size)

        val_results = backtest_portfolio(func, val_data)
        val_fit     = portfolio_fitness(val_results, tree_size)

        total_trades = sum(r.n_trades for r in train_results.values())
        if total_trades < MIN_TRADES_TOTAL:
            return (BASE_FAIL,)

        if train_fit <= -40.0:
            return (BASE_FAIL,)

        if train_fit > EPSILON:
            consistency = val_fit / (train_fit + EPSILON)
            if consistency < 0.3:
                overfit_penalty = 0.3
            elif consistency < 0.6:
                overfit_penalty = 0.7
            else:
                overfit_penalty = 1.0
        else:
            overfit_penalty = 0.5

        final_fit = (0.4 * train_fit + 0.6 * val_fit) * overfit_penalty

        return (float(final_fit),)

    except Exception:
        return (-50.0,)
