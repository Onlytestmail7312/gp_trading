"""
Fitness — composite fitness scoring with multi-stock aggregation.

Spec Section 9: Fitness Function
  composite_fitness = (
      w_sharpe * sharpe
    + w_sortino * sortino
    + w_annual_return * annual_return_pct / 100
    - w_max_drawdown * |max_drawdown_pct| / 100
    - w_trades_per_day * max(0, trades_per_day - 5) / 10
    - complexity_penalty_weight * tree_size / max_nodes
  )

Spec Section 11: Multi-Stock Aggregation
  fitness_total = mean(stock_fitness) - lambda * std(stock_fitness)
"""

import math
import traceback
import numpy as np
from typing import Tuple, Dict, List, Optional

from .config import GPConfig, DEFAULT_GP_CONFIG, V1_GP_FEATURES, EPSILON
from .backtester import (
    BacktestResult,
    backtest_stock,
    generate_signals_vectorised,
    run_backtest,
)
from .gp_primitives import check_signal_diversity, normalise_signal


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE FITNESS (single stock)
# ═══════════════════════════════════════════════════════════════════════════

def composite_fitness(
    result: BacktestResult,
    tree_size: int,
    cfg: GPConfig = None,
) -> float:
    """
    Compute composite fitness for a single stock backtest result.

    Parameters
    ----------
    result : BacktestResult
        Backtest performance metrics.
    tree_size : int
        Number of nodes in the GP tree.
    cfg : GPConfig
        Configuration with fitness weights.

    Returns
    -------
    float
        Composite fitness score. Higher is better.
    """
    if cfg is None:
        cfg = DEFAULT_GP_CONFIG

    # ── Component scores ───────────────────────────────────────────────
    sharpe_score = cfg.w_sharpe * result.sharpe_ratio

    sortino_score = cfg.w_sortino * result.sortino_ratio

    return_score = cfg.w_annual_return * (result.annual_return_pct / 100.0)

    dd_penalty = cfg.w_max_drawdown * (abs(result.max_drawdown_pct) / 100.0)

    # Penalise excessive trading (more than 5 trades/day)
    excess_trades = max(0.0, result.trades_per_day - 5.0)
    trade_penalty = cfg.w_trades_per_day * (excess_trades / 10.0)

    # Complexity penalty — prefer smaller trees
    complexity_penalty = cfg.complexity_penalty_weight * (
        tree_size / cfg.max_tree_nodes
    )

    # Penalise low trade counts
    low_trade_penalty = cfg.w_low_trades * max(0, cfg.min_trades - result.n_trades) / cfg.min_trades

    # Bonus for high trade counts
    trade_bonus = cfg.w_trade_bonus * min(result.n_trades / 50, 1)

    # ── Composite ──────────────────────────────────────────────────────
    fitness = (
        sharpe_score
        + sortino_score
        + return_score
        - dd_penalty
        - trade_penalty
        - complexity_penalty
        - low_trade_penalty
        + trade_bonus
    )

    return fitness


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-STOCK AGGREGATED FITNESS
# ═══════════════════════════════════════════════════════════════════════════

def multi_stock_fitness(
    per_stock_fitness: Dict[str, float],
    cfg: GPConfig = None,
) -> float:
    """
    Aggregate per-stock fitness scores into a single total fitness.

    fitness_total = mean(stock_fitness) - lambda * std(stock_fitness)

    This rewards strategies that work consistently across stocks while
    penalising high variance in performance.

    Parameters
    ----------
    per_stock_fitness : dict
        {symbol: fitness_score}
    cfg : GPConfig
        Configuration with lambda_robustness.

    Returns
    -------
    float
        Aggregated fitness score.
    """
    if cfg is None:
        cfg = DEFAULT_GP_CONFIG

    if not per_stock_fitness:
        return -100.0

    scores = list(per_stock_fitness.values())

    if len(scores) == 1:
        return scores[0]

    mean_fitness = np.mean(scores)
    std_fitness = np.std(scores)

    total = mean_fitness - cfg.lambda_robustness * std_fitness

    return float(total)


# ═══════════════════════════════════════════════════════════════════════════
# FULL EVALUATION PIPELINE (called by GP engine)
# ═══════════════════════════════════════════════════════════════════════════

# Minimum tree size to avoid degenerate formulas
MIN_TREE_LENGTH = 7

# Reject unrealistic returns (likely overfitting)
MAX_RETURN_CAP = 500.0  # percent


def evaluate_individual(
    individual,
    toolbox=None,
    train_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    val_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    train_feature_matrices: Dict[str, np.ndarray] = None,
    cfg: GPConfig = None,
) -> Tuple[float, ...]:
    """
    Full fitness evaluation pipeline for one GP individual.

    Multi-stage filter approach (from reference script):
    F0: Tree complexity check
    F0b: Signal diversity check
    F1: Train backtest — minimum activity
    F2: Train drawdown filter
    F3: Both train and val must be positive return
    F4: Val backtest — Sharpe filter
    F5: Composite fitness with multi-stock aggregation

    Parameters
    ----------
    individual : DEAP GP individual
        The tree to evaluate.
    toolbox : deap.base.Toolbox
        For compiling the individual.
    train_data : dict
        {symbol: (feature_matrix, close_prices)} for training period.
    val_data : dict
        {symbol: (feature_matrix, close_prices)} for validation period.
    train_feature_matrices : dict
        {symbol: feature_matrix} for signal diversity check (train only).
    cfg : GPConfig
        Configuration.

    Returns
    -------
    tuple of float
        (fitness,) — single-objective.
    """
    FAIL = (-100.0,)

    if cfg is None:
        cfg = DEFAULT_GP_CONFIG

    try:
        # ── F0: Minimum tree complexity ────────────────────────────────
        if len(individual) < MIN_TREE_LENGTH:
            return FAIL

        # ── Compile tree ───────────────────────────────────────────────
        if toolbox is None:
            return FAIL
        func = toolbox.compile(expr=individual)

        # ── F0b: Signal diversity check ────────────────────────────────
        # Check on first available stock's training data
        if train_feature_matrices:
            first_sym = next(iter(train_feature_matrices))
            fm = train_feature_matrices[first_sym]
            if not check_signal_diversity(func, fm, n_samples=500):
                return FAIL

        # ── Per-stock evaluation ───────────────────────────────────────
        if train_data is None or val_data is None:
            return FAIL

        per_stock_fitness = {}

        for symbol in train_data:
            if symbol not in val_data:
                continue

            train_features, train_prices = train_data[symbol]
            val_features, val_prices = val_data[symbol]

            # ── Stage 1: Train backtest ────────────────────────────────
            train_result = backtest_stock(func, train_features, train_prices, cfg)

            # ── F1: Minimum activity ───────────────────────────────────
            if train_result.n_trades < cfg.min_trades:
                continue  # Skip this stock, not reject entirely

            # ── F2: Train drawdown ─────────────────────────────────────
            if train_result.max_drawdown_pct < -cfg.max_drawdown_pct:
                continue

            # ── Stage 2: Val backtest ──────────────────────────────────
            val_result = backtest_stock(func, val_features, val_prices, cfg)

            # ── F3: Both positive return ───────────────────────────────
            if train_result.total_return_pct <= 0.0:
                continue
            if val_result.total_return_pct <= 0.0:
                continue

            # ── F3b: Reject unrealistically high returns ───────────────
            if train_result.total_return_pct > MAX_RETURN_CAP:
                continue
            if val_result.total_return_pct > MAX_RETURN_CAP:
                continue

            # ── F4: Val Sharpe filter ──────────────────────────────────
            if val_result.sharpe_ratio < 0.0:
                continue

            # ── F5: Composite fitness for this stock ───────────────────
            # Use validation result for fitness (reduces overfitting)
            stock_fitness = composite_fitness(
                val_result, len(individual), cfg
            )

            # Consistency bonus: penalise large train-val gap
            train_fit = composite_fitness(
                train_result, len(individual), cfg
            )
            if train_fit > EPSILON:
                consistency = stock_fitness / train_fit
                # If val fitness < 30% of train → likely overfit
                if consistency < 0.3:
                    stock_fitness *= 0.5

            per_stock_fitness[symbol] = stock_fitness

        # ── Must pass at least 2 stocks (for multi-stock robustness) ───
        if len(per_stock_fitness) < 2:
            return FAIL

        # ── Multi-stock aggregation ────────────────────────────────────
        total_fitness = multi_stock_fitness(per_stock_fitness, cfg)

        # ── Complexity penalty bonus for very small trees ──────────────
        tree_len = len(individual)
        if tree_len < 10:
            total_fitness *= 0.5
        elif tree_len < 15:
            total_fitness *= 0.8

        # ── Cap ────────────────────────────────────────────────────────
        total_fitness = min(total_fitness, 500.0)

        return (total_fitness,)

    except Exception:
        return FAIL


# ═══════════════════════════════════════════════════════════════════════════
# PREPARE EVALUATION DATA
# ═══════════════════════════════════════════════════════════════════════════

def prepare_eval_data(
    df: "pd.DataFrame",
    feature_names: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix and close prices from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain feature columns and 'Close' column.
    feature_names : list of str
        Feature column names.

    Returns
    -------
    feature_matrix : np.ndarray
        Shape (n_bars, n_features).
    close_prices : np.ndarray
        Shape (n_bars,).
    """
    if feature_names is None:
        feature_names = list(V1_GP_FEATURES)

    # Ensure all feature columns exist
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    feature_matrix = df[feature_names].values.astype(np.float64)
    close_prices = df["close"].values.astype(np.float64)

    # Replace NaN with 0 in features
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_matrix, close_prices


def prepare_multi_stock_eval_data(
    full_df: "pd.DataFrame",
    symbols: List[str],
    feature_names: List[str] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare evaluation data for multiple stocks from a combined DataFrame.

    Parameters
    ----------
    full_df : pd.DataFrame
        Combined DataFrame with 'symbol' column.
    symbols : list of str
        Symbols to extract.
    feature_names : list of str
        Feature column names.

    Returns
    -------
    dict
        {symbol: (feature_matrix, close_prices)}
    """
    result = {}
    for sym in symbols:
        sym_df = full_df[full_df["symbol"] == sym].copy()
        if len(sym_df) < 100:
            continue
        try:
            features, prices = prepare_eval_data(sym_df, feature_names)
            result[sym] = (features, prices)
        except Exception:
            continue
    return result