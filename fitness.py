"""
Fitness — composite fitness scoring with multi-stock aggregation.

SPEC SECTION 9: FITNESS FUNCTION
  composite_fitness = (
      w_sharpe * sharpe
    + w_sortino * sortino
    + w_annual_return * annual_return_pct / 100
    - w_max_drawdown * |max_drawdown_pct| / 100
    - w_trades_per_day * max(0, trades_per_day - 5) / 10
    - complexity_penalty_weight * tree_size / max_nodes
  )

SPEC SECTION 11: MULTI-STOCK AGGREGATION
  fitness_total = mean(stock_fitness) - lambda * std(stock_fitness)

REVISED APPROACH (after evolution failure diagnosis):
  - Graduated penalties instead of hard rejection
  - Phased difficulty: filters get stricter over generations
  - Minimum viable individuals can survive early generations
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


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE FITNESS (single stock)
# ═══════════════════════════════════════════════════════════════════════════════

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

    # ── Component scores ─────────────────────────────────────────────────────
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

    # ── Composite ─────────────────────────────────────────────────────────────
    fitness = (
        sharpe_score
        + sortino_score
        + return_score
        - dd_penalty
        - trade_penalty
        - complexity_penalty
    )

    return fitness


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-STOCK AGGREGATED FITNESS
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# GRADUATED PENALTY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tree_penalty(tree_size: int, min_size: int = 7) -> float:
    """
    Graduated penalty for tree complexity.
    
    Instead of hard rejection, apply a penalty that increases as tree
    gets smaller than minimum.
    
    Returns a multiplier: 1.0 = no penalty, 0.0 = maximum penalty
    """
    if tree_size >= min_size:
        return 1.0
    
    # Linear interpolation from 0.1 (size=1) to 1.0 (size=min_size)
    penalty = 0.1 + 0.9 * (tree_size - 1) / (min_size - 1)
    return max(0.1, penalty)


def compute_drawdown_penalty(dd_pct: float, threshold: float = 50.0) -> float:
    """
    Graduated penalty for drawdown.
    
    Instead of hard rejection when dd < -threshold, apply increasing penalty.
    
    Returns a penalty multiplier for the fitness.
    """
    if dd_pct >= -threshold:
        return 1.0
    
    # Exponential penalty for excessive drawdown
    excess = abs(dd_pct) - threshold
    penalty = math.exp(-excess / 20.0)  # Decays by ~half every 14% extra DD
    return max(0.1, penalty)


def compute_consistency_penalty(train_fit: float, val_fit: float) -> float:
    """
    Penalty for train/val inconsistency (overfitting indicator).
    
    Instead of hard rejection, reduce fitness when val << train.
    """
    if train_fit <= EPSILON:
        return 0.5
    
    ratio = val_fit / train_fit
    
    if ratio >= 0.5:
        return 1.0  # Good consistency
    elif ratio >= 0.3:
        return 0.8  # Moderate overfitting
    elif ratio >= 0.1:
        return 0.5  # Significant overfitting
    else:
        return 0.2  # Severe overfitting


# ═══════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION PIPELINE (called by GP engine)
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum tree size to avoid degenerate formulas (relaxed for early generations)
MIN_TREE_LENGTH_BASE = 5

# Reject unrealistic returns (likely overfitting)
MAX_RETURN_CAP = 500.0  # percent


def evaluate_individual(
    individual,
    toolbox=None,
    train_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    val_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    train_feature_matrices: Dict[str, np.ndarray] = None,
    cfg: GPConfig = None,
    generation: int = 0,
) -> Tuple[float, ...]:
    """
    Full fitness evaluation pipeline for one GP individual.

    REVISED APPROACH: Graduated penalties with phased difficulty.
    
    The evaluation applies soft penalties instead of hard rejection,
    allowing selection pressure to operate even on imperfect individuals.
    
    Phased difficulty:
    - Generations 0-4: Only basic checks, allow any tree with trades
    - Generations 5-9: Add drawdown and consistency checks
    - Generations 10+: Full multi-stock validation
    
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
    generation : int
        Current generation number (for phased difficulty).

    Returns
    -------
    tuple of float
        (fitness,) — single-objective.
    """
    BASE_FAIL = -50.0  # Better than -100, allows some selection pressure

    if cfg is None:
        cfg = DEFAULT_GP_CONFIG
    
    # Get generation from toolbox if not explicitly passed
    if generation == 0 and toolbox is not None and hasattr(toolbox, 'current_generation'):
        generation = toolbox.current_generation

    try:
        # ── Compile tree ───────────────────────────────────────────────────────
        if toolbox is None:
            return (BASE_FAIL,)
        func = toolbox.compile(expr=individual)

        tree_size = len(individual)
        
        # ── Phase-dependent minimum tree size ─────────────────────────────────
        # Early generations: allow smaller trees
        min_tree = MIN_TREE_LENGTH_BASE if generation < 5 else 7
        tree_penalty = compute_tree_penalty(tree_size, min_tree)
        
        # ── Signal diversity check (soft) ─────────────────────────────────────
        diversity_ok = True
        if train_feature_matrices and generation >= 3:
            first_sym = next(iter(train_feature_matrices))
            fm = train_feature_matrices[first_sym]
            diversity_ok = check_signal_diversity(func, fm, n_samples=500)
        
        diversity_penalty = 1.0 if diversity_ok else 0.7

        # ── Per-stock evaluation ───────────────────────────────────────────────
        if train_data is None or val_data is None:
            return (BASE_FAIL,)

        per_stock_fitness = {}
        per_stock_details = {}  # For debugging
        
        # Count how many stocks pass various filters
        stocks_with_trades = 0
        stocks_positive_train = 0
        stocks_positive_val = 0
        stocks_positive_both = 0

        for symbol in train_data:
            if symbol not in val_data:
                continue

            train_features, train_prices = train_data[symbol]
            val_features, val_prices = val_data[symbol]

            # ── Stage 1: Train backtest ───────────────────────────────────────
            train_result = backtest_stock(func, train_features, train_prices, cfg)

            # Skip if no trades at all (dead individual)
            if train_result.n_trades < 2:
                continue
            
            stocks_with_trades += 1

            # ── Stage 2: Val backtest ───────────────────────────────────────────
            val_result = backtest_stock(func, val_features, val_prices, cfg)

            # Track statistics
            if train_result.total_return_pct > 0:
                stocks_positive_train += 1
            if val_result.total_return_pct > 0:
                stocks_positive_val += 1
            if train_result.total_return_pct > 0 and val_result.total_return_pct > 0:
                stocks_positive_both += 1

            # ── Compute fitness with graduated penalties ───────────────────────
            train_fit = composite_fitness(train_result, tree_size, cfg)
            val_fit = composite_fitness(val_result, tree_size, cfg)
            
            # Apply drawdown penalty
            dd_penalty = compute_drawdown_penalty(train_result.max_drawdown_pct, 
                                                   cfg.max_drawdown_pct)
            
            # Apply consistency penalty
            consistency_penalty = compute_consistency_penalty(train_fit, val_fit)
            
            # Combined penalty
            total_penalty = tree_penalty * diversity_penalty * dd_penalty * consistency_penalty
            
            # Final stock fitness
            stock_fitness = val_fit * total_penalty
            
            # Bonus for positive returns on both splits
            if train_result.total_return_pct > 0 and val_result.total_return_pct > 0:
                stock_fitness += 0.5  # Small bonus
            
            # Cap unrealistic returns
            if train_result.total_return_pct > MAX_RETURN_CAP:
                stock_fitness *= 0.5
            if val_result.total_return_pct > MAX_RETURN_CAP:
                stock_fitness *= 0.5

            per_stock_fitness[symbol] = stock_fitness
            per_stock_details[symbol] = {
                'train_return': train_result.total_return_pct,
                'val_return': val_result.total_return_pct,
                'train_sharpe': train_result.sharpe_ratio,
                'val_sharpe': val_result.sharpe_ratio,
                'n_trades': train_result.n_trades,
                'fitness': stock_fitness,
            }

        # ── Multi-stock aggregation with phased requirements ───────────────────
        n_stocks_required = 1 if generation < 10 else 2
        
        if len(per_stock_fitness) < n_stocks_required:
            # Not enough stocks passed - return partial fitness
            if len(per_stock_fitness) == 0:
                return (BASE_FAIL - 10,)
            else:
                # Partial credit for at least trying
                partial_fit = np.mean(list(per_stock_fitness.values())) * 0.5
                return (partial_fit,)

        # ── Multi-stock aggregation ─────────────────────────────────────────────
        total_fitness = multi_stock_fitness(per_stock_fitness, cfg)

        # ── Generation-based bonus ─────────────────────────────────────────────
        # Encourage diversity in early generations
        if generation < 5:
            # Bonus for evaluating multiple stocks (even if not all positive)
            total_fitness += 0.1 * len(per_stock_fitness)
        
        # ── Cap ─────────────────────────────────────────────────────────────────
        total_fitness = min(total_fitness, 500.0)

        return (total_fitness,)

    except Exception as e:
        return (BASE_FAIL,)


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARE EVALUATION DATA
# ═══════════════════════════════════════════════════════════════════════════════

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
    close_prices = df["Close"].values.astype(np.float64)

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