"""
Evaluation [?] per-stock reporting and summary tables.

Spec Section 15: Evaluation Pipeline
- Evaluate best individual on train / val / test splits per stock
- Print formatted result tables
- Save results to CSV
"""

import os
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config import (
    GPConfig, DEFAULT_GP_CONFIG, V1_GP_FEATURES,
    TRADABLE_STOCKS, OUTPUT_DIR,
)
from backtester import (
    BacktestResult, backtest_stock, generate_signals_vectorised, run_backtest,
)
from fitness import prepare_eval_data, prepare_multi_stock_eval_data, composite_fitness
from gp_primitives import normalise_signal
from utils import get_logger, print_banner, print_table

logger = get_logger("evaluation")


# ===========================================================================
# EVALUATE ON ALL SPLITS & STOCKS
# ===========================================================================

def evaluate_best_individual(
    func,
    individual,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    symbols: List[str] = None,
    feature_names: List[str] = None,
    cfg: GPConfig = None,
) -> Dict[str, Dict[str, BacktestResult]]:
    """
    Evaluate a compiled GP function across all stocks and splits.

    Returns
    -------
    dict
        {symbol: {"train": BacktestResult, "val": ..., "test": ...}}
    """
    if symbols is None:
        symbols = list(TRADABLE_STOCKS)
    if feature_names is None:
        feature_names = list(V1_GP_FEATURES)
    if cfg is None:
        cfg = DEFAULT_GP_CONFIG

    results = {}

    for symbol in symbols:
        results[symbol] = {}

        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            sym_df = split_df[split_df["symbol"] == symbol]
            if len(sym_df) < 100:
                results[symbol][split_name] = BacktestResult(n_bars=len(sym_df))
                continue

            try:
                features, prices = prepare_eval_data(sym_df, feature_names)
                result = backtest_stock(func, features, prices, cfg)
                results[symbol][split_name] = result
            except Exception as exc:
                logger.warning(f"  [WARN][?] {symbol}/{split_name}: {exc}")
                results[symbol][split_name] = BacktestResult()

    return results


# ===========================================================================
# FORMATTED RESULT TABLES
# ===========================================================================

def print_evaluation_results(
    results: Dict[str, Dict[str, BacktestResult]],
    individual=None,
):
    """
    Print a comprehensive evaluation table.
    """
    print_banner("EVALUATION RESULTS")

    # -- Per-stock summary ----------------------------------------------
    headers = [
        "Symbol", "Split", "Return%", "Ann.Ret%", "Sharpe",
        "Sortino", "MaxDD%", "Trades", "WinRate%", "Trd/Day",
    ]

    rows = []
    for symbol in sorted(results.keys()):
        for split in ["train", "val", "test"]:
            r = results[symbol].get(split)
            if r is None:
                continue
            rows.append([
                symbol,
                split,
                f"{r.total_return_pct:>8.2f}",
                f"{r.annual_return_pct:>8.2f}",
                f"{r.sharpe_ratio:>7.3f}",
                f"{r.sortino_ratio:>7.3f}",
                f"{r.max_drawdown_pct:>8.2f}",
                f"{r.n_trades:>6}",
                f"{r.win_rate_pct:>7.1f}",
                f"{r.trades_per_day:>7.2f}",
            ])

    print_table(headers, rows)

    # -- Aggregated summary ---------------------------------------------
    print("\n" + "-" * 70)
    print("  AGGREGATED ACROSS STOCKS:")
    print("-" * 70)

    for split in ["train", "val", "test"]:
        returns = []
        sharpes = []
        for symbol in results:
            r = results[symbol].get(split)
            if r and r.n_bars > 0:
                returns.append(r.total_return_pct)
                sharpes.append(r.sharpe_ratio)

        if returns:
            print(
                f"  {split:>5}:  "
                f"avg_return={np.mean(returns):>7.2f}%  "
                f"std_return={np.std(returns):>7.2f}%  "
                f"avg_sharpe={np.mean(sharpes):>6.3f}  "
                f"n_stocks={len(returns)}"
            )

    # -- Tree info ------------------------------------------------------
    if individual is not None:
        print(f"\n  [?] Tree size: {len(individual)} nodes")
        tree_str = str(individual)
        if len(tree_str) > 200:
            tree_str = tree_str[:200] + "..."
        print(f"  [?] Formula: {tree_str}")

    print("=" * 70)


# ===========================================================================
# SAVE RESULTS TO CSV
# ===========================================================================

def save_evaluation_results(
    results: Dict[str, Dict[str, BacktestResult]],
    filepath: str = None,
):
    """Save evaluation results to CSV file."""
    if filepath is None:
        filepath = str(OUTPUT_DIR / "evaluation_results.csv")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    rows = []
    for symbol in sorted(results.keys()):
        for split in ["train", "val", "test"]:
            r = results[symbol].get(split)
            if r is None:
                continue
            row = {"symbol": symbol, "split": split}
            row.update(r.to_dict())
            rows.append(row)

    if not rows:
        logger.warning("  No results to save.")
        return

    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  [?] Results saved: {filepath}")


# ===========================================================================
# QUICK SUMMARY
# ===========================================================================

def quick_summary(
    results: Dict[str, Dict[str, BacktestResult]],
) -> pd.DataFrame:
    """
    Return a compact DataFrame summary of evaluation results.
    """
    rows = []
    for symbol in sorted(results.keys()):
        for split in ["train", "val", "test"]:
            r = results[symbol].get(split)
            if r is None:
                continue
            rows.append({
                "symbol": symbol,
                "split": split,
                **r.to_dict(),
            })

    return pd.DataFrame(rows)