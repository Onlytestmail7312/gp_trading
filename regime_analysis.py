"""
Regime Analysis [?] bull/bear/flat slicing and per-regime evaluation.

Spec Section 14: Regime-Aware Evaluation (Post-Training)
- Label each bar by trend_50d_return thresholds
- Slice data into BULL / BEAR / FLAT regimes
- Evaluate the unified model on each regime slice
- Report per-regime metrics
- This is evaluation-only [?] the model trains on ALL data unified
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config import (
    GPConfig, DEFAULT_GP_CONFIG, V1_GP_FEATURES,
    REGIME_BULL_THRESHOLD, REGIME_BEAR_THRESHOLD,
    TRADABLE_STOCKS, OUTPUT_DIR,
)
from backtester import BacktestResult, backtest_stock
from fitness import prepare_eval_data
from utils import get_logger, print_banner, print_table

logger = get_logger("regime_analysis")


# ===========================================================================
# REGIME LABELLING
# ===========================================================================

REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_FLAT = "FLAT"

ALL_REGIMES = [REGIME_BULL, REGIME_FLAT, REGIME_BEAR]


def label_regimes(
    df: pd.DataFrame,
    bull_threshold: float = None,
    bear_threshold: float = None,
) -> pd.Series:
    """
    Assign regime labels based on trend_50d_return column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'trend_50d_return' column.
    bull_threshold : float
        Return > this -> BULL. Default: +5%
    bear_threshold : float
        Return < this -> BEAR. Default: -5%

    Returns
    -------
    pd.Series
        Regime labels: "BULL", "BEAR", or "FLAT".
    """
    if bull_threshold is None:
        bull_threshold = REGIME_BULL_THRESHOLD
    if bear_threshold is None:
        bear_threshold = REGIME_BEAR_THRESHOLD

    if "trend_50d_return" not in df.columns:
        logger.warning("  [WARN][?] 'trend_50d_return' not in DataFrame [?] all FLAT")
        return pd.Series(REGIME_FLAT, index=df.index)

    trend = df["trend_50d_return"]

    conditions = [
        trend > bull_threshold,
        trend < bear_threshold,
    ]
    choices = [REGIME_BULL, REGIME_BEAR]

    return pd.Series(
        np.select(conditions, choices, default=REGIME_FLAT),
        index=df.index,
    )


def add_regime_column(
    df: pd.DataFrame,
    bull_threshold: float = None,
    bear_threshold: float = None,
) -> pd.DataFrame:
    """Add 'regime' column to DataFrame (modifies in place and returns)."""
    df["regime"] = label_regimes(df, bull_threshold, bear_threshold)
    return df


# ===========================================================================
# REGIME STATISTICS
# ===========================================================================

def regime_distribution(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Compute regime distribution statistics.

    Returns
    -------
    dict
        {regime: {"count": int, "pct": float, "avg_return": float}}
    """
    if "regime" not in df.columns:
        df = add_regime_column(df)

    total = len(df)
    stats = {}

    for regime in ALL_REGIMES:
        mask = df["regime"] == regime
        count = int(mask.sum())
        pct = (count / max(total, 1)) * 100.0

        avg_ret = 0.0
        if "trend_50d_return" in df.columns and count > 0:
            avg_ret = float(df.loc[mask, "trend_50d_return"].mean()) * 100.0

        stats[regime] = {
            "count": count,
            "pct": round(pct, 1),
            "avg_trend_return_pct": round(avg_ret, 2),
        }

    return stats


def print_regime_distribution(df: pd.DataFrame, title: str = ""):
    """Print regime distribution table."""
    stats = regime_distribution(df)

    header = f"REGIME DISTRIBUTION"
    if title:
        header += f" [?] {title}"
    print_banner(header)

    headers = ["Regime", "Bars", "Pct%", "Avg Trend Return%"]
    rows = []
    for regime in ALL_REGIMES:
        s = stats[regime]
        rows.append([
            regime,
            f"{s['count']:>10,}",
            f"{s['pct']:>6.1f}",
            f"{s['avg_trend_return_pct']:>8.2f}",
        ])

    print_table(headers, rows)


# ===========================================================================
# PER-REGIME EVALUATION
# ===========================================================================

def evaluate_by_regime(
    func,
    df: pd.DataFrame,
    symbols: List[str] = None,
    feature_names: List[str] = None,
    cfg: GPConfig = None,
    split_name: str = "",
) -> Dict[str, Dict[str, BacktestResult]]:
    """
    Evaluate a GP function on each regime slice, per stock.

    Parameters
    ----------
    func : callable
        Compiled GP function.
    df : pd.DataFrame
        Data with features, 'symbol', 'Close', and 'regime' columns.
    symbols : list of str
        Stocks to evaluate.
    feature_names : list of str
        Feature column names.
    cfg : GPConfig
        Configuration.
    split_name : str
        Label for logging (e.g. "test").

    Returns
    -------
    dict
        {regime: {symbol: BacktestResult}}
    """
    if symbols is None:
        symbols = list(TRADABLE_STOCKS)
    if feature_names is None:
        feature_names = list(V1_GP_FEATURES)
    if cfg is None:
        cfg = DEFAULT_GP_CONFIG

    if "regime" not in df.columns:
        df = add_regime_column(df)

    results = {}

    for regime in ALL_REGIMES:
        results[regime] = {}
        regime_df = df[df["regime"] == regime]

        if len(regime_df) < 100:
            logger.info(f"  [?]  {regime}: too few bars ({len(regime_df)}), skipping")
            continue

        for symbol in symbols:
            sym_df = regime_df[regime_df["symbol"] == symbol]
            if len(sym_df) < 50:
                continue

            try:
                features, prices = prepare_eval_data(sym_df, feature_names)
                result = backtest_stock(func, features, prices, cfg)
                results[regime][symbol] = result
            except Exception as exc:
                logger.warning(f"  [WARN][?] {regime}/{symbol}: {exc}")

    return results


# ===========================================================================
# PRINT REGIME EVALUATION
# ===========================================================================

def print_regime_evaluation(
    results: Dict[str, Dict[str, BacktestResult]],
    title: str = "REGIME EVALUATION",
):
    """Print per-regime, per-stock results in a formatted table."""
    print_banner(title)

    headers = [
        "Regime", "Symbol", "Return%", "Sharpe", "MaxDD%",
        "Trades", "WinRate%", "Trd/Day",
    ]
    rows = []

    for regime in ALL_REGIMES:
        regime_results = results.get(regime, {})
        if not regime_results:
            rows.append([regime, "[?]", "[?]", "[?]", "[?]", "[?]", "[?]", "[?]"])
            continue

        for symbol in sorted(regime_results.keys()):
            r = regime_results[symbol]
            rows.append([
                regime,
                symbol,
                f"{r.total_return_pct:>8.2f}",
                f"{r.sharpe_ratio:>7.3f}",
                f"{r.max_drawdown_pct:>8.2f}",
                f"{r.n_trades:>6}",
                f"{r.win_rate_pct:>7.1f}",
                f"{r.trades_per_day:>7.2f}",
            ])

    print_table(headers, rows)

    # -- Aggregated per-regime summary ----------------------------------
    print("\n" + "-" * 60)
    print("  REGIME SUMMARY (averaged across stocks):")
    print("-" * 60)

    for regime in ALL_REGIMES:
        regime_results = results.get(regime, {})
        if not regime_results:
            print(f"  {regime:>5}:  (no data)")
            continue

        returns = [r.total_return_pct for r in regime_results.values()]
        sharpes = [r.sharpe_ratio for r in regime_results.values()]

        print(
            f"  {regime:>5}:  "
            f"avg_return={np.mean(returns):>7.2f}%  "
            f"std_return={np.std(returns):>7.2f}%  "
            f"avg_sharpe={np.mean(sharpes):>6.3f}  "
            f"n_stocks={len(returns)}"
        )

    print("=" * 60)


# ===========================================================================
# SAVE REGIME RESULTS
# ===========================================================================

def save_regime_results(
    results: Dict[str, Dict[str, BacktestResult]],
    filepath: str = None,
):
    """Save per-regime evaluation results to CSV."""
    import csv

    if filepath is None:
        filepath = str(OUTPUT_DIR / "regime_evaluation.csv")

    import os
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    rows = []
    for regime in ALL_REGIMES:
        regime_results = results.get(regime, {})
        for symbol in sorted(regime_results.keys()):
            r = regime_results[symbol]
            row = {"regime": regime, "symbol": symbol}
            row.update(r.to_dict())
            rows.append(row)

    if not rows:
        logger.warning("  No regime results to save.")
        return

    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  [?] Regime results saved: {filepath}")


# ===========================================================================
# FULL REGIME ANALYSIS PIPELINE
# ===========================================================================

def run_regime_analysis(
    func,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    symbols: List[str] = None,
    feature_names: List[str] = None,
    cfg: GPConfig = None,
) -> Dict[str, Dict[str, Dict[str, BacktestResult]]]:
    """
    Full regime analysis: label + evaluate on each split.

    Returns
    -------
    dict
        {split_name: {regime: {symbol: BacktestResult}}}
    """
    if symbols is None:
        symbols = list(TRADABLE_STOCKS)
    if feature_names is None:
        feature_names = list(V1_GP_FEATURES)
    if cfg is None:
        cfg = DEFAULT_GP_CONFIG

    all_results = {}

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if len(split_df) == 0:
            continue

        # Add regime labels
        split_df = add_regime_column(split_df.copy())

        # Print distribution
        print_regime_distribution(split_df, title=f"{split_name.upper()}")

        # Evaluate
        regime_results = evaluate_by_regime(
            func, split_df, symbols, feature_names, cfg, split_name
        )
        all_results[split_name] = regime_results

        # Print results
        print_regime_evaluation(
            regime_results,
            title=f"REGIME EVAL [?] {split_name.upper()}",
        )

    return all_results