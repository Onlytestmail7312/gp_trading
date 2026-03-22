"""
Main Training Script — run GP evolution on multi-stock data.

Usage:
    python -m gp_system_complete.main_train_gp [--config CONFIG_OVERRIDES]

Workflow:
1. Load pre-computed feature dataset (parquet or DuckDB)
2. Split into train / val / test by time
3. Prepare per-stock feature matrices
4. Run GP evolution with multi-stock fitness
5. Save best model + stats
6. Evaluate best individual on all splits
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from .config import (
    GPConfig, DEFAULT_GP_CONFIG, V1_GP_FEATURES,
    TRADABLE_STOCKS, OUTPUT_DIR, FEATURE_PARQUET,
)
from .dataset_builder import load_feature_dataset, split_by_time
from .gp_individual import setup_gp_toolbox, register_evaluate, compile_individual
from .gp_engine import run_evolution, save_best_model, load_best_model
from .fitness import (
    evaluate_individual,
    prepare_eval_data,
    prepare_multi_stock_eval_data,
)
from .evaluation import (
    evaluate_best_individual,
    print_evaluation_results,
    save_evaluation_results,
)
from .regime_analysis import run_regime_analysis, save_regime_results
from .utils import get_logger, print_banner, ensure_output_dirs, Timer

logger = get_logger("main_train")


def main(cfg: GPConfig = None):
    """Run the full GP training pipeline."""

    if cfg is None:
        cfg = DEFAULT_GP_CONFIG

    ensure_output_dirs()
    print_banner("UNIFIED MULTI-STOCK GP TRAINING")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Output: {OUTPUT_DIR}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load features
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n📂 Loading feature dataset...")

    with Timer("Load features"):
        df = load_feature_dataset()

    if df is None or df.empty:
        logger.error("❌ No feature data found. Run main_build_features.py first.")
        sys.exit(1)

    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"  Symbols: {sorted(df['symbol'].unique())}")
    logger.info(f"  Date range: {df.index.min()} → {df.index.max()}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Time-based split
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n📅 Splitting by time...")

    train_df, val_df, test_df = split_by_time(df)

    logger.info(f"  Train: {len(train_df):>10,} rows")
    logger.info(f"  Val:   {len(val_df):>10,} rows")
    logger.info(f"  Test:  {len(test_df):>10,} rows")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Prepare per-stock evaluation data
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n🔧 Preparing per-stock data...")

    feature_names = list(V1_GP_FEATURES)
    symbols = [s for s in TRADABLE_STOCKS if s in df["symbol"].unique()]

    if len(symbols) < 2:
        logger.error(f"❌ Need at least 2 stocks, found: {symbols}")
        sys.exit(1)

    logger.info(f"  Tradable stocks: {symbols}")

    train_stock_data = prepare_multi_stock_eval_data(train_df, symbols, feature_names)
    val_stock_data = prepare_multi_stock_eval_data(val_df, symbols, feature_names)

    # Feature matrices for signal diversity check
    train_feature_matrices = {
        sym: data[0] for sym, data in train_stock_data.items()
    }

    logger.info(f"  Train stocks ready: {list(train_stock_data.keys())}")
    logger.info(f"  Val stocks ready: {list(val_stock_data.keys())}")

    for sym in symbols:
        if sym in train_stock_data:
            n_train = train_stock_data[sym][0].shape[0]
            n_val = val_stock_data.get(sym, (np.array([]),))[0].shape[0]
            logger.info(f"    {sym}: train={n_train:,} val={n_val:,}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Setup GP toolbox
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n🧬 Setting up GP toolbox...")

    toolbox, pset = setup_gp_toolbox(cfg, feature_names)

    # Register evaluation function with data
    register_evaluate(
        toolbox,
        evaluate_individual,
        train_data=train_stock_data,
        val_data=val_stock_data,
        train_feature_matrices=train_feature_matrices,
        cfg=cfg,
    )

    logger.info(f"  Primitives: {len(pset.primitives[pset.ret])} functions")
    logger.info(f"  Terminals: {len(pset.terminals[pset.ret])} inputs")
    logger.info(f"  Features: {len(feature_names)}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Run evolution
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n🚀 Starting GP evolution...")

    with Timer("GP Evolution"):
        best_individual, gen_stats, hof = run_evolution(
            toolbox=toolbox,
            pset=pset,
            cfg=cfg,
            use_multiprocessing=(cfg.n_workers != 1),
        )

    if best_individual is None:
        logger.error("❌ Evolution produced no valid individual.")
        sys.exit(1)

    logger.info(f"\n🏆 Best individual:")
    logger.info(f"  Fitness: {best_individual.fitness.values[0]:.4f}")
    logger.info(f"  Size: {len(best_individual)} nodes")
    logger.info(f"  Formula: {str(best_individual)[:300]}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Evaluate on all splits
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n📊 Evaluating best individual on all splits...")

    func = toolbox.compile(expr=best_individual)

    with Timer("Evaluation"):
        eval_results = evaluate_best_individual(
            func=func,
            individual=best_individual,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            symbols=symbols,
            feature_names=feature_names,
            cfg=cfg,
        )

    print_evaluation_results(eval_results, individual=best_individual)
    save_evaluation_results(eval_results)

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Regime analysis
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n🔬 Running regime analysis...")

    with Timer("Regime Analysis"):
        regime_results = run_regime_analysis(
            func=func,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            symbols=symbols,
            feature_names=feature_names,
            cfg=cfg,
        )

    # Save regime results for test split
    if "test" in regime_results:
        save_regime_results(regime_results["test"])

    # ══════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════
    print_banner("TRAINING COMPLETE")
    logger.info(f"  Best model: {cfg.best_model_file}")
    logger.info(f"  Stats: {cfg.stats_file}")
    logger.info(f"  Results: {OUTPUT_DIR / 'evaluation_results.csv'}")
    logger.info(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments for config overrides."""
    parser = argparse.ArgumentParser(
        description="Unified Multi-Stock GP Training"
    )
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-model", type=str, default=None,
                        help="Path to previous best model for seeding")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = GPConfig()

    if args.pop_size:
        cfg.pop_size = args.pop_size
    if args.generations:
        cfg.n_generations = args.generations
    if args.workers:
        cfg.n_workers = args.workers
    if args.seed:
        cfg.random_seed = args.seed

    main(cfg)