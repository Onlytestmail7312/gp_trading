import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

"""
Main Evaluation Script [?] evaluate a saved GP model on all data.

Usage:
    python -m gp_system_complete.main_evaluate_gp [--model PATH]

Workflow:
1. Load saved best model
2. Load feature dataset
3. Split into train / val / test
4. Evaluate on all splits per stock
5. Run regime analysis
6. Print and save results
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd

from config import DAILY_FEATURES, ALL_SYMBOLS, OUTPUT_DIR
# Aliases for backward compatibility
V1_GP_FEATURES = DAILY_FEATURES
TRADABLE_STOCKS = ALL_SYMBOLS
from dataset_builder import load_features, split_dataset
from gp_individual import setup_gp_toolbox, compile_individual
from gp_engine import load_best_model
from evaluation import (
    evaluate_best_individual,
    print_evaluation_results,
    save_evaluation_results,
    quick_summary,
)
from regime_analysis import (
    run_regime_analysis,
    save_regime_results,
    print_regime_distribution,
    add_regime_column,
)
from utils import get_logger, print_banner, ensure_output_dirs, Timer

logger = get_logger("main_evaluate")


def main(model_path: str = None, cfg=None):
    """Run the full evaluation pipeline on a saved model."""

    if cfg is None:
        cfg = {}
    if model_path is None:
        model_path = cfg.best_model_file

    ensure_output_dirs()
    print_banner("UNIFIED MULTI-STOCK GP EVALUATION")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Model: {model_path}")

    # Set up GP toolbox to restore creator classes
    feature_names = list(V1_GP_FEATURES)
    toolbox, pset = setup_gp_toolbox(cfg, feature_names)

    # ==================================================================
    # STEP 1: Load saved model
    # ==================================================================
    logger.info("\n[?] Loading saved model...")

    saved = load_best_model(model_path)
    if saved is None:
        logger.error(f"[FAIL] Could not load model from: {model_path}")
        sys.exit(1)

    individual = saved["individual"]
    logger.info(f"  Tree size: {saved.get('tree_size', len(individual))} nodes")
    logger.info(f"  Fitness: {saved.get('fitness', 'unknown')}")
    logger.info(f"  Saved at: {saved.get('timestamp', 'unknown')}")

    tree_str = saved.get("tree_str", str(individual))
    if len(tree_str) > 300:
        tree_str = tree_str[:300] + "..."
    logger.info(f"  Formula: {tree_str}")

    # ==================================================================
    # STEP 2: Load features
    # ==================================================================
    logger.info("\n[?] Loading feature dataset...")

    with Timer("Load features"):
        df = load_feature_dataset()

    if df is None or df.empty:
        logger.error("[FAIL] No feature data found. Run main_build_features.py first.")
        sys.exit(1)

    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Symbols: {sorted(df['symbol'].unique())}")

    # ==================================================================
    # STEP 3: Split by time
    # ==================================================================
    logger.info("\n[?] Splitting by time...")

    train_df, val_df, test_df = split_by_time(df)

    logger.info(f"  Train: {len(train_df):>10,} rows")
    logger.info(f"  Val:   {len(val_df):>10,} rows")
    logger.info(f"  Test:  {len(test_df):>10,} rows")

    # ==================================================================
    # STEP 4: Setup toolbox and compile
    # ==================================================================
    logger.info("\n[?] Compiling GP tree...")

    feature_names = list(V1_GP_FEATURES)
    toolbox, pset = setup_gp_toolbox(cfg, feature_names)
    func = toolbox.compile(expr=individual)

    symbols = [s for s in TRADABLE_STOCKS if s in df["symbol"].unique()]
    logger.info(f"  Stocks: {symbols}")

    # ==================================================================
    # STEP 5: Evaluate on all splits
    # ==================================================================
    logger.info("\n[?] Evaluating on all splits...")

    with Timer("Full evaluation"):
        eval_results = evaluate_best_individual(
            func=func,
            individual=individual,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            symbols=symbols,
            feature_names=feature_names,
            cfg=cfg,
        )

    print_evaluation_results(eval_results, individual=individual)

    eval_path = str(OUTPUT_DIR / "evaluation_results.csv")
    save_evaluation_results(eval_results, eval_path)

    # -- Quick summary DataFrame ----------------------------------------
    summary_df = quick_summary(eval_results)
    summary_path = str(OUTPUT_DIR / "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  Summary saved: {summary_path}")

    # ==================================================================
    # STEP 6: Regime analysis
    # ==================================================================
    logger.info("\n[?] Running regime analysis...")

    with Timer("Regime analysis"):
        regime_results = run_regime_analysis(
            func=func,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            symbols=symbols,
            feature_names=feature_names,
            cfg=cfg,
        )

    # Save all regime results
    for split_name, split_results in regime_results.items():
        regime_path = str(OUTPUT_DIR / f"regime_{split_name}.csv")
        save_regime_results(split_results, regime_path)

    # ==================================================================
    # STEP 7: Test set focus
    # ==================================================================
    if test_df is not None and len(test_df) > 0:
        print_banner("TEST SET FOCUS (Out-of-Sample)")

        for symbol in symbols:
            if symbol in eval_results and "test" in eval_results[symbol]:
                r = eval_results[symbol]["test"]
                logger.info(
                    f"  {symbol:>12}: "
                    f"return={r.total_return_pct:>7.2f}%  "
                    f"sharpe={r.sharpe_ratio:>6.3f}  "
                    f"dd={r.max_drawdown_pct:>7.2f}%  "
                    f"trades={r.n_trades}"
                )

    # ==================================================================
    # DONE
    # ==================================================================
    print_banner("EVALUATION COMPLETE")
    logger.info(f"  Results: {eval_path}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return eval_results, regime_results


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved GP model"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to saved best_model.pkl"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(model_path=args.model)