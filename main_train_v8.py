"""
main_train_v8.py
================
V8 main training script.
Runs 3 separate GP evolutions, one per regime:
    1. BULL_QUIET
    2. SIDEWAYS_BEAR
    3. VOLATILE

Each evolution produces a specialized formula.
Results saved to:
    gp_output/v8/formula_BULL_QUIET.pkl
    gp_output/v8/formula_SIDEWAYS_BEAR.pkl
    gp_output/v8/formula_VOLATILE.pkl
"""

import os
import sys
import time
import random
import argparse
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from typing import Dict
import dill
import dill as _dill

# =============================================================================
# SETUP
# =============================================================================
sys.path.insert(0, '.')

from config_v8 import (
    OUTPUT_DIR, PARQUET_FILE, ALL_SYMBOLS, DAILY_FEATURES,
    REGIMES, GP_POPULATION, GP_GENERATIONS, GP_CROSSOVER,
    GP_MUTATION, GP_ELITE, GP_TOURNAMENT, GP_MAX_DEPTH,
    GP_MAX_NODES, GP_EARLY_STOP, GP_MIN_DEPTH,
    GP_WORKERS, GP_CHUNK_SIZE,
)
from dataset_builder_v8 import (
    load_features, split_dataset, prepare_regime_data,
    get_regime_distribution,
)
from gp_individual import setup_gp_toolbox
from fitness_v8 import init_worker, evaluate_worker
from utils import get_logger, fmt_seconds

log = get_logger('gp_v8')

# Output directory for V8
V8_OUTPUT = OUTPUT_DIR / 'v8'
V8_OUTPUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EVOLUTION ENGINE (per regime)
# =============================================================================
def run_regime_evolution(
    regime:      str,
    train_data:  Dict,
    val_data:    Dict,
    toolbox,
    seed:        int = 42,
) -> Dict:
    """
    Run GP evolution for one regime.
    Returns best individual + stats.
    """
    from deap import tools, algorithms
    import copy

    log.info(f"\n{'='*60}")
    log.info(f"  EVOLVING REGIME: {regime}")
    log.info(f"{'='*60}")
    log.info(f"  Population : {GP_POPULATION}")
    log.info(f"  Generations: {GP_GENERATIONS}")
    log.info(f"  Train stocks: {list(train_data.keys())}")
    log.info(f"  Val stocks  : {list(val_data.keys())}")

    random.seed(seed)
    np.random.seed(seed)

    # Initialize population
    log.info(f"\n  Initialising population ({GP_POPULATION:,})...")
    population = toolbox.population(n=GP_POPULATION)

    # Stats
    stats = tools.Statistics(lambda ind: ind.fitness.values[0]
                              if ind.fitness.valid else -50.0)
    stats.register('best', np.max)
    stats.register('avg',  np.mean)
    stats.register('std',  np.std)

    hall_of_fame = tools.HallOfFame(GP_ELITE)

    best_fitness    = -999.0
    no_improve_gens = 0
    gen_stats       = []
    start_time      = time.time()

    # ==========================================================================
    # EVOLUTION LOOP
    # ==========================================================================
    for gen in range(1, GP_GENERATIONS + 1):
        gen_start = time.time()

        log.info(f"\n  {'='*50}")
        log.info(f"  GENERATION {gen}/{GP_GENERATIONS}  [{regime}]")
        log.info(f"  {'='*50}")

        # Evaluate unevaluated individuals
        invalid = [ind for ind in population if not ind.fitness.valid]
        log.info(f"  Evaluating {len(invalid):,} individuals...")

        # Chunked multiprocessing evaluation
        with mp.Pool(
            processes=GP_WORKERS,
            initializer=init_worker,
            initargs=(None, train_data, val_data, regime),
        ) as pool:
            # Serialize individuals with dill
            invalid_dill = [dill.dumps(ind) for ind in invalid]
            chunks = [
                invalid_dill[i:i+GP_CHUNK_SIZE]
                for i in range(0, len(invalid_dill), GP_CHUNK_SIZE)
            ]
            evaluated = []
            for ci, chunk in enumerate(chunks, 1):
                chunk_start = time.time()
                result = pool.map(evaluate_worker, chunk)
                evaluated.extend(result)
                chunk_time = time.time() - chunk_start
                remaining  = (len(chunks) - ci) * chunk_time
                log.info(f"    Chunk {ci}/{len(chunks)} "
                         f"done in {chunk_time:.0f}s  "
                         f"ETA: {fmt_seconds(remaining)}")

        # Update fitness
        for ind, fit_vals in zip(invalid, evaluated):
            ind.fitness.values = fit_vals

        # Update hall of fame
        hall_of_fame.update(population)

        # Stats
        record   = stats.compile(population)
        best_fit = record['best']
        avg_fit  = record['avg']
        std_fit  = record['std']
        gen_time = time.time() - gen_start

        log.info(f"\n  Gen {gen:3d}/{GP_GENERATIONS}: "
                 f"best={best_fit:8.3f}  "
                 f"avg={avg_fit:8.3f}  "
                 f"std={std_fit:7.3f}  "
                 f"time={fmt_seconds(gen_time)}")

        gen_stats.append({
            'regime': regime,
            'gen':    gen,
            'best':   best_fit,
            'avg':    avg_fit,
            'std':    std_fit,
            'time':   gen_time,
        })

        # Early stop check
        if best_fit > best_fitness + 0.001:
            best_fitness    = best_fit
            no_improve_gens = 0
        else:
            no_improve_gens += 1
            log.info(f"  No improvement for {no_improve_gens} gen(s) "
                     f"({GP_EARLY_STOP - no_improve_gens} more until stop)")

        if no_improve_gens >= GP_EARLY_STOP:
            log.info(f"  Early stop at gen {gen}")
            break

        # Selection + reproduction
        offspring = toolbox.select(population, len(population) - GP_ELITE)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < GP_CROSSOVER:
                offspring[i], offspring[i+1] = toolbox.mate(
                    offspring[i], offspring[i+1]
                )
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # Mutation
        for i in range(len(offspring)):
            if random.random() < GP_MUTATION:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Enforce size limits
        for ind in offspring:
            if len(ind) > GP_MAX_NODES:
                del ind.fitness.values

        # Next generation = elite + offspring
        elite      = list(map(toolbox.clone, hall_of_fame.items))
        population = elite + offspring

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    total_time = time.time() - start_time
    best_ind   = hall_of_fame[0]
    best_func  = toolbox.compile(expr=best_ind)

    log.info(f"\n  {'='*50}")
    log.info(f"  REGIME {regime} COMPLETE")
    log.info(f"  {'='*50}")
    log.info(f"  Best fitness : {best_fitness:.4f}")
    log.info(f"  Tree size    : {len(best_ind)} nodes")
    log.info(f"  Formula      : {str(best_ind)}")
    log.info(f"  Total time   : {fmt_seconds(total_time)}")

    return {
        'regime':     regime,
        'individual': best_ind,
        'fitness':    best_fitness,
        'tree_str':   str(best_ind),
        'tree_size':  len(best_ind),
        'gen_stats':  gen_stats,
        'total_time': total_time,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='V8 Regime-Aware GP Training')
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--regime', type=str, default='all',
                        help='Which regime to train: all / BULL_QUIET / SIDEWAYS_BEAR / VOLATILE')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  V8 REGIME-AWARE GP TRAINING")
    log.info("=" * 60)
    log.info(f"  Seed   : {args.seed}")
    log.info(f"  Regime : {args.regime}")
    log.info(f"  Output : {V8_OUTPUT}")

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    log.info("\n[1] Loading data...")
    df       = load_features(PARQUET_FILE)
    train_df, val_df, test_df = split_dataset(df)
    combined = pd.concat([train_df, val_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    log.info("\n[2] Preparing regime data...")
    regime_data = prepare_regime_data(combined)

    log.info("\n[3] Setting up GP toolbox...")
    toolbox, pset = setup_gp_toolbox(DAILY_FEATURES)

    # ==========================================================================
    # DETERMINE WHICH REGIMES TO TRAIN
    # ==========================================================================
    if args.regime == 'all':
        regimes_to_train = REGIMES
    else:
        if args.regime not in REGIMES:
            log.error(f"Unknown regime: {args.regime}. Choose from {REGIMES}")
            sys.exit(1)
        regimes_to_train = [args.regime]

    log.info(f"\n[4] Training regimes: {regimes_to_train}")

    # ==========================================================================
    # EVOLVE EACH REGIME
    # ==========================================================================
    all_results = {}
    total_start = time.time()

    for regime in regimes_to_train:
        if regime not in regime_data:
            log.warning(f"  Regime {regime} not in data -- skipping")
            continue

        data   = regime_data[regime]
        result = run_regime_evolution(
            regime     = regime,
            train_data = data['train'],
            val_data   = data['val'],
            toolbox    = toolbox,
            seed       = args.seed,
        )
        all_results[regime] = result

        # Save formula
        out_path = V8_OUTPUT / f"formula_{regime}.pkl"
        with open(out_path, 'wb') as f:
            dill.dump(result, f)
        log.info(f"  Saved: {out_path}")

        # Save gen stats
        stats_df = pd.DataFrame(result['gen_stats'])
        stats_path = V8_OUTPUT / f"stats_{regime}.csv"
        stats_df.to_csv(stats_path, index=False)
        log.info(f"  Stats: {stats_path}")

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    total_time = time.time() - total_start
    log.info(f"\n{'='*60}")
    log.info(f"  V8 TRAINING COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Total time: {fmt_seconds(total_time)}")
    log.info(f"\n  Results per regime:")
    for regime, result in all_results.items():
        log.info(f"    {regime}:")
        log.info(f"      Fitness : {result['fitness']:.4f}")
        log.info(f"      Size    : {result['tree_size']} nodes")
        log.info(f"      Formula : {result['tree_str']}")

    log.info(f"\n  Saved formulas:")
    for regime in all_results:
        log.info(f"    {V8_OUTPUT}/formula_{regime}.pkl")

    log.info(f"\n  Next: run backtest_v8.py to evaluate combined strategy")


if __name__ == '__main__':
    mp.freeze_support()
    main()
