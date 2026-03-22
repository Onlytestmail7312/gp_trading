"""
GP Engine [?] core evolution loop with checkpointing and logging.

Spec Section 12: Evolution Loop
- Chunked parallel evaluation
- Checkpoint/resume with dill
- Early stopping
- Generation statistics
- Elitism
"""

import os
import math
import time
import random
import copy
import traceback
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Callable

import numpy as np
import dill
from deap import tools, gp, creator

from config import GP_POPULATION, GP_GENERATIONS, GP_CROSSOVER, GP_MUTATION, GP_ELITE, GP_TOURNAMENT, GP_MIN_DEPTH, GP_MAX_DEPTH, GP_MAX_NODES, GP_EARLY_STOP, GP_CHUNK_SIZE, GP_WORKERS, DAILY_FEATURES, OUTPUT_DIR
from gp_individual import (
    setup_gp_toolbox,
    register_evaluate,
    compile_individual,
    create_population,
    seed_population,
    get_hall_of_fame,
    get_stats,
    _is_main_process,
)
from utils import get_logger, fmt_seconds, print_banner

logger = get_logger("gp_engine")


# ===========================================================================
# EARLY STOPPING
# ===========================================================================

class EarlyStopping:
    """Stop evolution when fitness plateaus."""

    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.counter = 0
        self.best_gen = 0

    def update(self, fitness: float, gen: int) -> bool:
        """
        Returns True if evolution should stop.
        """
        if fitness > self.best + self.min_delta:
            self.best = fitness
            self.counter = 0
            self.best_gen = gen
            return False

        self.counter += 1
        remaining = self.patience - self.counter

        if _is_main_process():
            logger.info(
                f"  [WAIT] No improvement for {self.counter} gen(s) "
                f"({remaining} more until early stop)"
            )

        if self.counter >= self.patience:
            if _is_main_process():
                logger.info(
                    f"  [?] Early stopping at gen {gen + 1}. "
                    f"Best was gen {self.best_gen + 1} "
                    f"(fitness={self.best:.4f})"
                )
            return True

        return False


# ===========================================================================
# CHECKPOINT MANAGEMENT
# ===========================================================================

def save_checkpoint(
    filepath: str,
    population: list,
    gen: int,
    chunk_idx: int,
    hof: tools.HallOfFame,
    gen_stats: list,
    rng_state=None,
):
    """Save evolution state to disk with dill."""
    cp = {
        "pop": population,
        "gen": gen,
        "idx": chunk_idx,
        "hof": hof,
        "stats": gen_stats,
        "rnd_state": rng_state or random.getstate(),
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "wb") as f:
        dill.dump(cp, f)
    if _is_main_process():
        logger.info(f"  [?] Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str) -> Optional[dict]:
    """Load evolution state from disk. Returns None on failure."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            cp = dill.load(f)
        if _is_main_process():
            logger.info(
                f"  [?]  Checkpoint loaded: gen={cp['gen']+1}, "
                f"chunk={cp['idx']}, saved={cp.get('timestamp', 'unknown')}"
            )
        return cp
    except Exception as exc:
        if _is_main_process():
            logger.warning(f"  [WARN][?]  Checkpoint load failed: {exc}")
        return None


def save_best_model(filepath: str, individual, pset):
    """Save the best individual to disk."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    payload = {
        "individual": individual,
        "tree_str": str(individual),
        "tree_size": len(individual),
        "fitness": individual.fitness.values[0] if individual.fitness.valid else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(filepath, "wb") as f:
        dill.dump(payload, f)
    if _is_main_process():
        logger.info(f"  [?] Best model saved: {filepath}")


def load_best_model(filepath: str) -> Optional[dict]:
    """Load a saved best individual. Returns None on failure."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            return dill.load(f)
    except Exception as exc:
        logger.warning(f"  [WARN][?]  Model load failed: {exc}")
        return None


# ===========================================================================
# GENERATION STATS
# ===========================================================================

def _compute_gen_stats(population: list, gen: int, elapsed: float) -> dict:
    """Compute statistics for one generation."""
    FITNESS_INVALID_THRESHOLD = -99.0
    valid_fits = [
        ind.fitness.values[0]
        for ind in population
        if ind.fitness.valid and ind.fitness.values[0] > FITNESS_INVALID_THRESHOLD
    ]

    if not valid_fits:
        return {
            "gen": gen,
            "best": -100.0,
            "avg": -100.0,
            "std": 0.0,
            "min": -100.0,
            "n_valid": 0,
            "n_total": len(population),
            "elapsed_s": elapsed,
        }

    return {
        "gen": gen,
        "best": max(valid_fits),
        "avg": np.mean(valid_fits),
        "std": np.std(valid_fits),
        "min": min(valid_fits),
        "n_valid": len(valid_fits),
        "n_total": len(population),
        "elapsed_s": elapsed,
    }


def _log_gen_stats(stats: dict, cfg):
    """Print generation stats in a formatted way."""
    if not _is_main_process():
        return

    logger.info(
        f"  [?] Gen {stats['gen']+1:>3}/{GP_GENERATIONS}: "
        f"best={stats['best']:>8.3f}  "
        f"avg={stats['avg']:>8.3f}  "
        f"std={stats['std']:>6.3f}  "
        f"valid={stats['n_valid']}/{stats['n_total']}  "
        f"time={fmt_seconds(stats['elapsed_s'])}"
    )


# ===========================================================================
# SAVE EVOLUTION STATS TO CSV
# ===========================================================================

def save_stats_csv(filepath: str, all_stats: list):
    """Save generation stats to CSV."""
    import csv

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    if not all_stats:
        return

    fieldnames = list(all_stats[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

    if _is_main_process():
        logger.info(f"  [?] Stats saved: {filepath}")


# ===========================================================================
# CORE EVOLUTION LOOP
# ===========================================================================

def run_evolution(
    toolbox,
    pset,
    cfg = None,
    evaluate_func: Callable = None,
    evaluate_kwargs: dict = None,
    seed_model_path: str = None,
    use_multiprocessing: bool = True,
) -> Tuple[object, list, tools.HallOfFame]:
    """
    Run the full GP evolution loop.

    Parameters
    ----------
    toolbox : deap.base.Toolbox
        Configured toolbox (from setup_gp_toolbox).
    pset : deap.gp.PrimitiveSet
        Primitive set.
    cfg : GPConfig
        GP configuration.
    evaluate_func : callable
        Fitness evaluation function. Signature:
            evaluate(individual, toolbox=..., **kwargs) -> (float,)
    evaluate_kwargs : dict
        Extra kwargs passed to evaluate_func.
    seed_model_path : str
        Path to a saved model to seed population from.
    use_multiprocessing : bool
        Whether to use multiprocessing for evaluation.

    Returns
    -------
    best_individual : GP individual
        Best evolved individual.
    generation_stats : list of dict
        Per-generation statistics.
    hof : tools.HallOfFame
        Hall of fame.
    """
    if cfg is None:
        cfg = None
    if evaluate_kwargs is None:
        evaluate_kwargs = {}

    phase_start = time.time()
    pool = None

    if _is_main_process():
        print_banner("GP EVOLUTION")
        logger.info(f"  Population: {GP_POPULATION}")
        logger.info(f"  Generations: {GP_GENERATIONS}")
        logger.info(f"  Tree depth: {GP_MIN_DEPTH}[?]{GP_MAX_DEPTH}")
        logger.info(f"  Max nodes: {GP_MAX_NODES}")
        logger.info(f"  Crossover: {GP_CROSSOVER}")
        logger.info(f"  Mutation: {GP_MUTATION}")
        logger.info(f"  Elite: {GP_ELITE}")
        logger.info(f"  Tournament: {GP_TOURNAMENT}")

    # -- Register evaluation function -----------------------------------
    if evaluate_func is not None:
        # Set global state for multiprocessing workers
        from fitness import init_worker
        _train_data = evaluate_kwargs.get("train_data")
        _val_data   = evaluate_kwargs.get("val_data")
        init_worker(toolbox, _train_data, _val_data)

        register_evaluate(toolbox, evaluate_func, **evaluate_kwargs)

    # -- Multiprocessing setup ------------------------------------------
    if use_multiprocessing:
        import multiprocessing as mp
        n_workers = GP_WORKERS if GP_WORKERS > 0 else mp.cpu_count()
        n_workers = min(n_workers, mp.cpu_count())
        try:
            pool = mp.Pool(processes=n_workers)
            toolbox.register("map", pool.map)
            if _is_main_process():
                logger.info(f"  Workers: {n_workers} (multiprocessing)")
        except Exception as exc:
            logger.warning(f"  [WARN][?]  Multiprocessing failed ({exc}), using serial")
            pool = None
    else:
        if _is_main_process():
            logger.info("  Workers: 1 (serial)")

    try:
        best, gen_stats, hof = _evolution_loop(
            toolbox, pset, cfg, seed_model_path
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    phase_elapsed = time.time() - phase_start
    if _is_main_process():
        logger.info(f"\n  [OK] Evolution complete in {fmt_seconds(phase_elapsed)}")
        if best is not None and best.fitness.valid:
            logger.info(f"  [?] Best fitness: {best.fitness.values[0]:.4f}")
            logger.info(f"  [?] Tree size: {len(best)} nodes")
            logger.info(f"  [?] Formula: {str(best)[:200]}")

    return best, gen_stats, hof


# ===========================================================================
# INTERNAL EVOLUTION LOOP
# ===========================================================================

def _evolution_loop(
    toolbox,
    pset,
    cfg,
    seed_model_path: str = None,
) -> Tuple[object, list, tools.HallOfFame]:
    """Internal evolution loop with checkpoint/resume."""

    # -- Try resuming from checkpoint -----------------------------------
    cp = load_checkpoint(str(OUTPUT_DIR / "checkpoint.pkl"))

    if cp is not None:
        population = cp["pop"]
        start_gen = cp["gen"]
        start_chunk = cp["idx"]
        hof = cp["hof"]
        gen_stats = cp.get("stats", [])
        random.setstate(cp["rnd_state"])
        if _is_main_process():
            logger.info(
                f"  [?]  Resuming from gen {start_gen+1}, chunk {start_chunk}"
            )
    else:
        # -- Fresh start ------------------------------------------------
        random.seed(42)
        np.random.seed(42)

        if _is_main_process():
            logger.info(
                f"  [?] Initialising population ({GP_POPULATION:,})..."
            )
        population = create_population(toolbox, GP_POPULATION)

        # Seed with oscillating signal templates
        from deap import gp as deap_gp, creator as deap_creator
        from gp_individual import get_pset
        _pset = get_pset()
        seed_exprs = [
            'sub(rsi_14, half)',
            'sub(close_vs_sma20, zero)',
            'macd_hist',
            'ret_5d',
            'sub(close_vs_sma50, zero)',
            'sub(bb_pct, half)',
            'ret_10d',
            'close_vs_sma20',
            'sub(rsi_14, zero)',
            'ret_1d',
        ]
        if _pset is not None:
            seeded = 0
            for expr_str in seed_exprs:
                try:
                    expr = deap_gp.PrimitiveTree.from_string(expr_str, _pset)
                    ind = creator.Individual(expr)
                    del ind.fitness.values
                    population[seeded] = ind
                    seeded += 1
                except Exception as e:
                    pass
            if seeded > 0:
                logger.info(f"  Seeded {seeded} oscillating signal templates")
        hof = get_hall_of_fame(maxsize=GP_ELITE)
        gen_stats = []
        start_gen = 0
        start_chunk = 0

        # -- Seed from previous model ----------------------------------
        if seed_model_path and os.path.exists(seed_model_path):
            try:
                saved = load_best_model(seed_model_path)
                if saved and "individual" in saved:
                    population = seed_population(
                        toolbox, population, saved["individual"], n_seeds=10
                    )
                    if _is_main_process():
                        logger.info(
                            f"  [?] Seeded 10 individuals from '{seed_model_path}'"
                        )
            except Exception as exc:
                if _is_main_process():
                    logger.warning(f"  [WARN][?]  Seed failed: {exc}")

    # -- Early stopping -------------------------------------------------
    early_stopper = EarlyStopping(patience=GP_EARLY_STOP, min_delta=0.01)
    for row in gen_stats:
        early_stopper.update(row["best"], row["gen"])

    # -- Main loop ------------------------------------------------------
    for gen in range(start_gen, GP_GENERATIONS):

        gen_start = time.time()

        if _is_main_process():
            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"  [?] GENERATION {gen + 1} / {GP_GENERATIONS}"
            )
            logger.info(f"{'=' * 60}")

        # -- Find unevaluated individuals -------------------------------
        invalid_ind = [ind for ind in population if not ind.fitness.valid]

        if _is_main_process():
            logger.info(f"  Evaluating {len(invalid_ind):,} individuals...")

        if len(invalid_ind) > 0:
            # Chunked evaluation with progress tracking
            total_chunks = math.ceil(len(invalid_ind) / GP_CHUNK_SIZE)
            resume_from = start_chunk if gen == start_gen else 0
            chunk_times = []

            for i in range(resume_from, len(invalid_ind), GP_CHUNK_SIZE):
                chunk = invalid_ind[i: i + GP_CHUNK_SIZE]
                chunk_num = (i // GP_CHUNK_SIZE) + 1
                chunk_start = time.time()

                if _is_main_process():
                    end_idx = min(i + GP_CHUNK_SIZE, len(invalid_ind))
                    logger.info(
                        f"    [WAIT] Chunk {chunk_num}/{total_chunks} "
                        f"[{i}->{end_idx}]..."
                    )

                # Parallel evaluation with generation-aware fitness
                try:
                    # Store current generation on toolbox for fitness function to access
                    toolbox.current_generation = gen
                    fitnesses = list(map(toolbox.evaluate, chunk))
                    for ind, fit in zip(chunk, fitnesses):
                        ind.fitness.values = fit
                except Exception as exc:
                    logger.error(f"    [FAIL] Eval error: {exc}")
                    for ind in chunk:
                        if not ind.fitness.valid:
                            ind.fitness.values = (-100.0,)

                chunk_elapsed = time.time() - chunk_start
                chunk_times.append(chunk_elapsed)

                if _is_main_process() and chunk_times:
                    avg_ct = sum(chunk_times) / len(chunk_times)
                    remaining = total_chunks - len(chunk_times)
                    eta = fmt_seconds(avg_ct * remaining)
                    logger.info(
                        f"      done in {fmt_seconds(chunk_elapsed)}, "
                        f"ETA: {eta}"
                    )

                # -- Intra-generation checkpoint ------------------------
                if (chunk_num % max(1, total_chunks // 3)) == 0:
                    save_checkpoint(
                        str(OUTPUT_DIR / "checkpoint.pkl"),
                        population, gen, i + GP_CHUNK_SIZE,
                        hof, gen_stats,
                    )

        # -- Update Hall of Fame ----------------------------------------
        hof.update(population)

        # -- Generation statistics --------------------------------------
        gen_elapsed = time.time() - gen_start
        stats = _compute_gen_stats(population, gen, gen_elapsed)
        gen_stats.append(stats)
        _log_gen_stats(stats, cfg)

        # -- Early stopping check ---------------------------------------
        if early_stopper.update(stats["best"], gen):
            break

        # -- Checkpoint at generation boundary --------------------------
        if (gen + 1) % 5 == 0:
            save_checkpoint(
                str(OUTPUT_DIR / "checkpoint.pkl"), population, gen + 1, 0, hof, gen_stats
            )
            save_stats_csv(str(OUTPUT_DIR / "evolution_stats.csv"), gen_stats)

        # -- Save best model --------------------------------------------
        if hof.items:
            save_best_model(str(OUTPUT_DIR / "best_model.pkl"), hof.items[0], pset)

        # -- Selection + Genetic Operators (skip on last gen) -----------
        if gen < GP_GENERATIONS - 1:
            population = _breed_next_generation(
                toolbox, population, hof, cfg
            )

        # Reset start_chunk after first generation
        if gen == start_gen:
            start_chunk = 0

    # -- Final saves ----------------------------------------------------
    save_checkpoint(
        str(OUTPUT_DIR / "checkpoint.pkl"), population, GP_GENERATIONS, 0, hof, gen_stats
    )
    save_stats_csv(str(OUTPUT_DIR / "evolution_stats.csv"), gen_stats)

    if hof.items:
        save_best_model(str(OUTPUT_DIR / "best_model.pkl"), hof.items[0], pset)

    best = hof.items[0] if hof.items else None
    return best, gen_stats, hof


# ===========================================================================
# BREED NEXT GENERATION
# ===========================================================================

def _breed_next_generation(
    toolbox,
    population: list,
    hof: tools.HallOfFame,
    cfg,
) -> list:
    """
    Create the next generation via:
    1. Elitism [?] keep top N individuals unchanged
    2. Tournament selection
    3. Crossover + Mutation
    """
    pop_size = len(population)

    # -- Elitism --------------------------------------------------------
    elite = []
    if GP_ELITE > 0 and hof.items:
        for ind in hof.items[:GP_ELITE]:
            elite.append(toolbox.clone(ind))

    # -- Tournament selection for breeding pool -------------------------
    n_offspring = pop_size - len(elite)
    offspring = toolbox.select(population, n_offspring)
    offspring = [toolbox.clone(ind) for ind in offspring]

    # -- Crossover ------------------------------------------------------
    for i in range(1, len(offspring), 2):
        if random.random() < GP_CROSSOVER:
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i]
            )
            del offspring[i - 1].fitness.values
            del offspring[i].fitness.values

    # -- Mutation -------------------------------------------------------
    for i in range(len(offspring)):
        if random.random() < GP_MUTATION:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    # -- Combine elite + offspring --------------------------------------
    next_gen = elite + offspring

    # Ensure population size is maintained
    while len(next_gen) < pop_size:
        next_gen.append(toolbox.individual())
    next_gen = next_gen[:pop_size]

    return next_gen