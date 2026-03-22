"""
GP Individual — DEAP creator/toolbox setup for GP individuals.

Spec Section 8: GP Representation
- Tree depth: 2–8 levels
- Max nodes: 50
- Fitness: single-objective maximisation
"""

import random
import multiprocessing
import os
from typing import Optional, List

from deap import base, creator, gp, tools

from .config import GPConfig, DEFAULT_GP_CONFIG, V1_GP_FEATURES
from .gp_primitives import build_primitive_set


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE
# ═══════════════════════════════════════════════════════════════════════════

# These are module-level so they can be pickled for multiprocessing
_pset = None
_toolbox = None
_creator_initialized = False


def _is_main_process() -> bool:
    """Check if we're in the main process (not a worker)."""
    return multiprocessing.current_process().name == "MainProcess"


# ═══════════════════════════════════════════════════════════════════════════
# CREATOR SETUP
# ═══════════════════════════════════════════════════════════════════════════

def _setup_creator():
    """
    Create DEAP FitnessMax and Individual classes.
    Safe to call multiple times — only creates once.
    """
    global _creator_initialized
    if _creator_initialized:
        return

    # Single-objective maximisation: weights=(1.0,)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create(
            "Individual",
            gp.PrimitiveTree,
            fitness=creator.FitnessMax,
            pset=None,  # Will be set per-tree
        )

    _creator_initialized = True


# ═══════════════════════════════════════════════════════════════════════════
# TOOLBOX SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_gp_toolbox(
    cfg: GPConfig = None,
    feature_names: List[str] = None,
) -> tuple:
    """
    Build and configure the DEAP toolbox for GP evolution.

    Parameters
    ----------
    cfg : GPConfig
        GP configuration. Defaults to DEFAULT_GP_CONFIG.
    feature_names : list of str
        Feature names for terminals. Defaults to V1_GP_FEATURES.

    Returns
    -------
    toolbox : deap.base.Toolbox
        Configured DEAP toolbox.
    pset : deap.gp.PrimitiveSet
        Configured primitive set.
    """
    global _pset, _toolbox

    if cfg is None:
        cfg = DEFAULT_GP_CONFIG
    if feature_names is None:
        feature_names = list(V1_GP_FEATURES)

    # ── Setup creator ──────────────────────────────────────────────────
    _setup_creator()

    # ── Build primitive set ────────────────────────────────────────────
    pset = build_primitive_set(feature_names)
    _pset = pset

    # ── Build toolbox ──────────────────────────────────────────────────
    toolbox = base.Toolbox()

    # Expression generator (ramped half-and-half)
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=cfg.init_min_depth,
        max_=cfg.init_max_depth,
    )

    # Individual from expression
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        toolbox.expr,
    )

    # Population generator
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,
    )

    # Compile GP tree → callable function
    toolbox.register("compile", gp.compile, pset=pset)

    # ── Genetic operators ──────────────────────────────────────────────

    # Selection: tournament
    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=cfg.tournament_size,
    )

    # Crossover: one-point
    toolbox.register("mate", gp.cxOnePoint)

    # Mutation: uniform with small sub-trees
    toolbox.register(
        "expr_mut",
        gp.genFull,
        min_=0,
        max_=2,
        pset=pset,
    )
    toolbox.register(
        "mutate",
        gp.mutUniform,
        expr=toolbox.expr_mut,
        pset=pset,
    )

    # ── Tree size limits (bloat control) ───────────────────────────────
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=len, max_value=cfg.max_tree_nodes),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=len, max_value=cfg.max_tree_nodes),
    )

    # Also limit depth
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator_height, max_value=cfg.max_tree_depth),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator_height, max_value=cfg.max_tree_depth),
    )

    _toolbox = toolbox
    return toolbox, pset


def operator_height(individual) -> int:
    """Get the height of a GP individual (for staticLimit decorator)."""
    try:
        return individual.height
    except AttributeError:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# REGISTER EVALUATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def register_evaluate(toolbox, evaluate_func, **kwargs):
    """
    Register the fitness evaluation function in the toolbox.

    Parameters
    ----------
    toolbox : deap.base.Toolbox
        The DEAP toolbox.
    evaluate_func : callable
        Function with signature: evaluate(individual, **kwargs) -> (float,)
    **kwargs
        Additional keyword arguments passed to evaluate_func.
    """
    if hasattr(toolbox, "evaluate"):
        toolbox.unregister("evaluate")

    toolbox.register("evaluate", evaluate_func, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_pset():
    """Get the module-level primitive set."""
    global _pset
    return _pset


def get_toolbox():
    """Get the module-level toolbox."""
    global _toolbox
    return _toolbox


def clone_individual(toolbox, individual):
    """Deep clone a GP individual."""
    return toolbox.clone(individual)


def compile_individual(toolbox, individual):
    """Compile a GP tree into a callable function."""
    return toolbox.compile(expr=individual)


def create_population(toolbox, n: int) -> list:
    """Create a fresh random population."""
    return toolbox.population(n=n)


def seed_population(
    toolbox,
    population: list,
    seed_individual,
    n_seeds: int = 10,
) -> list:
    """
    Seed a population with mutated copies of a seed individual.

    Parameters
    ----------
    toolbox : deap.base.Toolbox
    population : list
        The population to seed into.
    seed_individual
        The individual to clone and mutate.
    n_seeds : int
        Number of slots to replace with seeds.

    Returns
    -------
    list
        Modified population.
    """
    n_seeds = min(n_seeds, len(population))
    for i in range(n_seeds):
        clone = toolbox.clone(seed_individual)
        clone, = toolbox.mutate(clone)
        del clone.fitness.values
        population[i] = clone
    return population


def get_hall_of_fame(maxsize: int = 10) -> tools.HallOfFame:
    """Create a Hall of Fame to track best individuals."""
    return tools.HallOfFame(maxsize)


def get_stats() -> tools.Statistics:
    """Create a Statistics object for tracking evolution metrics."""
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(v for v in x if v > -99) / max(1, sum(1 for v in x if v > -99)))
    stats.register("max", max)
    stats.register("min", min)
    stats.register("valid", lambda x: sum(1 for v in x if v > -99))
    return stats