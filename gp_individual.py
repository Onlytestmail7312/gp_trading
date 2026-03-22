"""
gp_individual.py -- DEAP creator/toolbox setup for GP individuals.
"""

import random
import multiprocessing
import os
from typing import Optional, List

from deap import base, creator, gp, tools

from config import (
    GP_MIN_DEPTH, GP_MAX_DEPTH, GP_MAX_NODES,
    GP_TOURNAMENT, DAILY_FEATURES,
)
from gp_primitives import build_primitive_set


_pset = None
_toolbox = None
_creator_initialized = False


def _is_main_process() -> bool:
    return multiprocessing.current_process().name == "MainProcess"


def _setup_creator():
    global _creator_initialized
    if _creator_initialized:
        return
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create(
            "Individual",
            gp.PrimitiveTree,
            fitness=creator.FitnessMax,
            pset=None,
        )
    _creator_initialized = True


def operator_height(individual) -> int:
    try:
        return individual.height
    except AttributeError:
        return 0


def setup_gp_toolbox(
    feature_names: List[str] = None,
) -> tuple:
    global _pset, _toolbox

    if feature_names is None:
        feature_names = list(DAILY_FEATURES)

    _setup_creator()

    pset = build_primitive_set(feature_names)
    _pset = pset

    toolbox = base.Toolbox()

    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=GP_MIN_DEPTH,
        max_=GP_MAX_DEPTH,
    )

    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        toolbox.expr,
    )

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,
    )

    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=GP_TOURNAMENT,
    )

    toolbox.register("mate", gp.cxOnePoint)

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

    toolbox.decorate(
        "mate",
        gp.staticLimit(key=len, max_value=GP_MAX_NODES),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=len, max_value=GP_MAX_NODES),
    )

    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator_height, max_value=GP_MAX_DEPTH),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator_height, max_value=GP_MAX_DEPTH),
    )

    _toolbox = toolbox
    return toolbox, pset


def register_evaluate(toolbox, evaluate_func, **kwargs):
    if hasattr(toolbox, "evaluate"):
        toolbox.unregister("evaluate")
    toolbox.register("evaluate", evaluate_func, **kwargs)


def get_pset():
    global _pset
    return _pset


def get_toolbox():
    global _toolbox
    return _toolbox


def clone_individual(toolbox, individual):
    return toolbox.clone(individual)


def compile_individual(toolbox, individual):
    return toolbox.compile(expr=individual)


def create_population(toolbox, n: int) -> list:
    return toolbox.population(n=n)


def seed_population(toolbox, population, seed_individual, n_seeds=10):
    n_seeds = min(n_seeds, len(population))
    for i in range(n_seeds):
        clone = toolbox.clone(seed_individual)
        clone, = toolbox.mutate(clone)
        del clone.fitness.values
        population[i] = clone
    return population


def get_hall_of_fame(maxsize=10):
    return tools.HallOfFame(maxsize)


def get_stats():
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(v for v in x if v > -99) / max(1, sum(1 for v in x if v > -99)))
    stats.register("max", max)
    stats.register("min", min)
    stats.register("valid", lambda x: sum(1 for v in x if v > -99))
    return stats
