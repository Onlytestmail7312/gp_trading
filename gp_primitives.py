"""
GP Primitives [?] safe mathematical functions for tree evaluation.

Spec Section 8: GP Representation
Primitives: add, sub, mul, safe_div, abs, min, max, neg
Terminals: V1_GP_FEATURES columns + ephemeral constants in [-1, 1]
"""

import math
import operator
import random
import numpy as np
from typing import List

from deap import gp

from config import DAILY_FEATURES as V1_GP_FEATURES, EPSILON


# ===========================================================================
# SAFE PRIMITIVE FUNCTIONS
# ===========================================================================

def safe_div(a: float, b: float) -> float:
    """Protected division [?] returns 0.0 when denominator is near-zero."""
    try:
        if abs(b) < EPSILON:
            return 0.0
        result = a / b
        if math.isinf(result) or math.isnan(result):
            return 0.0
        return result
    except (ZeroDivisionError, OverflowError, ValueError):
        return 0.0


def safe_neg(a: float) -> float:
    """Negation with NaN guard."""
    try:
        result = -a
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (OverflowError, ValueError):
        return 0.0


def safe_abs(a: float) -> float:
    """Absolute value with NaN guard."""
    try:
        result = abs(a)
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (OverflowError, ValueError):
        return 0.0


def safe_exp(a: float) -> float:
    """Exponential with overflow protection."""
    try:
        if a > 10:  # Prevent overflow
            return 1.0
        result = math.exp(a)
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (OverflowError, ValueError):
        return 0.0


def safe_log(a: float) -> float:
    """Natural log with domain protection."""
    try:
        if a <= 0:
            return 0.0
        result = math.log(a)
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (ValueError, OverflowError):
        return 0.0


def safe_sqrt(a: float) -> float:
    """Square root with domain protection."""
    try:
        if a < 0:
            return 0.0
        result = math.sqrt(a)
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (ValueError, OverflowError):
        return 0.0


def safe_add(a: float, b: float) -> float:
    """Addition with overflow guard."""
    try:
        result = a + b
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (OverflowError, ValueError):
        return 0.0


def safe_sub(a: float, b: float) -> float:
    """Subtraction with overflow guard."""
    try:
        result = a - b
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (OverflowError, ValueError):
        return 0.0


def safe_mul(a: float, b: float) -> float:
    """Multiplication with overflow guard."""
    try:
        result = a * b
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result
    except (OverflowError, ValueError):
        return 0.0


def safe_max(a: float, b: float) -> float:
    """Max with NaN guard."""
    try:
        if math.isnan(a) and math.isnan(b):
            return 0.0
        if math.isnan(a):
            return b
        if math.isnan(b):
            return a
        return max(a, b)
    except (TypeError, ValueError):
        return 0.0


def safe_min(a: float, b: float) -> float:
    """Min with NaN guard."""
    try:
        if math.isnan(a) and math.isnan(b):
            return 0.0
        if math.isnan(a):
            return b
        if math.isnan(b):
            return a
        return min(a, b)
    except (TypeError, ValueError):
        return 0.0


# ===========================================================================
# NORMALISE SIGNAL [?] tanh scaling for GP output
# ===========================================================================

def normalise_signal(raw: float) -> float:
    """
    Convert raw GP tree output -> normalised signal in [-1, 1].
    Uses tanh to smoothly squash any real number.
    +1 = max long, -1 = max short, 0 = neutral.
    """
    try:
        if math.isnan(raw) or math.isinf(raw):
            return 0.0
        return math.tanh(raw)
    except (OverflowError, ValueError):
        return 0.0


# ===========================================================================
# EPHEMERAL CONSTANT GENERATOR
# ===========================================================================

def ephemeral_constant() -> float:
    """Generate random constant in [-1.0, 1.0] for GP tree terminals."""
    return round(random.uniform(-1.0, 1.0), 4)


# ===========================================================================
# PRIMITIVE SET BUILDER
# ===========================================================================

def build_primitive_set(feature_names: List[str] = None) -> gp.PrimitiveSetTyped:
    """
    Build a DEAP PrimitiveSet with:
      - 8 safe primitives (add, sub, mul, safe_div, abs, min, max, neg)
      - Named terminals for each feature column
      - Ephemeral constant terminal in [-1, 1]

    Parameters
    ----------
    feature_names : list of str, optional
        Feature column names to use as terminals. 
        Defaults to V1_GP_FEATURES.

    Returns
    -------
    pset : gp.PrimitiveSet
        Configured DEAP primitive set.
    """
    if feature_names is None:
        feature_names = list(V1_GP_FEATURES)

    n_features = len(feature_names)

    # Create untyped primitive set with N input arguments
    pset = gp.PrimitiveSet("GPStrategy", n_features)

    # -- Register primitives --------------------------------------------
    # Binary (arity=2)
    pset.addPrimitive(safe_add, 2, name="add")
    pset.addPrimitive(safe_sub, 2, name="sub")
    pset.addPrimitive(safe_mul, 2, name="mul")
    pset.addPrimitive(safe_div, 2, name="div")
    pset.addPrimitive(safe_max, 2, name="max")
    pset.addPrimitive(safe_min, 2, name="min")

    # Unary (arity=1)
    # NOTE: abs() removed intentionally -- forces GP to generate
    # both positive (long) and negative (short) signals naturally
    pset.addPrimitive(safe_neg, 1, name="neg")
    pset.addPrimitive(safe_exp, 1, name="exp")
    pset.addPrimitive(safe_log, 1, name="log")
    pset.addPrimitive(safe_sqrt, 1, name="sqrt")

    # -- Rename arguments to feature names ------------------------------
    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f"ARG{i}": name})

    # -- Fixed constants ------------------------------------------------
    pset.addTerminal(0.0, name="zero")
    pset.addTerminal(1.0, name="one")
    pset.addTerminal(-1.0, name="neg_one")
    pset.addTerminal(0.5, name="half")
    pset.addTerminal(0.1, name="tenth")
    pset.addTerminal(0.01, name="hundredth")

    # -- Ephemeral constant ---------------------------------------------
    pset.addEphemeralConstant("rand_const", ephemeral_constant)

    return pset


# ===========================================================================
# SIGNAL DIVERSITY CHECK
# ===========================================================================

MIN_SIGNAL_STD = 0.01  # Minimum std of normalised signals to be valid

def check_signal_diversity(
    func,
    feature_matrix: np.ndarray,
    n_samples: int = 500,
) -> bool:
    """
    Check whether a compiled GP function produces diverse signals.
    
    Samples `n_samples` rows from the feature matrix, evaluates the GP
    function, normalises via tanh, and checks std > MIN_SIGNAL_STD.
    
    Parameters
    ----------
    func : callable
        Compiled GP function (takes N float args).
    feature_matrix : np.ndarray
        Shape (n_rows, n_features).
    n_samples : int
        Number of random rows to sample.
    
    Returns
    -------
    bool
        True if signal is diverse enough, False if nearly constant.
    """
    n_rows = feature_matrix.shape[0]
    if n_rows == 0:
        return False

    indices = np.random.choice(n_rows, size=min(n_samples, n_rows), replace=False)
    signals = []

    for idx in indices:
        try:
            row = feature_matrix[idx]
            raw = float(func(*row))
            signals.append(normalise_signal(raw))
        except Exception:
            signals.append(0.0)

    if len(signals) < 10:
        return False

    std = np.std(signals)
    return std >= MIN_SIGNAL_STD


# ===========================================================================
# TREE UTILITIES
# ===========================================================================

def tree_depth(individual) -> int:
    """Compute the depth of a GP tree."""
    try:
        return individual.height
    except AttributeError:
        return 0


def tree_size(individual) -> int:
    """Number of nodes in a GP tree."""
    return len(individual)


def tree_to_string(individual, pset) -> str:
    """Convert GP tree to human-readable string."""
    try:
        return str(individual)
    except Exception:
        return "<unprintable>"