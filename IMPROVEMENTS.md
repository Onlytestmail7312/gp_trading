# Code Review & Improvement Suggestions

A thorough review of the `gp_trading` codebase. Issues are grouped by priority and category.

---

## 🔴 Critical Bugs & Correctness Issues

### 1. Broken `creator` Reference in `gp_engine.py` (NameError at Runtime)

**File:** `gp_engine.py` — `_evolution_loop()`, around the seed expressions block.

```python
# CURRENT (BUG) — `creator` is never imported in gp_engine.py
ind = creator.Individual(expr)
```

**Problem:** `creator` from `deap` is imported in `gp_individual.py` but never imported in `gp_engine.py`. This will raise a `NameError` on every fresh evolution run that tries to seed templates.

**Fix:**
```python
# At the top of gp_engine.py, add:
from deap import creator

# Or import it alongside the existing deap imports:
from deap import tools, gp, creator
```

---

### 2. `equity_curve` Computation is Incorrect in `backtester.py`

**File:** `backtester.py` — `_compute_metrics()`

```python
# CURRENT (BUG)
equity = np.ones(n_days)
for t in trades:
    for day in range(t.exit_day, n_days):   # <-- applies return from exit_day onwards
        equity[day] *= (1 + t.net_return)
```

**Problem:** The equity curve applies a trade's return to *every* day from exit onwards, compounding all trade returns on top of each other after each exit. This is not a realistic equity curve — it double-counts and creates artificial compounding. The correct approach is to build a daily P&L series and cumulatively compound it.

**Fix:**
```python
def _compute_metrics(trades, prices, symbol):
    n_days = len(prices)
    daily_pnl = np.zeros(n_days)

    for t in trades:
        # Spread return evenly or mark it on exit day
        if t.exit_day < n_days:
            daily_pnl[t.exit_day] += t.net_return

    equity = np.cumprod(1 + daily_pnl)
    equity = np.concatenate([[1.0], equity[:-1]])  # start at 1
    ...
```

---

### 3. `evaluation.py` Imports Non-Existent Names

**File:** `evaluation.py`

```python
# CURRENT (BUG) — these names do not exist in config.py or other modules
from config import (
    GPConfig, DEFAULT_GP_CONFIG, V1_GP_FEATURES,
    TRADABLE_STOCKS, OUTPUT_DIR,
)
from backtester import (
    BacktestResult, backtest_stock, generate_signals_vectorised, run_backtest,
)
from fitness import prepare_eval_data, prepare_multi_stock_eval_data, composite_fitness
from utils import get_logger, print_banner, print_table, ensure_output_dirs, Timer
```

**Problems:**
- `config.py` has no `GPConfig`, `DEFAULT_GP_CONFIG`, `V1_GP_FEATURES`, or `TRADABLE_STOCKS`.
- `backtester.py` has no `backtest_stock`, `generate_signals_vectorised`, or `run_backtest`.
- `fitness.py` has no `prepare_eval_data`, `prepare_multi_stock_eval_data`, or `composite_fitness`.
- `utils.py` has no `Timer` (it's defined there actually — this one is fine) but missing `ensure_output_dirs` export check.

**Fix:** Align `evaluation.py` imports with what actually exists:
```python
from config import ALL_SYMBOLS, DAILY_FEATURES, OUTPUT_DIR
from backtester import BacktestResult, backtest_portfolio, generate_signal
from fitness import evaluate_individual
```

---

### 4. `regime_analysis.py` Imports Non-Existent Config Variables

**File:** `regime_analysis.py`

```python
from config import (
    GPConfig, DEFAULT_GP_CONFIG, V1_GP_FEATURES,
    REGIME_BULL_THRESHOLD, REGIME_BEAR_THRESHOLD,
    TRADABLE_STOCKS, OUTPUT_DIR,
)
```

**Problem:** `config.py` does not define `GPConfig`, `DEFAULT_GP_CONFIG`, `V1_GP_FEATURES`, `REGIME_BULL_THRESHOLD`, `REGIME_BEAR_THRESHOLD`, or `TRADABLE_STOCKS`.

**Fix:** Either add these to `config.py`, or replace with what exists:
```python
# Add to config.py:
V1_GP_FEATURES      = DAILY_FEATURES        # alias
TRADABLE_STOCKS     = ALL_SYMBOLS           # alias
REGIME_BULL_THRESHOLD = 0.05
REGIME_BEAR_THRESHOLD = -0.05

# Or update the import in regime_analysis.py:
from config import ALL_SYMBOLS as TRADABLE_STOCKS, DAILY_FEATURES as V1_GP_FEATURES, OUTPUT_DIR
REGIME_BULL_THRESHOLD = 0.05
REGIME_BEAR_THRESHOLD = -0.05
```

---

### 5. `main_evaluate_gp.py` Imports Non-Existent Names

**File:** `main_evaluate_gp.py`

```python
from config import GPConfig, DEFAULT_GP_CONFIG, V1_GP_FEATURES, TRADABLE_STOCKS, OUTPUT_DIR
from dataset_builder import load_feature_dataset, split_by_time
```

**Problem:** `config.py` does not define `GPConfig`, `DEFAULT_GP_CONFIG`, `V1_GP_FEATURES`, or `TRADABLE_STOCKS`. `dataset_builder.py` has `load_features` and `split_dataset`, not `load_feature_dataset` and `split_by_time`.

**Fix:**
```python
from config import DAILY_FEATURES, ALL_SYMBOLS, OUTPUT_DIR
from dataset_builder import load_features, split_dataset
```

---

### 6. `backtest_all_stocks.py` Hardcodes Stop-Loss Values Inconsistently

**File:** `backtest_all_stocks.py` — `generate_trade_signals()`

```python
STOP_LOSS        = 0.05
TRAILING_STOP    = 0.07
MIN_HOLD         = 3       # <-- config.py says MIN_HOLD_DAYS = 5
COST             = 0.0005  # <-- config.py says TOTAL_COST_PCT = 0.0005 (OK)
```

**Problem:** `MIN_HOLD = 3` differs from `config.MIN_HOLD_DAYS = 5`. The standalone script will produce different trade results from the main backtester, causing confusion when comparing results.

**Fix:** Import constants from `config.py` instead of hardcoding:
```python
from config import (
    STOP_LOSS_PCT, TRAILING_STOP_PCT, MIN_HOLD_DAYS,
    MAX_HOLD_DAYS, TOTAL_COST_PCT
)
```

---

### 7. `evaluate_results.py` References Undefined `colors` Variable

**File:** `evaluate_results.py` — `plot_trade_distribution()`

```python
bars = ax.bar(sym_counts.keys(), sym_counts.values(),
             color=colors[:len(sym_counts)])   # `colors` not defined in scope
```

**Problem:** `colors` is defined as a module-level variable *after* the function definition. While Python resolves this at call time (not definition time), it's a fragile design that confuses readers and linters.

**Fix:** Define `colors` at the top of the file, before all function definitions.

---

## 🟡 Logic & Design Issues

### 8. Fitness Function Overfitting Guard May Be Too Lenient

**File:** `fitness.py` — `evaluate_individual()`

```python
if train_fit > EPSILON:
    consistency = val_fit / (train_fit + EPSILON)
    if consistency < 0.3:
        overfit_penalty = 0.3
    elif consistency < 0.6:
        overfit_penalty = 0.7
    else:
        overfit_penalty = 1.0
```

**Problem:** When `val_fit` is negative but `train_fit` is positive, `consistency` becomes negative, and `consistency < 0.3` is True → `overfit_penalty = 0.3`. This means a strategy that loses money on validation still gets 30% of its training fitness. A strategy that loses on out-of-sample data should receive a heavier penalty.

**Suggested Fix:**
```python
if train_fit > EPSILON:
    if val_fit <= 0:
        # Loses money on validation — strong overfitting signal
        overfit_penalty = 0.1
    else:
        consistency = val_fit / (train_fit + EPSILON)
        if consistency < 0.3:
            overfit_penalty = 0.2
        elif consistency < 0.6:
            overfit_penalty = 0.6
        else:
            overfit_penalty = 1.0
```

---

### 9. `portfolio_fitness()` Requires ≥ 2 Valid Stocks But May Miss Edge Cases

**File:** `fitness.py` — `portfolio_fitness()`

```python
valid = [s for s in scores if s > -50.0]
if len(valid) < 2:
    return -50.0 + len(valid) * 5.0
```

**Problem:** With 6 stocks in the universe, a strategy only needs to work on 2 to pass this gate. A strategy that catastrophically fails on 4 of 6 stocks may still score positively. Consider requiring a minimum of 3–4 valid stocks.

**Suggested Fix:**
```python
MIN_VALID_STOCKS = 3  # Add to config.py
if len(valid) < MIN_VALID_STOCKS:
    return -50.0 + len(valid) * 5.0
```

---

### 10. `get_logger()` Returns a Singleton That May Bleed Across Modules

**File:** `utils.py`

```python
_logger: Optional[logging.Logger] = None

def get_logger(name: str = "gp_system") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    ...
    logger = logging.getLogger(name)
```

**Problem:** The function caches only one logger regardless of `name`. Calling `get_logger("gp_engine")` after `get_logger("fitness")` silently returns the `"fitness"` logger, making the `name` parameter misleading. This also prevents fine-grained log filtering per module.

**Fix:**
```python
_loggers: dict = {}

def get_logger(name: str = "gp_system") -> logging.Logger:
    global _loggers
    if name in _loggers:
        return _loggers[name]
    
    ensure_output_dirs()
    logger = logging.getLogger(name)
    # ... setup handlers ...
    _loggers[name] = logger
    return logger
```

---

### 11. Multiprocessing Pool Initialiser Is Not Used Correctly

**File:** `gp_engine.py` — `run_evolution()`

```python
pool = mp.Pool(processes=n_workers)
toolbox.register("map", pool.map)
```

**Problem:** `pool.map` does not pass worker-initialisation state to the subprocess. The `init_worker()` function is called on the *main process* but not passed as an initializer to `mp.Pool`, so worker processes will have `_GLOBAL_TOOLBOX = None` and `_GLOBAL_TRAIN_DATA = None`. The code works only because `evaluate_individual` falls back to the globals set by `init_worker()` in the main process — but in separate worker processes these globals are never set.

**Fix:**
```python
from fitness import init_worker

pool = mp.Pool(
    processes=n_workers,
    initializer=init_worker,
    initargs=(toolbox, train_data, val_data),
)
```

---

### 12. `dataset_builder.split_dataset()` Has an Off-By-One on TRAIN_END

**File:** `dataset_builder.py`

```python
train = df[df.index <  TRAIN_END].copy()   # strict less-than
val   = df[(df.index >= VAL_START) & (df.index < VAL_END)].copy()
```

**Problem:** `TRAIN_END = "2023-12-31"` and `VAL_START = "2024-01-01"`. If the index contains a timestamp on `2023-12-31`, it will be included in train. But `df.index < "2023-12-31"` drops December 31 itself when the index is a date (not datetime). This means the last trading day of 2023 is silently excluded from training.

**Fix:**
```python
train = df[df.index <= TRAIN_END].copy()   # inclusive end
```

---

## 🟢 Code Quality & Best Practices

### 13. Dead / One-Off Utility Scripts Should Be Moved to a `scripts/` Subfolder

**Files:** `fix.py`, `update_config.py`, `check_formula.py`

These files are one-time patch scripts that were used to modify other files or check a saved model. They do not belong in the package root alongside production modules. They clutter the namespace and risk being run accidentally.

**Recommendation:** Move to `scripts/` or `tools/` directory and add a note in the README explaining their historical purpose.

---

### 14. Magic Numbers Are Scattered Across `backtest_all_stocks.py`

As noted in Bug #6, the standalone backtest script defines its own constants rather than importing from `config.py`. This means any tuning change in `config.py` will not be reflected in this script.

**Recommendation:** Centralise all constants in `config.py` and import everywhere.

---

### 15. `feature_engineering.py` Silently Ignores NIFTY Alignment Failures

**File:** `feature_engineering.py` — `build_daily_features()`

```python
if matched > 0:
    nifty_close = nc_aligned
```

**Problem:** If only 1 out of 500 dates matches (e.g., timezone issues), the code will proceed with almost entirely forward-filled Nifty data — a near-useless signal — without any warning.

**Fix:**
```python
MIN_NIFTY_MATCH_PCT = 0.8
match_pct = matched / len(daily)
if match_pct >= MIN_NIFTY_MATCH_PCT:
    nifty_close = nc_aligned
else:
    log.warning(
        f"  {symbol}: Nifty match rate too low ({match_pct:.1%}), "
        "setting Nifty features to zero"
    )
    nifty_close = None
```

---

### 16. `gp_engine.py` Uses `toolbox.current_generation` as an Ad-Hoc Attribute

**File:** `gp_engine.py`

```python
toolbox.current_generation = gen
```

**Problem:** Storing state on the DEAP toolbox as an ad-hoc attribute is fragile — it can be overwritten by DEAP internals or cause confusion. The current generation should be passed explicitly.

**Fix:** Pass `generation` as a keyword argument through `evaluate_kwargs`:
```python
evaluate_kwargs["generation"] = gen
register_evaluate(toolbox, evaluate_func, **evaluate_kwargs)
```

---

### 17. No `requirements.txt` or `pyproject.toml`

The repository has no dependency file. New users have no way to know which packages or versions are required beyond the brief mention in the existing README.

**Recommendation:** Add a `requirements.txt`:
```
deap>=1.4.1
dill>=0.3.7
duckdb>=0.10.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pyarrow>=14.0.0   # for parquet support
```

---

### 18. `EarlyStopping` Patience Check Is Off-by-One

**File:** `gp_engine.py` — `EarlyStopping.update()`

```python
if self.counter >= self.patience:
    ...
    return True
```

**Problem:** With `patience=20`, stopping triggers when `counter == 20`, meaning 20 generations without improvement. But the log message `f"({remaining} more until early stop)"` computes `remaining = patience - counter`, which becomes 0 at trigger — correctly showing "0 more" — so the message is accurate. However, the intent of `patience` (usually meaning "wait N more generations") is met: this is fine as-is but worth documenting clearly.

**Recommendation:** Add a docstring clarification:
```python
"""
patience : int
    Number of consecutive generations without improvement
    before stopping. E.g., patience=20 stops after 20 stagnant
    generations.
"""
```

---

### 19. `_compute_gen_stats()` Threshold for "Valid" Fitness Is Hardcoded

**File:** `gp_engine.py`

```python
valid_fits = [
    ind.fitness.values[0]
    for ind in population
    if ind.fitness.valid and ind.fitness.values[0] > -99
]
```

**Problem:** The `-99` threshold is a magic number not defined in `config.py`. It should be a named constant to be consistent with `BASE_FAIL = -50.0` in `fitness.py`.

**Fix:**
```python
# In config.py:
FITNESS_INVALID_THRESHOLD = -99.0

# In gp_engine.py:
from config import FITNESS_INVALID_THRESHOLD
valid_fits = [
    ind.fitness.values[0]
    for ind in population
    if ind.fitness.valid and ind.fitness.values[0] > FITNESS_INVALID_THRESHOLD
]
```

---

## 🔵 Enhancement Suggestions

### 20. Add Vectorised Signal Generation for Faster Backtesting

**File:** `backtester.py` — `generate_signal()`

```python
def generate_signal(func, feature_matrix):
    n = feature_matrix.shape[0]
    signal = np.zeros(n)
    for i in range(n):
        row = feature_matrix[i]
        try:
            val = func(*row)
            ...
```

**Problem:** This is a pure Python loop over potentially thousands of rows. For a population of 5,000 individuals evaluated per generation, this is a significant bottleneck.

**Suggestion:** For simple tree expressions, use `numpy` vectorised evaluation where possible, or batch rows using `np.apply_along_axis`. At minimum, wrap the loop in `numba.jit` if available:

```python
from functools import partial

def generate_signal_vectorised(func, feature_matrix):
    """Vectorised signal generation using numpy apply."""
    try:
        signals = np.apply_along_axis(
            lambda row: float(func(*row)), axis=1, arr=feature_matrix
        )
        signals = np.where(np.isfinite(signals), signals, 0.0)
        return signals
    except Exception:
        return generate_signal(func, feature_matrix)  # fallback
```

---

### 21. Add Sharpe-Weighted Signal Normalisation Option

Currently the signal is normalised with `tanh`, which maps any real number to `(-1, 1)`. Consider adding an alternative position-sizing function based on signal strength:

```python
def signal_to_weight(signal: float, max_weight: float = 0.30) -> float:
    """Convert normalised signal to position weight."""
    abs_sig = abs(signal)
    if abs_sig < 0.1:   # Dead zone
        return 0.0
    weight = min(abs_sig, 1.0) * max_weight
    return weight if signal > 0 else -weight
```

This enables proper position sizing as documented in `config.py` (`MAX_POSITION_WEIGHT = 0.30`) but not currently implemented in the backtester.

---

### 22. Implement Walk-Forward Validation

The current train/val/test split is a single fixed-window split. For a robust trading strategy, **walk-forward optimisation** significantly reduces look-ahead bias:

```python
def walk_forward_splits(df, n_splits=4, train_years=2, val_months=6):
    """Generate walk-forward train/val splits."""
    splits = []
    for i in range(n_splits):
        train_end = TRAIN_START + relativedelta(years=train_years + i * val_months // 12)
        val_end   = train_end + relativedelta(months=val_months)
        splits.append((
            df[df.index < train_end],
            df[(df.index >= train_end) & (df.index < val_end)]
        ))
    return splits
```

---

### 23. Add Semantic Mutation Operators

The current mutation uses `gp.mutUniform` (randomly replaces a subtree). Consider adding **semantic-aware** mutation operators that are more likely to produce meaningful variation:

- **Point mutation**: replace a single primitive with another of the same arity
- **Hoist mutation**: replace a subtree with one of its own sub-subtrees (reduces bloat)
- **Shrink mutation**: replace a subtree with a terminal (complexity control)

```python
# In gp_individual.py
toolbox.register("mutate_hoist", gp.mutShrink)     # reduces tree size
toolbox.register("mutate_point", gp.mutNodeReplacement, pset=pset)

# In gp_engine.py _breed_next_generation():
for i in range(len(offspring)):
    r = random.random()
    if r < GP_MUTATION * 0.5:
        offspring[i], = toolbox.mutate(offspring[i])       # uniform
    elif r < GP_MUTATION * 0.75:
        offspring[i], = toolbox.mutate_hoist(offspring[i]) # shrink
    elif r < GP_MUTATION:
        offspring[i], = toolbox.mutate_point(offspring[i]) # point
```

---

### 24. Add a Bloat Control Mechanism

GP trees tend to grow in size over generations (bloat), leading to overfitting and slower evaluation. The current `GP_MAX_NODES = 50` cap hard-limits this, but the complexity penalty in fitness is very mild (`COMPLEXITY_PENALTY = 0.1`).

**Suggestion:** Add a length-weighted parsimony pressure to the fitness function:

```python
# In fitness.py single_stock_fitness():
# Current:
complexity = COMPLEXITY_PENALTY * (tree_size / GP_MAX_NODES)

# Enhanced:
# Quadratic penalty to more aggressively penalise large trees
complexity = COMPLEXITY_PENALTY * (tree_size / GP_MAX_NODES) ** 2
```

---

### 25. Add a Configuration Validation Function

`config.py` has no validation. Invalid configurations (e.g., `GP_MUTATION + GP_CROSSOVER > 1.0`, `MIN_HOLD_DAYS > MAX_HOLD_DAYS`) will silently produce wrong results.

```python
# Add to config.py:
def validate_config():
    assert GP_POPULATION > 0, "Population must be positive"
    assert GP_GENERATIONS > 0, "Generations must be positive"
    assert 0 < GP_CROSSOVER < 1, "Crossover must be in (0, 1)"
    assert 0 < GP_MUTATION < 1, "Mutation must be in (0, 1)"
    assert GP_CROSSOVER + GP_MUTATION <= 1.0, "Crossover + Mutation should be <= 1.0"
    assert MIN_HOLD_DAYS < MAX_HOLD_DAYS, "MIN_HOLD must be less than MAX_HOLD"
    assert GP_MIN_DEPTH < GP_MAX_DEPTH, "GP_MIN_DEPTH must be less than GP_MAX_DEPTH"
    assert STOP_LOSS_PCT < TRAILING_STOP_PCT, "Hard stop should be tighter than trailing"

if __name__ == "__main__":
    validate_config()
    print("Config OK")
```

---

### 26. Add Reproducibility Seeding Support via CLI

Currently seeds are hardcoded (`random.seed(42)`, `np.random.seed(42)`). For reproducibility studies, expose seed as a command-line argument:

```python
# In main_train_gp.py:
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
```

---

### 27. Log the Evolved Formula in Human-Readable Infix Notation

The formula is currently logged in prefix (LISP-style) notation, e.g.:
```
add(sub(rsi_14, half), mul(close_vs_sma20, ret_5d))
```

A human-readable infix converter would make interpretation much easier:
```
((rsi_14 - 0.5) + (close_vs_sma20 * ret_5d))
```

```python
# In gp_primitives.py, add:
def tree_to_infix(individual, pset) -> str:
    """Convert GP tree to infix notation for readability."""
    # Use DEAP's export utilities or a custom recursive walker
    try:
        from deap.gp import graph
        nodes, edges, labels = graph(individual)
        # Walk the tree recursively to build infix string
        ...
    except Exception:
        return str(individual)
```

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🔴 Critical | `gp_engine.py` | `creator` not imported — `NameError` on fresh run |
| 2 | 🔴 Critical | `backtester.py` | Equity curve computation is incorrect |
| 3 | 🔴 Critical | `evaluation.py` | Imports non-existent names — `ImportError` |
| 4 | 🔴 Critical | `regime_analysis.py` | Imports non-existent config variables |
| 5 | 🔴 Critical | `main_evaluate_gp.py` | Imports non-existent names |
| 6 | 🔴 Critical | `backtest_all_stocks.py` | `MIN_HOLD = 3` conflicts with `config.MIN_HOLD_DAYS = 5` |
| 7 | 🟡 Medium | `evaluate_results.py` | `colors` variable used before definition |
| 8 | 🟡 Medium | `fitness.py` | Overfitting guard too lenient for negative val fitness |
| 9 | 🟡 Medium | `fitness.py` | Only 2/6 stocks required for valid portfolio score |
| 10 | 🟡 Medium | `utils.py` | Logger singleton ignores the `name` parameter |
| 11 | 🟡 Medium | `gp_engine.py` | Multiprocessing pool workers never initialised with data |
| 12 | 🟡 Medium | `dataset_builder.py` | `TRAIN_END` date boundary drops last day of training |
| 13 | 🟢 Quality | multiple | One-off scripts mixed with production modules |
| 14 | 🟢 Quality | `backtest_all_stocks.py` | Magic numbers not sourced from `config.py` |
| 15 | 🟢 Quality | `feature_engineering.py` | Silently uses mostly-NaN Nifty data |
| 16 | 🟢 Quality | `gp_engine.py` | `current_generation` stored as ad-hoc toolbox attribute |
| 17 | 🟢 Quality | root | No `requirements.txt` or dependency specification |
| 18 | 🟢 Quality | `gp_engine.py` | `EarlyStopping.patience` semantics undocumented |
| 19 | 🟢 Quality | `gp_engine.py` | Magic number `-99` should be a named constant |
| 20 | 🔵 Enhancement | `backtester.py` | Python loop in signal generation — vectorise for speed |
| 21 | 🔵 Enhancement | `backtester.py` | Position sizing not implemented despite config entries |
| 22 | 🔵 Enhancement | `dataset_builder.py` | Walk-forward validation would reduce look-ahead bias |
| 23 | 🔵 Enhancement | `gp_individual.py` | Add semantic mutation operators (hoist, point) |
| 24 | 🔵 Enhancement | `fitness.py` | Quadratic complexity penalty to reduce bloat |
| 25 | 🔵 Enhancement | `config.py` | Add `validate_config()` to catch misconfiguration early |
| 26 | 🔵 Enhancement | `main_train_gp.py` | Expose random seed as CLI argument |
| 27 | 🔵 Enhancement | `gp_primitives.py` | Add infix notation converter for formula readability |