# GP Trading — Genetic Programming Strategy Discovery

> **Automatically discover profitable long/short trading strategies for Indian equities using Genetic Evolution Programming (GEP), evaluated across ICICIBANK, HDFCBANK, RELIANCE, INFY, TCS, and WIPRO with NIFTY as market context.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How Genetic Evolution Programming Works Here](#2-how-genetic-evolution-programming-works-here)
3. [Architecture & Project Structure](#3-architecture--project-structure)
4. [Installation & Dependencies](#4-installation--dependencies)
5. [Data Requirements](#5-data-requirements)
6. [Usage Guide](#6-usage-guide)
7. [Configuration Reference](#7-configuration-reference)
8. [Feature Engineering](#8-feature-engineering)
9. [Trading Rules & Backtester](#9-trading-rules--backtester)
10. [Fitness Function](#10-fitness-function)
11. [Output Files](#11-output-files)
12. [Extending the System](#12-extending-the-system)
13. [Known Issues & Improvements](#13-known-issues--improvements)
14. [Contributing](#14-contributing)
15. [License](#15-license)

---

## 1. Project Overview

This project uses **Genetic Programming (GP)** — a form of evolutionary computation — to automatically search for mathematical trading signal formulas that generate consistent profits across a diversified portfolio of Indian large-cap stocks.

Rather than hand-crafting trading rules, the system evolves a population of mathematical expressions (trees) over many generations. Each expression is tested as a trading signal on historical data, and the best-performing formulas are selectively bred, crossed over, and mutated — analogous to natural selection — until robust strategies emerge.

### Key Design Goals

| Goal | Implementation |
|------|---------------|
| **Anti-overfitting** | Train on 2021–2023, validate on 2024, test on 2025+ |
| **Generalisation** | Single formula evaluated across 6 stocks simultaneously |
| **Regime awareness** | NIFTY bull/bear filter — longs only in bull, shorts only in bear |
| **Realistic simulation** | Next-day open execution, 0.05% commission + 0.02% slippage, stop-losses |
| **No look-ahead bias** | All features are point-in-time; no future data leaks into signals |
| **Interpretability** | Evolved formulas are human-readable mathematical expressions |

### What the System Produces

After a full training run the system outputs:

- **A trading formula** — a mathematical expression like `add(sub(rsi_14, half), mul(close_vs_sma20, ret_5d))` that maps daily market features to a continuous signal in `[-1, 1]`
- **Backtest performance** across train, validation, and test periods per stock
- **Regime-sliced analysis** — how the strategy performs in bull, flat, and bear markets
- **A trade log** with entry/exit dates, prices, hold periods, and P&L for every stock

---

## 2. How Genetic Evolution Programming Works Here

### 2.1 The Individual: A Mathematical Formula Tree

Each **individual** in the GP population is a tree-structured mathematical expression. The tree's leaves (terminals) are technical indicator values for a given stock on a given day, and the internal nodes (primitives) are mathematical operations.

**Example tree** representing the formula `(RSI_14 - 0.5) + (Close_vs_SMA20 × Return_5d)`:

```
         add
        /   \
      sub    mul
      / \   /   \
  rsi_14 0.5 close_vs_sma20 ret_5d
```

This formula is evaluated for each stock on each day. Its output is passed through `tanh` to produce a normalised signal in `[-1, 1]`:

- **Signal > 0** → Buy (Long) signal; position size ∝ signal strength
- **Signal < 0** → Sell (Short) signal; position size ∝ |signal| strength
- **Signal ≈ 0** → No position

### 2.2 The Primitive Set (Building Blocks)

**Mathematical Operators (internal nodes):**

| Primitive | Arity | Description |
|-----------|-------|-------------|
| `add(a, b)` | 2 | Protected addition — returns 0 on overflow |
| `sub(a, b)` | 2 | Protected subtraction |
| `mul(a, b)` | 2 | Protected multiplication |
| `div(a, b)` | 2 | Protected division — returns 0 when denominator ≈ 0 |
| `max(a, b)` | 2 | Maximum of two values |
| `min(a, b)` | 2 | Minimum of two values |
| `neg(a)` | 1 | Negation |
| `exp(a)` | 1 | Exponential (capped at a=10 to prevent overflow) |
| `log(a)` | 1 | Natural log (domain-protected) |
| `sqrt(a)` | 1 | Square root (domain-protected) |

> **Note:** `abs()` is intentionally excluded. This forces the GP to generate both positive (long) and negative (short) signals naturally rather than always outputting non-negative values.

**Feature Terminals (leaf nodes):** 19 normalised daily features — see [Section 8](#8-feature-engineering).

**Constant Terminals:**

| Terminal | Value |
|----------|-------|
| `zero` | 0.0 |
| `one` | 1.0 |
| `neg_one` | -1.0 |
| `half` | 0.5 |
| `tenth` | 0.1 |
| `hundredth` | 0.01 |
| `rand_const` | Random value in [-1.0, 1.0] (ephemeral) |

### 2.3 Population Initialisation

A fresh evolution run initialises a population of **5,000 trees** using the `genHalfAndHalf` method (DEAP), which generates trees of depth 2–8 using either the "full" method (all leaves at max depth) or "grow" method (random depth), 50/50. This produces diverse trees of varying shapes and sizes.

**Seed templates:** To give evolution a head-start, 10 hand-crafted "oscillating signal" templates are injected into the initial population:
```python
"sub(rsi_14, half)"           # RSI mean-reversion signal
"sub(close_vs_sma20, zero)"   # Trend-following signal
"macd_hist"                   # Momentum signal
"ret_5d"                      # Short-term momentum
...
```

### 2.4 Fitness Evaluation

Each individual's fitness is a **composite score** combining:

```
fitness = 0.4 × train_fitness + 0.6 × val_fitness
```

Where `train_fitness` and `val_fitness` each equal:

```
per_stock_fitness = W_annual_return × annual_return
                  + W_sharpe       × sharpe_ratio
                  + W_sortino      × sortino_ratio
                  + W_win_rate     × win_rate
                  - W_max_drawdown × |max_drawdown|
                  + W_trade_count  × trade_frequency_bonus
                  - complexity_penalty × (tree_size / max_nodes)
```

And the **portfolio fitness** (across all 6 stocks) is:

```
portfolio_fitness = mean(per_stock_scores) - λ × std(per_stock_scores)
```

The `λ × std` term penalises strategies that work well on some stocks but poorly on others, encouraging generalisation.

An **overfitting penalty multiplier** is applied when validation performance significantly lags training:

| Val/Train Consistency | Penalty Multiplier |
|----------------------|--------------------|
| ≥ 60% | 1.0 (no penalty) |
| 30–60% | 0.7 |
| < 30% | 0.3 |

**Hard gates** reject individuals outright (returning score = -50) if:
- Fewer than 10 total trades across all stocks
- Training fitness ≤ -40 (catastrophic failure)

### 2.5 Selection, Crossover & Mutation

Each generation applies three genetic operators to produce the next generation:

**Elitism:** The top 30 individuals (Hall of Fame) are carried over unchanged, ensuring the best solutions are never lost.

**Tournament Selection:** From the remaining population, individuals compete in random groups of 7 (`GP_TOURNAMENT = 7`). The fittest individual in each group is selected for breeding. Higher tournament size = stronger selection pressure.

**Crossover (75% probability):** Two parent trees swap randomly selected subtrees:
```
Parent A: add(rsi_14, mul(ret_5d, half))
Parent B: sub(close_vs_sma20, macd_hist)

After crossover:
Child A:  add(rsi_14, macd_hist)                     <- took subtree from B
Child B:  sub(close_vs_sma20, mul(ret_5d, half))     <- took subtree from A
```

**Mutation (15% probability):** A random subtree is replaced with a newly generated random subtree:
```
Before: add(sub(rsi_14, half), ret_5d)
After:  add(mul(close_vs_sma50, neg_one), ret_5d)    <- subtree replaced
```

**Size limits** are enforced via DEAP's `staticLimit` decorator — trees that would exceed 50 nodes or depth 8 after crossover/mutation are reverted to a copy of the parent.

### 2.6 Early Stopping & Checkpointing

Evolution runs for up to **60 generations**. If the best fitness does not improve by at least 0.01 for 20 consecutive generations, evolution halts early.

Checkpoints are saved to `gp_output/checkpoint.pkl` every 5 generations and during chunked evaluation. If a run is interrupted, it resumes from the last checkpoint automatically.

### 2.7 Full Evolution Pipeline

```
+---------------------------------------------------------------------+
|  1. Initialise population (5,000 trees) + inject seed templates      |
|  2. For each generation (up to 60):                                  |
|     a. Evaluate unevaluated individuals in chunks of 300             |
|        -> Backtest on train data (2021-2023) -> train_fitness        |
|        -> Backtest on val data   (2024)      -> val_fitness          |
|        -> Apply overfitting penalty          -> final fitness        |
|     b. Update Hall of Fame (top 30)                                  |
|     c. Log generation statistics                                     |
|     d. Check early stopping                                          |
|     e. Save checkpoint + best model                                  |
|     f. Breed next generation:                                        |
|        -> Keep top 30 elite                                          |
|        -> Tournament selection for breeding pool                     |
|        -> Apply crossover (75%) + mutation (15%)                     |
|        -> Enforce size/depth limits                                  |
|  3. Save final best model to gp_output/best_model.pkl               |
+---------------------------------------------------------------------+
```

---

## 3. Architecture & Project Structure

```
gp_trading/
|
+-- config.py                  # All configuration: paths, GP params, trading rules
+-- utils.py                   # Logging, timing (Timer), formatting helpers
|
+-- data_loader.py             # Load raw 1-minute OHLCV bars from DuckDB
+-- feature_engineering.py    # Compute normalised daily features (TA indicators)
+-- dataset_builder.py        # Combine features, split train/val/test, save parquet
|
+-- gp_primitives.py           # Safe math primitives + DEAP PrimitiveSet builder
+-- gp_individual.py           # DEAP creator/toolbox setup, population management
+-- gp_engine.py               # Evolution loop, checkpointing, early stopping
|
+-- backtester.py              # Signal generation + positional backtest engine
+-- fitness.py                 # Composite fitness function, overfitting guard
+-- evaluation.py              # Per-stock reporting, summary tables, CSV export
+-- regime_analysis.py         # Bull/bear/flat regime labelling + per-regime eval
|
+-- main_build_features.py    # Script: build feature dataset from DuckDB
+-- main_train_gp.py           # Script: run GP evolution
+-- main_evaluate_gp.py        # Script: evaluate a saved model
+-- backtest_all_stocks.py     # Script: backtest across all DB stocks + trade log
+-- evaluate_results.py        # Script: visual equity curve + trade distribution
|
+-- fix.py                     # One-time patch script (historical)
+-- update_config.py           # One-time config migration script (historical)
+-- check_formula.py           # One-time formula verification script (historical)
|
+-- IMPROVEMENTS.md            # Code review: bugs, quality issues, enhancements
+-- README.md                  # This file
```

### Module Dependency Graph

```
config.py
    |-> utils.py
    |-> data_loader.py ---------> dataset_builder.py
    |-> feature_engineering.py -> dataset_builder.py
    |-> gp_primitives.py ------> gp_individual.py -> gp_engine.py
    |-> backtester.py ----------> fitness.py ------> gp_engine.py
    |-> evaluation.py ----------> main_evaluate_gp.py
    |-> regime_analysis.py -----> main_evaluate_gp.py
```

---

## 4. Installation & Dependencies

### 4.1 Prerequisites

- **Python 3.10+** (tested on 3.11)
- **DuckDB database** — `gp_research.duckdb` in the project root with 1-minute OHLCV data

### 4.2 Install Dependencies

```bash
pip install deap>=1.4.1 \
            dill>=0.3.7 \
            duckdb>=0.10.0 \
            pandas>=2.0.0 \
            numpy>=1.24.0 \
            pyarrow>=14.0.0 \
            matplotlib>=3.7.0
```

Or using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**`requirements.txt`** (create this file in the project root):
```
deap>=1.4.1
dill>=0.3.7
duckdb>=0.10.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
matplotlib>=3.7.0
```

### 4.3 Verify Installation

```bash
python -c "import deap, dill, duckdb, pandas, numpy; print('All dependencies OK')"
```

---

## 5. Data Requirements

### 5.1 DuckDB Database Schema

The system reads from a DuckDB file named `gp_research.duckdb` located in the project root. It expects a table `min_1m` with 1-minute OHLCV bars:

```sql
CREATE TABLE min_1m (
    symbol   VARCHAR,    -- e.g. 'ICICIBANK', 'NIFTY'
    ts       TIMESTAMP,  -- Bar timestamp (IST timezone recommended)
    open     DOUBLE,     -- Open price
    high     DOUBLE,     -- High price
    low      DOUBLE,     -- Low price
    close    DOUBLE,     -- Close price
    volume   BIGINT      -- Volume (0 for NIFTY is acceptable)
);

CREATE INDEX idx_min1m_sym_ts ON min_1m(symbol, ts);
```

### 5.2 Required Symbols

| Symbol | Type | Role |
|--------|------|------|
| `ICICIBANK` | Stock | Tradable |
| `HDFCBANK` | Stock | Tradable |
| `RELIANCE` | Stock | Tradable |
| `INFY` | Stock | Tradable |
| `TCS` | Stock | Tradable |
| `WIPRO` | Stock | Tradable |
| `NIFTY` | Index | Market context (regime filter + relative strength) |

### 5.3 Time Coverage

| Split | Period | Minimum Bars/Stock |
|-------|--------|--------------------|
| Train | 2021-08-16 to 2023-12-31 | ~125,000 (1-min) |
| Val | 2024-01-01 to 2024-12-31 | ~60,000 (1-min) |
| Test | 2025-01-01 to present | Any available |

### 5.4 Check Database Contents

```bash
python -c "
from data_loader import get_database_summary
print(get_database_summary().to_string())
"
```

---

## 6. Usage Guide

### Step 1 — Build Daily Feature Dataset

Resample 1-minute bars to daily, compute all technical indicators, and save to parquet:

```bash
python main_build_features.py
```

**Output:** `gp_output/gp_features_daily.parquet`

Expected console output:
```
======================================================================
  BUILDING DAILY FEATURE DATASET
======================================================================
  Processing ICICIBANK...
  ICICIBANK: nifty matched 598/600 dates
  ICICIBANK: 600 daily bars (dropped 14 NaN rows)
  ...
  Combined daily dataset: 3,492 rows, 26 columns
```

---

### Step 2 — Run GP Evolution

```bash
python main_train_gp.py
```

This will run for up to 60 generations (several hours on an 8-core machine). Progress is logged to console and `gp_output/logs/gp_system.log`.

```
======================================================================
  GP POSITIONAL TRADING -- EVOLUTION
======================================================================
  Population: 5000
  Generations: 60
  Tree depth: 2->8
  ...
  Gen   1 / 60:  best=   1.234  avg=  -8.432  std=  12.341  valid=3821/5000
  Gen   2 / 60:  best=   1.891  avg=  -5.123  std=  10.234  valid=4102/5000
  ...
```

**Resume an interrupted run** — just re-run the same command. The checkpoint is automatically detected:
```bash
python main_train_gp.py   # resumes from gp_output/checkpoint.pkl
```

**Force a fresh run** — delete the checkpoint first:
```bash
rm gp_output/checkpoint.pkl && python main_train_gp.py
```

---

### Step 3 — Evaluate the Best Model

```bash
python main_evaluate_gp.py
# or specify the model path explicitly:
python main_evaluate_gp.py --model gp_output/best_model.pkl
```

This prints a full evaluation table across all stocks and time splits:

```
  Symbol       Split    Return%  Ann.Ret%   Sharpe  Sortino   MaxDD%  Trades  WinRate%
  ICICIBANK    train      45.21     18.23    1.342    1.891   -12.30      87      58.6
  ICICIBANK    val        12.34     12.34    0.987    1.234    -8.90      22      54.5
  ICICIBANK    test        3.21      n/a     0.712    0.891    -4.20       5      60.0
  ...
```

---

### Step 4 — Generate Full Trade Log

Backtest the saved model across all stocks in the database and export a detailed trade-by-trade CSV:

```bash
python backtest_all_stocks.py
```

**Output:** `gp_output/all_trades.csv`

Sample trade log:
```
symbol,direction,entry_date,exit_date,entry_price,exit_price,hold_days,pnl_pct,exit_reason
ICICIBANK,LONG,2024-03-12,2024-03-28,1052.30,1123.45,16,6.74,SIGNAL
HDFCBANK,SHORT,2024-05-20,2024-05-31,1621.80,1580.20,11,2.56,SIGNAL
...
```

Exit reasons:
- `SIGNAL` — GP signal reversed direction
- `STOP_LOSS` — Hard 5% stop triggered
- `TRAIL_STOP` — Trailing 7% stop triggered
- `REGIME_FLIP` — NIFTY regime flipped against position
- `MAX_HOLD` — Maximum 60-day hold period reached
- `OPEN_TRADE` — Trade still open at end of data

---

### Step 5 — Visualise Equity Curves

```bash
python evaluate_results.py
```

Generates equity curve charts for each stock on each time split, plus trade return distributions. Charts are saved to `gp_output/` as PNG files.

---

## 7. Configuration Reference

All parameters are in `config.py`. The most important settings:

### Time Splits

```python
TRAIN_START = "2021-08-16"   # Start of training data
TRAIN_END   = "2023-12-31"   # End of training data
VAL_START   = "2024-01-01"   # Start of validation data
VAL_END     = "2024-12-31"   # End of validation data
TEST_START  = "2025-01-01"   # Start of out-of-sample test
```

### GP Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GP_POPULATION` | 5,000 | Number of individuals per generation |
| `GP_GENERATIONS` | 60 | Maximum number of generations |
| `GP_CROSSOVER` | 0.75 | Probability of crossover between two parents |
| `GP_MUTATION` | 0.15 | Probability of mutating an individual |
| `GP_ELITE` | 30 | Number of top individuals preserved each generation |
| `GP_TOURNAMENT` | 7 | Tournament group size for selection |
| `GP_MIN_DEPTH` | 2 | Minimum tree depth at initialisation |
| `GP_MAX_DEPTH` | 8 | Maximum allowed tree depth |
| `GP_MAX_NODES` | 50 | Maximum allowed tree nodes |
| `GP_EARLY_STOP` | 20 | Generations without improvement before stopping |
| `GP_CHUNK_SIZE` | 300 | Individuals evaluated per checkpoint interval |
| `GP_WORKERS` | 8 | Number of parallel worker processes |

### Trading Rules

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_HOLD_DAYS` | 5 | Minimum days to hold a position |
| `MAX_HOLD_DAYS` | 60 | Maximum days to hold a position |
| `STOP_LOSS_PCT` | 0.05 | Hard stop loss (5% from entry) |
| `TRAILING_STOP_PCT` | 0.07 | Trailing stop (7% from peak) |
| `COMMISSION_PCT` | 0.0003 | One-way commission (0.03%) |
| `SLIPPAGE_PCT` | 0.0002 | Estimated slippage (0.02%) |
| `MAX_POSITION_WEIGHT` | 0.30 | Maximum 30% of portfolio per stock |
| `MIN_POSITION_WEIGHT` | 0.05 | Minimum 5% position threshold |

### Fitness Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `W_ANNUAL_RETURN` | 3.0 | Weight for annualised return |
| `W_SHARPE` | 2.0 | Weight for Sharpe ratio |
| `W_SORTINO` | 1.0 | Weight for Sortino ratio |
| `W_WIN_RATE` | 1.0 | Weight for win rate |
| `W_MAX_DRAWDOWN` | 2.0 | Penalty weight for max drawdown |
| `W_TRADE_COUNT` | 0.5 | Bonus/penalty for trade frequency |
| `W_CONSISTENCY` | 1.5 | Cross-stock consistency penalty (lambda) |
| `COMPLEXITY_PENALTY` | 0.1 | Penalty for large tree size |

### Minimum Trade Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_TRADES_TOTAL` | 10 | Minimum total trades across all stocks |
| `MIN_TRADES_PER_STOCK` | 2 | Minimum trades per stock (else -50 fitness) |

---

## 8. Feature Engineering

All features are normalised ratios, returns, or bounded indicators — never raw prices. This ensures the same formula generalises across stocks at different price levels.

### Feature List (19 Features)

| Category | Feature | Description | Range |
|----------|---------|-------------|-------|
| **Returns** | `ret_1d` | 1-day price return | typically +-0.1 |
| | `ret_5d` | 5-day price return | |
| | `ret_10d` | 10-day price return | |
| | `ret_20d` | 20-day price return | |
| **Trend** | `close_vs_sma10` | (Close / SMA10) - 1 | typically +-0.2 |
| | `close_vs_sma20` | (Close / SMA20) - 1 | |
| | `close_vs_sma50` | (Close / SMA50) - 1 | |
| | `close_vs_sma200` | (Close / SMA200) - 1 | |
| **Momentum** | `rsi_14` | RSI(14) / 100 | [0, 1] |
| | `macd_signal` | MACD signal line / Close | small value |
| | `macd_hist` | MACD histogram / Close | small value |
| **Volatility** | `atr_pct_14` | ATR(14) / Close | [0, inf) |
| | `bb_pct` | Bollinger Band %B position | [0, 1] |
| **Volume** | `volume_rel_20` | Volume / 20-day avg volume | [0, inf) |
| **Market** | `nifty_ret_5d` | NIFTY 5-day return | varies |
| | `nifty_ret_20d` | NIFTY 20-day return | |
| | `nifty_vs_sma50` | (NIFTY / NIFTY SMA50) - 1 | |
| **Rel. Strength** | `rs_5d` | ret_5d - nifty_ret_5d | varies |
| | `rs_20d` | ret_20d - nifty_ret_20d | |

### Feature Pipeline

```
DuckDB min_1m table
        |
        v
resample_to_daily()      <- OHLCV aggregation (open=first, high=max, low=min, close=last, vol=sum)
        |
        v
compute_daily_features() <- All 19 indicators computed on daily bars
        |
        v
dropna(subset=DAILY_FEATURES)   <- Remove warmup rows (SMA200 needs 200 days)
        |
        v
save to gp_output/gp_features_daily.parquet
```

---

## 9. Trading Rules & Backtester

### Signal Generation

The compiled GP formula is evaluated for each stock on each day to produce a raw signal. This is passed through `tanh` to normalise:

```
signal = tanh(gp_formula(feature_1, feature_2, ..., feature_19))
signal in (-1, 1)
```

### NIFTY Regime Filter

Before entering a trade, the current NIFTY regime is checked:

```
nifty_vs_sma50 > 0  ->  BULL regime  ->  Only LONG entries allowed
nifty_vs_sma50 < 0  ->  BEAR regime  ->  Only SHORT entries allowed
```

If the regime flips while in a position, the trade is force-exited at the next available price.

### Entry Logic

A new position is opened when the signal crosses zero (in the allowed direction):

```
LONG entry:   signal crosses 0 upward   (prev_sig <= 0, curr_sig > 0)  +  BULL regime
SHORT entry:  signal crosses 0 downward (prev_sig >= 0, curr_sig < 0)  +  BEAR regime
```

Execution uses the **next day's open price** for realism (no same-bar fills).

### Exit Logic

Positions are exited by any of the following (checked in priority order):

1. **Hard stop loss** — price moves 5% against entry
2. **Trailing stop** — price retraces 7% from the peak/trough since entry
3. **Signal exit** — signal crosses back through zero (only after minimum hold)
4. **Regime flip** — NIFTY regime reverses against the position direction
5. **Maximum hold** — position held for 60 days

### Performance Metrics

The backtester computes the following for each stock:

| Metric | Formula |
|--------|---------|
| Total Return % | `(final_equity / initial_equity - 1) x 100` |
| Annual Return % | `(final_equity ^ (1 / years)) - 1` |
| Sharpe Ratio | `mean(daily_ret) / std(daily_ret) x sqrt(252)` |
| Sortino Ratio | `mean(daily_ret) / std(negative_daily_ret) x sqrt(252)` |
| Max Drawdown % | `min((equity - running_peak) / running_peak) x 100` |
| Win Rate % | `wins / total_trades x 100` |
| Profit Factor | `sum(winning_returns) / sum(abs(losing_returns))` |
| Trades per Year | `n_trades / years` |

---

## 10. Fitness Function

The fitness function is the core of what guides evolution. A higher score means a better trading strategy. Here is the complete scoring logic:

### Per-Stock Scoring

```python
fitness = (W_ANNUAL_RETURN * annual_return_decimal
         + W_SHARPE        * sharpe_ratio
         + W_SORTINO       * sortino_ratio
         + W_WIN_RATE      * win_rate_decimal
         - W_MAX_DRAWDOWN  * abs(max_drawdown_decimal)
         + W_TRADE_COUNT   * trade_bonus
         - COMPLEXITY_PENALTY * (tree_size / GP_MAX_NODES))
```

**Trade frequency bonus:**
- < 4 trades/year  -> `-1.0` (too rare to be useful)
- 4 to 50 trades/year -> `+0.5` (ideal frequency)
- > 50 trades/year -> `-0.5` (over-trading)

**Minimum gate:** If fewer than `MIN_TRADES_PER_STOCK = 2` trades, the stock returns `-50` (rejected).

### Portfolio Scoring

```python
per_stock_scores = [score for each stock]
valid_scores     = [s for s in per_stock_scores if s > -50]

if len(valid_scores) < 2:
    return -50 + len(valid_scores) * 5.0   # partial credit

portfolio_fitness = mean(valid_scores) - W_CONSISTENCY * std(valid_scores)
```

### Final Fitness with Overfitting Guard

```python
final_fitness = (0.4 * train_fitness + 0.6 * val_fitness) * overfitting_multiplier
```

| Condition | Multiplier |
|-----------|-----------|
| Val/Train consistency >= 60% | 1.0 |
| Val/Train consistency 30-60% | 0.7 |
| Val/Train consistency < 30% | 0.3 |
| Train fitness <= 0 | 0.5 |

---

## 11. Output Files

All outputs are saved to `gp_output/`:

| File | Description |
|------|-------------|
| `gp_features_daily.parquet` | Pre-computed feature dataset (daily bars x features) |
| `best_model.pkl` | Best evolved GP individual (dill serialised) |
| `checkpoint.pkl` | Latest evolution checkpoint for resume |
| `evolution_stats.csv` | Per-generation statistics (best, avg, std fitness) |
| `evaluation_results.csv` | Per-stock per-split backtest results |
| `evaluation_summary.csv` | Compact summary of evaluation results |
| `regime_train.csv` | Regime-sliced results on training data |
| `regime_val.csv` | Regime-sliced results on validation data |
| `regime_test.csv` | Regime-sliced results on test data |
| `all_trades.csv` | Full trade log with entry/exit details |
| `equity_train.png` | Equity curves chart for training period |
| `equity_val.png` | Equity curves chart for validation period |
| `equity_test.png` | Equity curves chart for test period |
| `trades_train.png` | Trade distribution chart for training period |
| `trades_test.png` | Trade distribution chart for test period |
| `logs/gp_system.log` | Full detailed log file |

### Loading a Saved Model

```python
import dill
from gp_individual import setup_gp_toolbox
from config import DAILY_FEATURES

# Must setup DEAP creator before loading
toolbox, pset = setup_gp_toolbox(feature_names=DAILY_FEATURES)

with open("gp_output/best_model.pkl", "rb") as f:
    model = dill.load(f)

print(f"Fitness  : {model['fitness']:.4f}")
print(f"Tree size: {model['tree_size']} nodes")
print(f"Formula  : {model['tree_str']}")
print(f"Saved at : {model['timestamp']}")

# Compile to callable function
func = toolbox.compile(expr=model["individual"])

# Use: func(ret_1d, ret_5d, ..., rs_20d) -> float signal
```

---

## 12. Extending the System

### Adding New Features

1. Add the computation in `feature_engineering.py` inside `compute_daily_features()`:
   ```python
   df["my_new_feature"] = ...
   ```
2. Add the feature name to `DAILY_FEATURES` in `config.py`:
   ```python
   DAILY_FEATURES = [
       ...,
       "my_new_feature",
   ]
   ```
3. Rebuild features:
   ```bash
   python main_build_features.py
   ```

> **Important:** Adding features changes the GP tree input arity. Existing saved models trained on the old feature set will be incompatible — you must retrain from scratch.

---

### Adding New Primitives

1. Add a safe function in `gp_primitives.py`:
   ```python
   def safe_tanh(a: float) -> float:
       """Hyperbolic tangent with NaN guard."""
       try:
           result = math.tanh(a)
           return 0.0 if math.isnan(result) or math.isinf(result) else result
       except (OverflowError, ValueError):
           return 0.0
   ```
2. Register it in `build_primitive_set()`:
   ```python
   pset.addPrimitive(safe_tanh, 1, name="tanh")
   ```

> Adding primitives is backward-compatible as long as the feature set does not change.

---

### Adding New Stocks

1. Add the symbol to `ALL_SYMBOLS` in `config.py`:
   ```python
   ALL_SYMBOLS = ["ICICIBANK", "HDFCBANK", "RELIANCE", "INFY", "TCS", "WIPRO", "AXISBANK"]
   ```
2. Ensure data for the new symbol exists in `gp_research.duckdb`
3. Rebuild features:
   ```bash
   python main_build_features.py
   ```
4. Retrain the GP (strongly recommended since the fitness landscape changes):
   ```bash
   rm gp_output/checkpoint.pkl
   python main_train_gp.py
   ```

---

### Tuning the GP

The most impactful parameters for exploration:

| Goal | Parameters to Tune |
|------|-------------------|
| Better generalisation | Increase `GP_EARLY_STOP`, raise `W_CONSISTENCY` |
| Simpler formulas | Lower `GP_MAX_NODES`, raise `COMPLEXITY_PENALTY` |
| Faster runs | Lower `GP_POPULATION`, lower `GP_GENERATIONS` |
| More diverse search | Raise `GP_MUTATION`, lower `GP_TOURNAMENT` |
| Higher quality | Raise `GP_POPULATION`, raise `GP_ELITE` |

---

## 13. Known Issues & Improvements

See **[IMPROVEMENTS.md](IMPROVEMENTS.md)** for a full code review including:

- **5 critical bugs** — including a `NameError` from a missing `creator` import in `gp_engine.py`, incorrect equity curve computation in `backtester.py`, and broken imports in `evaluation.py`, `regime_analysis.py`, and `main_evaluate_gp.py`
- **6 logic/design issues** — including the multiprocessing pool not properly passing worker initialisation state
- **8 code quality improvements** — including missing `requirements.txt`, magic numbers, and logger singleton issues
- **8 enhancement suggestions** — including vectorised signal generation, walk-forward validation, and semantic mutation operators

---

## 14. Contributing

Contributions are welcome. Suggested areas:

1. **Fix critical bugs** listed in `IMPROVEMENTS.md` — especially the import errors in `evaluation.py` and `regime_analysis.py`
2. **Add walk-forward validation** — replacing the single fixed split with rolling windows
3. **Vectorise signal generation** — the Python loop in `backtester.generate_signal()` is the main performance bottleneck
4. **Implement position sizing** — `config.py` defines `MAX_POSITION_WEIGHT` and `MIN_POSITION_WEIGHT` but the backtester uses a fixed weight of 1.0
5. **Add infix formula display** — convert LISP-prefix tree strings to human-readable infix notation

### Workflow

```bash
# 1. Clone the repository
git clone https://github.com/Onlytestmail7312/gp_trading.git
cd gp_trading

# 2. Create a feature branch
git checkout -b feature/your-improvement

# 3. Make your changes and test
python -c "from config import *; print('Config OK')"

# 4. Commit with a descriptive message
git commit -m "fix: add missing creator import in gp_engine.py"

# 5. Push and open a pull request
git push origin feature/your-improvement
```

---

## 15. License

This project is private research software. All rights reserved.

---

*Built with [DEAP](https://github.com/DEAP/deap) — Distributed Evolutionary Algorithms in Python.*