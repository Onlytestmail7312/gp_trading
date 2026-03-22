# Unified Multi-Stock GP Trading Research System

A modular Genetic Programming (GP) system for discovering robust trading strategies across multiple Indian equities using NIFTY as market context.

---

## Architecture Overview

```
gp_system_complete/
├── config.py              # Central configuration (paths, features, GP params)
├── utils.py               # Logging, timing, formatting helpers
├── data_loader.py         # Load raw OHLCV from DuckDB
├── feature_engineering.py # Compute normalised features (returns, trend, vol, etc.)
├── dataset_builder.py     # Combine features, NIFTY context, time splits, save parquet
├── gp_primitives.py       # Safe math primitives, signal normalisation, primitive set
├── gp_individual.py       # DEAP creator/toolbox setup, population management
├── gp_engine.py           # Evolution loop, checkpointing, early stopping
├── backtester.py          # Vectorised backtest engine, signal generation
├── fitness.py             # Composite fitness, multi-stock aggregation
├── evaluation.py          # Per-stock reporting, summary tables
├── regime_analysis.py     # Bull/bear/flat regime labelling and evaluation
├── main_build_features.py # Script: build feature dataset from DuckDB
├── main_train_gp.py       # Script: run GP evolution
├── main_evaluate_gp.py    # Script: evaluate saved model
└── README.md              # This file
```

---

## Quick Start

### Prerequisites

```bash
pip install deap dill duckdb pandas numpy
```

### 1. Build Features

Ensure `gp_research.duckdb` is in the working directory with table `min_1m` containing 1-minute OHLCV data.

```bash
python -m gp_system_complete.main_build_features
```

This will:
- Load raw data for 6 stocks + NIFTY from DuckDB
- Compute all normalised features (returns, trend, volatility, volume, time, NIFTY context, relative strength)
- Save to `gp_output/gp_features_1m.parquet` and DuckDB table `gp_features_1m`

### 2. Train GP Model

```bash
python -m gp_system_complete.main_train_gp
```

Optional arguments:
```bash
python -m gp_system_complete.main_train_gp --pop-size 1000 --generations 20 --workers 4
```

This will:
- Load pre-computed features
- Split into train (2021-2023) / val (2024) / test (2025+)
- Run GP evolution with multi-stock fitness
- Save best model to `gp_output/best_model.pkl`
- Evaluate on all splits with regime analysis

### 3. Evaluate Saved Model

```bash
python -m gp_system_complete.main_evaluate_gp
python -m gp_system_complete.main_evaluate_gp --model gp_output/best_model.pkl
```

---

## Design Principles

### Unified Multi-Stock Training
The GP evolves a **single formula** evaluated across 6 stocks simultaneously. The fitness aggregation:

```
fitness_total = mean(stock_fitness) - λ * std(stock_fitness)
```

This rewards strategies that generalise across stocks while penalising high variance in per-stock performance.

### Normalised Features (No Raw Prices)
All GP inputs are normalised ratios, returns, or z-scores — never raw price levels. This ensures the same formula works across stocks trading at different price levels (e.g., RELIANCE ₹2500 vs WIPRO ₹450).

**V1 Feature Set (17 features):**

| Category | Features |
|----------|----------|
| Stock Returns | ret_5m, ret_15m, ret_60m |
| Trend | close_vs_sma20, close_vs_sma50, close_vs_sma200 |
| Volatility | atr_pct |
| Volume | volume_rel_20 |
| Market Context | nifty_ret_5m, nifty_ret_15m, nifty_ret_60m, nifty_close_vs_sma200 |
| Relative Strength | rs_15m, rs_60m, rs_20d |
| Time | minutes_from_open_norm |

### Regime-Aware Evaluation (Not Training)
The model trains on ALL data unified. Regime analysis (bull/bear/flat) is applied **post-training** to understand how the strategy performs in different market conditions:

- **BULL**: trend_50d_return > +5%
- **BEAR**: trend_50d_return < -5%
- **FLAT**: in between

### GP Representation
- **Primitives**: add, sub, mul, safe_div, abs, min, max, neg (8 functions)
- **Terminals**: 17 feature columns + ephemeral constants in [-1, 1] + fixed constants (0, 1, -1, 0.5, 0.1, 0.01)
- **Tree depth**: 2–8 levels, max 50 nodes
- **Signal**: GP output → tanh → normalised to [-1, 1]
- **Strategy**: signal > 0.5 → long, signal < -0.5 → flat (long-only)

### Composite Fitness
```
composite = w_sharpe * sharpe
          + w_sortino * sortino
          + w_return * annual_return / 100
          - w_dd * |max_drawdown| / 100
          - w_trades * excess_trades / 10
          - w_complexity * tree_size / max_nodes
```

### Multi-Stage Filters
Each individual passes through filters before computing fitness:
1. **F0**: Minimum tree size (≥7 nodes)
2. **F0b**: Signal diversity check (std > 0.01)
3. **F1**: Minimum trades on training data
4. **F2**: Max drawdown filter
5. **F3**: Both train and val must have positive returns
6. **F4**: Validation Sharpe ≥ 0
7. **F5**: Must pass ≥2 stocks

---

## Time Splits

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2021-01-01 to 2023-12-31 | GP evolution fitness |
| Val | 2024-01-01 to 2024-12-31 | Overfitting guard |
| Test | 2025-01-01 onwards | Out-of-sample evaluation |

---

## Data Requirements

### DuckDB Table: `min_1m`

| Column | Type | Description |
|--------|------|-------------|
| symbol | VARCHAR | Stock symbol (e.g., "ICICIBANK", "NIFTY") |
| timestamp | TIMESTAMP | Bar timestamp (IST) |
| Open | DOUBLE | Open price |
| High | DOUBLE | High price |
| Low | DOUBLE | Low price |
| Close | DOUBLE | Close price |
| Volume | BIGINT | Volume |

### Supported Symbols
- **Tradable**: ICICIBANK, HDFCBANK, RELIANCE, INFY, TCS, WIPRO
- **Context**: NIFTY (market index, used for context features only)

---

## Output Files

All outputs go to `gp_output/`:

| File | Description |
|------|-------------|
| `gp_features_1m.parquet` | Pre-computed feature dataset |
| `best_model.pkl` | Best evolved GP individual (dill) |
| `checkpoint.pkl` | Evolution checkpoint for resume |
| `evolution_stats.csv` | Per-generation statistics |
| `training.log` | Training log file |
| `evaluation_results.csv` | Per-stock, per-split results |
| `regime_evaluation.csv` | Per-regime evaluation results |

---

## Configuration

All parameters are in `config.py`. Key settings:

```python
GPConfig(
    pop_size=2000,          # Population size
    n_generations=30,       # Max generations
    tournament_size=5,      # Selection pressure
    crossover_prob=0.7,     # Crossover probability
    mutation_prob=0.2,      # Mutation probability
    elite_size=10,          # Elitism count
    max_tree_depth=8,       # Max tree depth
    max_tree_nodes=50,      # Max tree nodes
    lambda_robustness=0.5,  # Multi-stock penalty
    chunk_size=200,         # Parallel eval chunk size
    checkpoint_every=5,     # Checkpoint frequency
)
```

---

## Extending the System

### Adding New Features
1. Add computation function in `feature_engineering.py`
2. Add to the appropriate feature list in `config.py` (e.g., `V2_GP_FEATURES`)
3. Update `build_features_for_stock()` in `feature_engineering.py`

### Adding New Primitives
1. Add safe function in `gp_primitives.py`
2. Register in `build_primitive_set()`

### Adding New Stocks
1. Add symbol to `TRADABLE_STOCKS` in `config.py`
2. Ensure data exists in `gp_research.duckdb`
3. Rebuild features: `python -m gp_system_complete.main_build_features`