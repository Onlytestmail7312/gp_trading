"""
evaluate_results.py -- load best model and show trading performance
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dill
import numpy as np
import pandas as pd
from pathlib import Path

from config import ALL_SYMBOLS, DAILY_FEATURES, OUTPUT_DIR
from dataset_builder import load_features, split_dataset, prepare_stock_data
from backtester import backtest_portfolio, generate_signal
from utils import get_logger

log = get_logger()


def load_best_model(path="gp_output/best_model.pkl"):
    # Must initialize DEAP creator before loading
    from gp_individual import setup_gp_toolbox
    from config import DAILY_FEATURES
    setup_gp_toolbox(feature_names=DAILY_FEATURES)
    with open(path, "rb") as f:
        data = dill.load(f)
    return data


def print_results(results, period_name):
    print(f"\n{'='*65}")
    print(f"  {period_name} RESULTS")
    print(f"{'='*65}")
    print(f"  {'Symbol':<12} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} {'WinRate%':>9} {'MaxDD%':>7} {'AvgHold':>8}")
    print(f"  {'-'*62}")

    total_trades = 0
    returns = []
    sharpes = []

    for sym, r in sorted(results.items()):
        print(f"  {sym:<12} {r.n_trades:>6} {r.total_return_pct:>8.1f} "
              f"{r.sharpe_ratio:>7.3f} {r.win_rate:>9.1f} "
              f"{r.max_drawdown_pct:>7.1f} {r.avg_hold_days:>8.1f}")
        total_trades += r.n_trades
        returns.append(r.annual_return_pct)
        sharpes.append(r.sharpe_ratio)

    print(f"  {'-'*62}")
    print(f"  {'AVERAGE':<12} {total_trades:>6} {np.mean(returns):>8.1f} "
          f"{np.mean(sharpes):>7.3f}")
    print(f"  Total trades: {total_trades}")


def plot_equity_curves(results, period_name, save_path=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Equity Curves -- {period_name}", fontsize=14, fontweight='bold')
        axes = axes.flatten()

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

        for idx, (sym, r) in enumerate(sorted(results.items())):
            ax = axes[idx]
            equity = r.equity_curve
            x = np.arange(len(equity))

            ax.plot(x, (equity - 1) * 100, color=colors[idx], linewidth=1.5)
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
            ax.fill_between(x, (equity - 1) * 100, 0,
                          where=(equity >= 1), alpha=0.1, color='green')
            ax.fill_between(x, (equity - 1) * 100, 0,
                          where=(equity < 1), alpha=0.1, color='red')

            ax.set_title(f"{sym}\nReturn: {r.total_return_pct:.1f}%  "
                        f"Sharpe: {r.sharpe_ratio:.2f}  "
                        f"Trades: {r.n_trades}",
                        fontsize=9)
            ax.set_xlabel("Trading Days")
            ax.set_ylabel("Return %")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Chart saved: {save_path}")
        plt.show()

    except ImportError:
        print("  matplotlib not available -- skipping charts")


def plot_trade_distribution(results, period_name, save_path=None):
    try:
        import matplotlib.pyplot as plt

        all_returns = []
        all_holds   = []
        symbols     = []

        for sym, r in results.items():
            for t in r.trades:
                all_returns.append(t.net_return * 100)
                all_holds.append(t.hold_days)
                symbols.append(sym)

        if not all_returns:
            print("  No trades to plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Trade Analysis -- {period_name}", fontsize=13, fontweight='bold')

        # Return distribution
        ax = axes[0]
        wins   = [r for r in all_returns if r > 0]
        losses = [r for r in all_returns if r <= 0]
        ax.hist(wins,   bins=30, color='green', alpha=0.6, label=f'Wins ({len(wins)})')
        ax.hist(losses, bins=30, color='red',   alpha=0.6, label=f'Losses ({len(losses)})')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_title("Trade Return Distribution")
        ax.set_xlabel("Return %")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Hold days distribution
        ax = axes[1]
        ax.hist(all_holds, bins=30, color='#2196F3', alpha=0.7)
        ax.set_title("Hold Days Distribution")
        ax.set_xlabel("Days Held")
        ax.set_ylabel("Count")
        ax.axvline(x=np.mean(all_holds), color='red', linewidth=1.5,
                  label=f'Mean: {np.mean(all_holds):.1f}d')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Per-stock trade count
        ax = axes[2]
        sym_counts = {}
        for sym, r in results.items():
            sym_counts[sym] = r.n_trades
        bars = ax.bar(sym_counts.keys(), sym_counts.values(),
                     color=colors[:len(sym_counts)])
        ax.set_title("Trades per Stock")
        ax.set_xlabel("Symbol")
        ax.set_ylabel("Number of Trades")
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, sym_counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(val), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Chart saved: {save_path}")
        plt.show()

    except ImportError:
        print("  matplotlib not available -- skipping charts")


colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']


def main():
    print("\n" + "="*65)
    print("  GP TRADING SYSTEM -- RESULTS EVALUATION")
    print("="*65)

    # Load model
    model_path = "gp_output/best_model.pkl"
    if not Path(model_path).exists():
        print(f"ERROR: No model found at {model_path}")
        print("Run main_train_gp.py first")
        return

    model = load_best_model(model_path)
    print(f"\n  Model loaded:")
    print(f"  Formula   : {model.get('formula', 'N/A')}")
    print(f"  Fitness   : {model.get('fitness', 'N/A'):.4f}")
    print(f"  Tree size : {model.get('tree_size', 'N/A')} nodes")
    print(f"  Generation: {model.get('generation', 'N/A')}")

    # Load data
    df = load_features(Path("gp_output/gp_features_daily.parquet"))
    train_df, val_df, test_df = split_dataset(df)

    train_data = prepare_stock_data(train_df, ALL_SYMBOLS)
    val_data   = prepare_stock_data(val_df,   ALL_SYMBOLS)
    test_data  = prepare_stock_data(test_df,  ALL_SYMBOLS)

    # Get compiled function
    from gp_individual import setup_gp_toolbox
    toolbox, pset = setup_gp_toolbox(feature_names=DAILY_FEATURES)

    individual = model["individual"]
    func = toolbox.compile(expr=individual)

    print(f"\n  Running backtest on all periods...")

    # Backtest all periods
    train_results = backtest_portfolio(func, train_data)
    val_results   = backtest_portfolio(func, val_data)
    test_results  = backtest_portfolio(func, test_data)

    # Print results
    print_results(train_results, "TRAIN (2021-2023)")
    print_results(val_results,   "VALIDATION (2024)")
    print_results(test_results,  "TEST (2025-2026)")

    # Save charts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_equity_curves(train_results, "Train",
                      str(OUTPUT_DIR / "equity_train.png"))
    plot_equity_curves(val_results,   "Validation",
                      str(OUTPUT_DIR / "equity_val.png"))
    plot_equity_curves(test_results,  "Test",
                      str(OUTPUT_DIR / "equity_test.png"))

    plot_trade_distribution(train_results, "Train",
                           str(OUTPUT_DIR / "trades_train.png"))
    plot_trade_distribution(test_results,  "Test",
                           str(OUTPUT_DIR / "trades_test.png"))

    print(f"\n  Charts saved to: {OUTPUT_DIR}")
    print("\n" + "="*65)


if __name__ == "__main__":
    main()
