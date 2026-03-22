"""
test_new_stocks.py -- test saved model on new stocks
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dill
import numpy as np
import pandas as pd
from pathlib import Path

from config import DAILY_FEATURES
from gp_individual import setup_gp_toolbox
from backtester import backtest_portfolio, generate_signal
from feature_engineering import build_daily_features
from data_loader import load_symbol, validate_data
from config import DB_PATH, NIFTY_SYMBOL
from utils import get_logger

log = get_logger()

NEW_STOCKS = ["BHARTIARTL", "TATAMOTORS", "HINDUNILVR", "SBIN", "SUNPHARMA"]

TRAIN_END  = pd.Timestamp("2024-01-01")
VAL_START  = pd.Timestamp("2024-01-01")
VAL_END    = pd.Timestamp("2025-01-01")
TEST_START = pd.Timestamp("2025-01-01")


def load_model(path):
    setup_gp_toolbox(feature_names=DAILY_FEATURES)
    with open(path, "rb") as f:
        return dill.load(f)


def prepare_data(symbols):
    log.info("Loading NIFTY...")
    nifty_1m = load_symbol(NIFTY_SYMBOL, DB_PATH)

    all_dfs = []
    for sym in symbols:
        try:
            log.info(f"Processing {sym}...")
            raw = load_symbol(sym, DB_PATH)
            if len(raw) == 0:
                log.warning(f"  {sym}: no data found -- skipping")
                continue
            validate_data(raw, sym)
            feat_df = build_daily_features(raw, nifty_1m, sym)
            all_dfs.append(feat_df)
        except Exception as e:
            log.warning(f"  {sym}: failed -- {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No data loaded")

    df = pd.concat(all_dfs).sort_index()
    return df


def prepare_stock_data(df, symbols, start, end):
    result = {}
    # Make timestamps tz-aware UTC for comparison
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_index()
        # Normalize index to UTC
        if sym_df.index.tz is None:
            sym_df.index = sym_df.index.tz_localize("UTC")
        else:
            sym_df.index = sym_df.index.tz_convert("UTC")
        sym_df = sym_df[(sym_df.index >= start) & (sym_df.index < end)]
        if len(sym_df) < 50:
            continue
        features = sym_df[DAILY_FEATURES].values.astype(np.float64)
        prices   = sym_df["close"].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        result[sym] = (features, prices)
        log.info(f"  {sym}: {len(sym_df)} bars")
    return result


def print_results(results, period_name):
    print(f"\n{'='*65}")
    print(f"  {period_name}")
    print(f"{'='*65}")
    print(f"  {'Symbol':<12} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} {'WinRate%':>9} {'MaxDD%':>7}")
    print(f"  {'-'*55}")
    returns = []
    sharpes = []
    for sym, r in sorted(results.items()):
        print(f"  {sym:<12} {r.n_trades:>6} {r.total_return_pct:>8.1f} "
              f"{r.sharpe_ratio:>7.3f} {r.win_rate:>9.1f} {r.max_drawdown_pct:>7.1f}")
        returns.append(r.total_return_pct)
        sharpes.append(r.sharpe_ratio)
    print(f"  {'-'*55}")
    print(f"  {'AVERAGE':<12} {'':>6} {np.mean(returns):>8.1f} {np.mean(sharpes):>7.3f}")


def main():
    print("\n" + "="*65)
    print("  TESTING V1 FORMULA ON NEW STOCKS")
    print("="*65)

    model = load_model("gp_output/best_model_v1_fitness17.pkl")
    print(f"\n  Fitness  : {model['fitness']:.4f}")

    toolbox, _ = setup_gp_toolbox(feature_names=DAILY_FEATURES)
    func = toolbox.compile(expr=model["individual"])

    df = prepare_data(NEW_STOCKS)

    train_data = prepare_stock_data(df, NEW_STOCKS,
                                    pd.Timestamp("2021-01-01"), TRAIN_END)
    val_data   = prepare_stock_data(df, NEW_STOCKS, VAL_START, VAL_END)
    test_data  = prepare_stock_data(df, NEW_STOCKS, TEST_START,
                                    pd.Timestamp("2030-01-01"))

    train_results = backtest_portfolio(func, train_data)
    val_results   = backtest_portfolio(func, val_data)
    test_results  = backtest_portfolio(func, test_data)

    print_results(train_results, "TRAIN (2021-2023) - NEW STOCKS")
    print_results(val_results,   "VALIDATION (2024) - NEW STOCKS")
    print_results(test_results,  "TEST (2025-2026)  - NEW STOCKS")


if __name__ == "__main__":
    main()
