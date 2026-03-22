"""
backtest_all_stocks.py -- run backtest on all stocks in DB and generate entry/exit signals
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dill
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import DAILY_FEATURES
from gp_individual import setup_gp_toolbox
from backtester import backtest_portfolio, generate_signal, get_nifty_regime
from feature_engineering import build_daily_features
from data_loader import load_symbol, validate_data
from config import DB_PATH, NIFTY_SYMBOL
from utils import get_logger

log = get_logger()


def load_model(path):
    setup_gp_toolbox(feature_names=DAILY_FEATURES)
    with open(path, "rb") as f:
        return dill.load(f)


def get_all_symbols():
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    syms = con.execute(
        "SELECT DISTINCT symbol FROM min_1m WHERE symbol != 'NIFTY' ORDER BY symbol"
    ).fetchall()
    con.close()
    return [s[0] for s in syms]


def prepare_stock(sym, nifty_1m):
    try:
        raw = load_symbol(sym, DB_PATH)
        if len(raw) == 0:
            return None
        validate_data(raw, sym)
        feat_df = build_daily_features(raw, nifty_1m, sym)
        return feat_df
    except Exception as e:
        log.warning(f"  {sym}: failed -- {e}")
        return None


def generate_trade_signals(func, feat_df, sym):
    """Generate entry/exit signals with dates and prices"""
    features = feat_df[DAILY_FEATURES].values.astype(np.float64)
    prices   = feat_df["close"].values.astype(np.float64)
    dates    = feat_df.index

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    signal = generate_signal(func, features)
    regime = get_nifty_regime(features)

    trades = []
    in_trade    = False
    direction   = 0
    entry_day   = 0
    entry_price = 0.0
    entry_sig   = 0.0
    peak_price  = 0.0

    from config import (
        STOP_LOSS_PCT, TRAILING_STOP_PCT,
        MIN_HOLD_DAYS, MAX_HOLD_DAYS, TOTAL_COST_PCT,
    )
    STOP_LOSS     = STOP_LOSS_PCT
    TRAILING_STOP = TRAILING_STOP_PCT
    MIN_HOLD      = MIN_HOLD_DAYS
    MAX_HOLD      = MAX_HOLD_DAYS
    COST          = TOTAL_COST_PCT

    for i in range(1, len(signal)):
        prev_sig    = signal[i - 1]
        curr_sig    = signal[i]
        price       = prices[i]
        hold        = i - entry_day
        curr_regime = regime[i]

        if not in_trade:
            if curr_regime == 1 and prev_sig <= 0 and curr_sig > 0:
                in_trade    = True
                direction   = 1
                entry_day   = i
                entry_price = price * (1 + COST)
                entry_sig   = curr_sig
                peak_price  = entry_price

            elif curr_regime == -1 and prev_sig >= 0 and curr_sig < 0:
                in_trade    = True
                direction   = -1
                entry_day   = i
                entry_price = price * (1 - COST)
                entry_sig   = curr_sig
                peak_price  = entry_price

        else:
            exit_signal  = False
            exit_reason  = ""
            exit_price   = price

            # Update peak
            if direction == 1:
                peak_price = max(peak_price, price)
            else:
                peak_price = min(peak_price, price)

            # Hard stop loss
            if direction == 1 and price <= entry_price * (1 - STOP_LOSS):
                exit_signal = True
                exit_reason = "STOP_LOSS"
                exit_price  = entry_price * (1 - STOP_LOSS)

            elif direction == -1 and price >= entry_price * (1 + STOP_LOSS):
                exit_signal = True
                exit_reason = "STOP_LOSS"
                exit_price  = entry_price * (1 + STOP_LOSS)

            # Trailing stop
            if not exit_signal:
                if direction == 1 and price <= peak_price * (1 - TRAILING_STOP):
                    exit_signal = True
                    exit_reason = "TRAIL_STOP"
                    exit_price  = peak_price * (1 - TRAILING_STOP)

                elif direction == -1 and price >= peak_price * (1 + TRAILING_STOP):
                    exit_signal = True
                    exit_reason = "TRAIL_STOP"
                    exit_price  = peak_price * (1 + TRAILING_STOP)

            # Signal exit
            if not exit_signal and hold >= MIN_HOLD:
                if direction == 1 and curr_sig <= 0:
                    exit_signal = True
                    exit_reason = "SIGNAL"
                elif direction == -1 and curr_sig >= 0:
                    exit_signal = True
                    exit_reason = "SIGNAL"

            # Regime flip
            if not exit_signal:
                if direction == 1 and curr_regime == -1:
                    exit_signal = True
                    exit_reason = "REGIME_FLIP"
                elif direction == -1 and curr_regime == 1:
                    exit_signal = True
                    exit_reason = "REGIME_FLIP"

            # Max hold
            if hold >= MAX_HOLD:
                exit_signal = True
                exit_reason = "MAX_HOLD"

            if exit_signal:
                if direction == 1:
                    exit_price   = exit_price * (1 - COST)
                    pnl          = (exit_price / entry_price - 1) * 100
                else:
                    exit_price   = exit_price * (1 + COST)
                    pnl          = (entry_price / exit_price - 1) * 100

                trades.append({
                    "symbol":        sym,
                    "direction":     "LONG" if direction == 1 else "SHORT",
                    "entry_date":    str(dates[entry_day])[:10],
                    "exit_date":     str(dates[i])[:10],
                    "entry_price":   round(entry_price, 2),
                    "exit_price":    round(exit_price, 2),
                    "hold_days":     hold,
                    "pnl_pct":       round(pnl, 2),
                    "exit_reason":   exit_reason,
                    "entry_signal":  round(entry_sig, 4),
                    "exit_signal":   round(curr_sig, 4),
                })
                in_trade = False

    # Open trade
    if in_trade:
        i     = len(signal) - 1
        price = prices[i]
        if direction == 1:
            exit_price = price * (1 - COST)
            pnl        = (exit_price / entry_price - 1) * 100
        else:
            exit_price = price * (1 + COST)
            pnl        = (entry_price / exit_price - 1) * 100

        trades.append({
            "symbol":        sym,
            "direction":     "LONG" if direction == 1 else "SHORT",
            "entry_date":    str(dates[entry_day])[:10],
            "exit_date":     "OPEN",
            "entry_price":   round(entry_price, 2),
            "exit_price":    round(exit_price, 2),
            "hold_days":     i - entry_day,
            "pnl_pct":       round(pnl, 2),
            "exit_reason":   "OPEN_TRADE",
            "entry_signal":  round(entry_sig, 4),
            "exit_signal":   round(signal[i], 4),
        })

    return trades


def browse_model_path():
    """Ask user to select model file"""
    import tkinter as tk
    from tkinter import filedialog

    # Try GUI file browser first
    try:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select GP Model File",
            initialdir="gp_output",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        root.destroy()
        if path:
            return path
    except Exception:
        pass

    # Fallback to console input
    print("\nAvailable models in gp_output/:")
    import glob
    models = glob.glob("gp_output/*.pkl")
    for i, m in enumerate(models):
        print(f"  {i+1}. {m}")

    print("\nEnter model path (or press Enter for default best_model.pkl):")
    path = input("  > ").strip()

    if not path:
        path = "gp_output/best_model.pkl"

    return path


def main():
    print("\n" + "="*70)
    print("  GP BACKTEST -- ALL STOCKS -- ENTRY/EXIT SIGNALS")
    print("="*70)

    # Ask user for model path
    print("\n  Select model file...")
    model_path = browse_model_path()

    if not model_path or not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        return

    model = load_model(model_path)
    print(f"\n  Model loaded  : {model_path}")
    print(f"  Model fitness : {model['fitness']:.4f}")
    print(f"  Formula       : {str(model['individual'])[:80]}...")

    toolbox, _ = setup_gp_toolbox(feature_names=DAILY_FEATURES)
    func = toolbox.compile(expr=model["individual"])

    # Load Nifty
    log.info("Loading NIFTY...")
    nifty_1m = load_symbol(NIFTY_SYMBOL, DB_PATH)

    # Get all symbols
    symbols = get_all_symbols()
    print(f"\n  Testing on {len(symbols)} stocks: {symbols}")

    all_trades = []

    for sym in symbols:
        log.info(f"Processing {sym}...")
        feat_df = prepare_stock(sym, nifty_1m)
        if feat_df is None or len(feat_df) < 50:
            continue

        trades = generate_trade_signals(func, feat_df, sym)
        all_trades.extend(trades)
        log.info(f"  {sym}: {len(trades)} trades")

    # Convert to DataFrame
    df = pd.DataFrame(all_trades)

    if df.empty:
        print("No trades generated")
        return

    # Save to CSV
    output_path = "gp_output/all_trades.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Saved {len(df)} trades to {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  TRADE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total trades  : {len(df)}")
    print(f"  Open trades   : {len(df[df['exit_date']=='OPEN'])}")
    print(f"  Closed trades : {len(df[df['exit_date']!='OPEN'])}")

    closed = df[df['exit_date'] != 'OPEN']
    if len(closed) > 0:
        print(f"  Win rate      : {(closed['pnl_pct'] > 0).mean()*100:.1f}%")
        print(f"  Avg PnL       : {closed['pnl_pct'].mean():.2f}%")
        print(f"  Best trade    : {closed['pnl_pct'].max():.2f}%")
        print(f"  Worst trade   : {closed['pnl_pct'].min():.2f}%")
        print(f"  Avg hold days : {closed['hold_days'].mean():.1f}")

    # Show open trades
    open_trades = df[df['exit_date'] == 'OPEN']
    if len(open_trades) > 0:
        print(f"\n{'='*70}")
        print(f"  CURRENTLY OPEN TRADES")
        print(f"{'='*70}")
        print(f"  {'Symbol':<12} {'Dir':<6} {'Entry Date':<12} {'Entry':>8} {'Current':>8} {'PnL%':>7} {'Days':>5}")
        print(f"  {'-'*60}")
        for _, t in open_trades.iterrows():
            print(f"  {t['symbol']:<12} {t['direction']:<6} {t['entry_date']:<12} "
                  f"{t['entry_price']:>8.2f} {t['exit_price']:>8.2f} "
                  f"{t['pnl_pct']:>7.2f} {t['hold_days']:>5}")

    # Per stock summary
    print(f"\n{'='*70}")
    print(f"  PER STOCK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Symbol':<12} {'Trades':>6} {'WinRate%':>9} {'AvgPnL%':>8} {'TotalPnL%':>10}")
    print(f"  {'-'*50}")

    for sym in sorted(df['symbol'].unique()):
        sym_df = df[df['symbol'] == sym]
        closed_sym = sym_df[sym_df['exit_date'] != 'OPEN']
        if len(closed_sym) == 0:
            continue
        win_rate  = (closed_sym['pnl_pct'] > 0).mean() * 100
        avg_pnl   = closed_sym['pnl_pct'].mean()
        total_pnl = closed_sym['pnl_pct'].sum()
        print(f"  {sym:<12} {len(closed_sym):>6} {win_rate:>9.1f} {avg_pnl:>8.2f} {total_pnl:>10.2f}")

    print(f"\n  Full trade log saved to: {output_path}")


if __name__ == "__main__":
    main()
