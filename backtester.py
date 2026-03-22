"""
backtester.py -- positional long/short backtester.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from config import (
    MIN_HOLD_DAYS, MAX_HOLD_DAYS,
    TOTAL_COST_PCT, EPSILON,
)


@dataclass
class Trade:
    symbol:       str
    direction:    int
    entry_day:    int
    exit_day:     int
    entry_price:  float
    exit_price:   float
    weight:       float
    hold_days:    int
    gross_return: float
    net_return:   float
    signal_entry: float


@dataclass
class BacktestResult:
    symbol:            str
    trades:            List[Trade]
    equity_curve:      np.ndarray
    total_return_pct:  float
    annual_return_pct: float
    sharpe_ratio:      float
    sortino_ratio:     float
    max_drawdown_pct:  float
    win_rate:          float
    avg_hold_days:     float
    trades_per_year:   float
    n_trades:          int
    profit_factor:     float


def generate_signal(func, feature_matrix: np.ndarray) -> np.ndarray:
    n = feature_matrix.shape[0]
    signal = np.zeros(n)

    for i in range(n):
        row = feature_matrix[i]
        try:
            val = func(*row)
            signal[i] = float(val) if np.isfinite(val) else 0.0
        except Exception:
            signal[i] = 0.0

    return signal


def backtest_single_stock(
    signal: np.ndarray,
    prices: np.ndarray,
    symbol: str = "",
    min_hold: int = MIN_HOLD_DAYS,
    max_hold: int = MAX_HOLD_DAYS,
    cost_pct: float = TOTAL_COST_PCT,
) -> BacktestResult:
    n      = len(signal)
    trades = []

    in_trade    = False
    direction   = 0
    entry_day   = 0
    entry_price = 0.0
    entry_sig   = 0.0

    for i in range(1, n):
        prev_sig = signal[i - 1]
        curr_sig = signal[i]
        price    = prices[i]
        hold     = i - entry_day

        if not in_trade:
            if prev_sig <= 0 and curr_sig > 0:
                in_trade    = True
                direction   = 1
                entry_day   = i
                entry_price = price * (1 + cost_pct)
                entry_sig   = curr_sig
            elif prev_sig >= 0 and curr_sig < 0:
                in_trade    = True
                direction   = -1
                entry_day   = i
                entry_price = price * (1 - cost_pct)
                entry_sig   = curr_sig
        else:
            exit_signal = False
            if hold >= min_hold:
                if direction == 1 and curr_sig <= 0:
                    exit_signal = True
                elif direction == -1 and curr_sig >= 0:
                    exit_signal = True

            if hold >= max_hold:
                exit_signal = True

            if exit_signal:
                if direction == 1:
                    exit_price   = price * (1 - cost_pct)
                    gross_return = (exit_price / entry_price) - 1
                else:
                    exit_price   = price * (1 + cost_pct)
                    gross_return = (entry_price / exit_price) - 1

                trades.append(Trade(
                    symbol       = symbol,
                    direction    = direction,
                    entry_day    = entry_day,
                    exit_day     = i,
                    entry_price  = entry_price,
                    exit_price   = exit_price,
                    weight       = 1.0,
                    hold_days    = hold,
                    gross_return = gross_return,
                    net_return   = gross_return,
                    signal_entry = entry_sig,
                ))
                in_trade = False

    if in_trade and len(prices) > 0:
        i     = n - 1
        price = prices[i]
        hold  = i - entry_day

        if direction == 1:
            exit_price   = price * (1 - cost_pct)
            gross_return = (exit_price / entry_price) - 1
        else:
            exit_price   = price * (1 + cost_pct)
            gross_return = (entry_price / exit_price) - 1

        trades.append(Trade(
            symbol       = symbol,
            direction    = direction,
            entry_day    = entry_day,
            exit_day     = i,
            entry_price  = entry_price,
            exit_price   = exit_price,
            weight       = 1.0,
            hold_days    = hold,
            gross_return = gross_return,
            net_return   = gross_return,
            signal_entry = entry_sig,
        ))

    return _compute_metrics(trades, prices, symbol)


def _compute_metrics(
    trades: List[Trade],
    prices: np.ndarray,
    symbol: str,
) -> BacktestResult:
    n_days = len(prices)

    if not trades:
        return BacktestResult(
            symbol=symbol, trades=[],
            equity_curve=np.ones(n_days),
            total_return_pct=0.0, annual_return_pct=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown_pct=0.0, win_rate=0.0,
            avg_hold_days=0.0, trades_per_year=0.0,
            n_trades=0, profit_factor=0.0,
        )

    equity = np.ones(n_days)
    for t in trades:
        for day in range(t.exit_day, n_days):
            equity[day] *= (1 + t.net_return)

    daily_ret = np.diff(equity) / (equity[:-1] + EPSILON)

    n_trades      = len(trades)
    returns       = np.array([t.net_return for t in trades])
    hold_days     = np.array([t.hold_days  for t in trades])
    wins          = returns > 0
    win_rate      = wins.mean() if n_trades else 0.0
    avg_hold      = hold_days.mean() if n_trades else 0.0
    years         = n_days / 252.0
    trades_per_yr = n_trades / max(years, EPSILON)

    total_ret  = equity[-1] - 1.0
    annual_ret = (equity[-1] ** (1.0 / max(years, EPSILON))) - 1.0

    if len(daily_ret) > 1 and daily_ret.std() > EPSILON:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    downside = daily_ret[daily_ret < 0]
    if len(downside) > 1 and downside.std() > EPSILON:
        sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / (peak + EPSILON)
    max_dd = dd.min() * 100.0

    gross_wins   = returns[wins].sum() if wins.any() else 0.0
    gross_losses = -returns[~wins].sum() if (~wins).any() else EPSILON
    profit_factor = gross_wins / max(gross_losses, EPSILON)

    return BacktestResult(
        symbol            = symbol,
        trades            = trades,
        equity_curve      = equity,
        total_return_pct  = total_ret * 100.0,
        annual_return_pct = annual_ret * 100.0,
        sharpe_ratio      = sharpe,
        sortino_ratio     = sortino,
        max_drawdown_pct  = max_dd,
        win_rate          = win_rate * 100.0,
        avg_hold_days     = avg_hold,
        trades_per_year   = trades_per_yr,
        n_trades          = n_trades,
        profit_factor     = profit_factor,
    )


def backtest_portfolio(
    func,
    stock_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    min_hold: int = MIN_HOLD_DAYS,
    max_hold: int = MAX_HOLD_DAYS,
    cost_pct: float = TOTAL_COST_PCT,
) -> Dict[str, BacktestResult]:
    results = {}

    for symbol, (features, prices) in stock_data.items():
        try:
            signal = generate_signal(func, features)
            result = backtest_single_stock(signal, prices, symbol, min_hold, max_hold, cost_pct)
            results[symbol] = result
        except Exception:
            pass

    return results
