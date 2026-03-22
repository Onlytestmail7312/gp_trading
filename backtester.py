"""
Backtester — per-stock backtest engine using vectorised simulation.

Spec Section 10: Backtesting
- Long-only strategy (simplification: position ∈ {0, 1})
- Signal > entry_threshold → go long
- Signal < exit_threshold → go flat
- Commission + slippage applied
- Returns a dict of performance metrics

Note: This is a lightweight vectorised backtester — no dependency on
backtesting.py library. Operates on 1-minute bars with pre-computed
GP signals.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .config import GPConfig, DEFAULT_GP_CONFIG, EPSILON, BARS_PER_DAY
from .gp_primitives import normalise_signal


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Container for backtest performance metrics."""
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    n_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return_pct: float = 0.0
    trades_per_day: float = 0.0
    exposure_pct: float = 0.0
    n_bars: int = 0
    n_days: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_return_pct": round(self.total_return_pct, 4),
            "annual_return_pct": round(self.annual_return_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "n_trades": self.n_trades,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "profit_factor": round(self.profit_factor, 4),
            "avg_trade_return_pct": round(self.avg_trade_return_pct, 4),
            "trades_per_day": round(self.trades_per_day, 4),
            "exposure_pct": round(self.exposure_pct, 2),
            "n_bars": self.n_bars,
            "n_days": round(self.n_days, 2),
        }

    def is_valid(self, cfg: GPConfig = None) -> bool:
        """Check if this result passes basic quality filters."""
        if cfg is None:
            cfg = DEFAULT_GP_CONFIG
        if self.n_trades < cfg.min_trades:
            return False
        if self.max_drawdown_pct < -cfg.max_drawdown_pct:
            return False
        if self.trades_per_day > cfg.max_trades_per_day:
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

def generate_signals(
    func,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply compiled GP function to feature matrix → normalised signals.

    Parameters
    ----------
    func : callable
        Compiled GP function taking N float args.
    feature_matrix : np.ndarray
        Shape (n_bars, n_features).

    Returns
    -------
    signals : np.ndarray
        Shape (n_bars,), values in [-1, 1].
    """
    n_bars = feature_matrix.shape[0]
    signals = np.zeros(n_bars, dtype=np.float64)

    for i in range(n_bars):
        try:
            row = feature_matrix[i]
            raw = float(func(*row))
            signals[i] = normalise_signal(raw)
        except Exception:
            signals[i] = 0.0

    return signals


def generate_signals_vectorised(
    func,
    feature_matrix: np.ndarray,
    batch_size: int = 10000,
) -> np.ndarray:
    """
    Vectorised signal generation with error handling in batches.
    Falls back to row-by-row for any batch that fails.
    """
    n_bars = feature_matrix.shape[0]
    signals = np.zeros(n_bars, dtype=np.float64)

    for start in range(0, n_bars, batch_size):
        end = min(start + batch_size, n_bars)
        batch = feature_matrix[start:end]

        try:
            # Try vectorised evaluation
            raw_signals = np.array([
                float(func(*batch[j])) for j in range(len(batch))
            ])
            # Apply tanh normalisation
            safe_raw = np.where(
                np.isfinite(raw_signals), raw_signals, 0.0
            )
            signals[start:end] = np.tanh(safe_raw)
        except Exception:
            # Fallback to row-by-row
            for j in range(len(batch)):
                try:
                    raw = float(func(*batch[j]))
                    signals[start + j] = normalise_signal(raw)
                except Exception:
                    signals[start + j] = 0.0

    return signals


# ═══════════════════════════════════════════════════════════════════════════
# VECTORISED BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(
    signals: np.ndarray,
    close_prices: np.ndarray,
    cfg: GPConfig = None,
) -> BacktestResult:
    """
    Run a vectorised backtest on pre-computed signals.

    Strategy Logic (long-only):
    - Signal > entry_threshold → position = 1 (long)
    - Signal < exit_threshold → position = 0 (flat)
    - Otherwise → hold current position

    Parameters
    ----------
    signals : np.ndarray
        Normalised signals in [-1, 1], shape (n_bars,).
    close_prices : np.ndarray
        Close prices, shape (n_bars,).
    cfg : GPConfig
        Configuration with thresholds, costs etc.

    Returns
    -------
    BacktestResult
        Performance metrics.
    """
    if cfg is None:
        cfg = DEFAULT_GP_CONFIG

    n_bars = len(signals)
    if n_bars < 2:
        return BacktestResult(n_bars=n_bars)

    entry_thresh = cfg.entry_threshold
    exit_thresh = cfg.exit_threshold
    commission = cfg.commission
    slippage = cfg.slippage

    # ── Compute positions ──────────────────────────────────────────────
    positions = np.zeros(n_bars, dtype=np.int8)  # 0=flat, 1=long

    for i in range(1, n_bars):
        if signals[i] > entry_thresh:
            positions[i] = 1
        elif signals[i] < exit_thresh:
            positions[i] = 0
        else:
            positions[i] = positions[i - 1]  # hold

    # ── Compute returns ────────────────────────────────────────────────
    price_returns = np.zeros(n_bars, dtype=np.float64)
    price_returns[1:] = (close_prices[1:] - close_prices[:-1]) / (
        close_prices[:-1] + EPSILON
    )

    # Strategy returns = position[i-1] * return[i] (use previous bar's position)
    strategy_returns = np.zeros(n_bars, dtype=np.float64)
    strategy_returns[1:] = positions[:-1] * price_returns[1:]

    # ── Apply transaction costs ────────────────────────────────────────
    position_changes = np.diff(positions, prepend=0)
    trades_mask = position_changes != 0
    n_trades_raw = int(np.sum(trades_mask))

    # Each trade costs commission + slippage
    cost_per_trade = commission + slippage
    trade_costs = np.zeros(n_bars, dtype=np.float64)
    trade_costs[trades_mask] = cost_per_trade
    strategy_returns -= trade_costs

    # ── Equity curve ───────────────────────────────────────────────────
    equity = np.cumprod(1.0 + strategy_returns)

    # ── Metrics ────────────────────────────────────────────────────────
    total_return_pct = (equity[-1] - 1.0) * 100.0

    # Trading days
    n_days = n_bars / BARS_PER_DAY
    n_years = n_days / 252.0

    # Annualised return
    if n_years > 0 and equity[-1] > 0:
        annual_return_pct = (
            (equity[-1] ** (1.0 / max(n_years, 0.01))) - 1.0
        ) * 100.0
    else:
        annual_return_pct = 0.0

    # Sharpe ratio (annualised, per-bar)
    bars_per_year = BARS_PER_DAY * 252
    if len(strategy_returns) > 1:
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)
        if std_ret > EPSILON:
            sharpe_ratio = (mean_ret / std_ret) * math.sqrt(bars_per_year)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (annualised, per-bar)
    downside_returns = strategy_returns[strategy_returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.std(downside_returns)
        if downside_std > EPSILON:
            mean_ret = np.mean(strategy_returns)
            sortino_ratio = (mean_ret / downside_std) * math.sqrt(bars_per_year)
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / (running_max + EPSILON)
    max_drawdown_pct = float(np.min(drawdowns)) * 100.0  # negative number

    # Trade-level analysis
    n_trades = n_trades_raw // 2  # entry+exit = 1 round trip
    n_trades = max(n_trades, n_trades_raw - 1)  # at least count transitions

    # Win rate from trade segments
    trade_returns = _compute_trade_returns(positions, price_returns)
    if len(trade_returns) > 0:
        n_trades = len(trade_returns)
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate_pct = (wins / n_trades) * 100.0

        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        profit_factor = (
            gross_profit / max(gross_loss, EPSILON)
        )
        avg_trade_return_pct = np.mean(trade_returns) * 100.0
    else:
        win_rate_pct = 0.0
        profit_factor = 0.0
        avg_trade_return_pct = 0.0

    # Trades per day
    trades_per_day = n_trades / max(n_days, 1.0)

    # Exposure
    exposure_pct = (np.sum(positions) / max(n_bars, 1)) * 100.0

    return BacktestResult(
        total_return_pct=total_return_pct,
        annual_return_pct=annual_return_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown_pct=max_drawdown_pct,
        n_trades=n_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        avg_trade_return_pct=avg_trade_return_pct,
        trades_per_day=trades_per_day,
        exposure_pct=exposure_pct,
        n_bars=n_bars,
        n_days=n_days,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TRADE-LEVEL RETURN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_trade_returns(
    positions: np.ndarray,
    price_returns: np.ndarray,
) -> list:
    """
    Compute per-trade cumulative returns.

    A trade starts when position goes from 0→1 and ends when 1→0.
    Returns a list of cumulative returns for each completed trade.
    """
    trade_returns = []
    in_trade = False
    cum_return = 0.0

    for i in range(1, len(positions)):
        if not in_trade and positions[i] == 1 and positions[i - 1] == 0:
            # Trade entry
            in_trade = True
            cum_return = 0.0

        if in_trade:
            cum_return += price_returns[i]

        if in_trade and positions[i] == 0 and positions[i - 1] == 1:
            # Trade exit
            trade_returns.append(cum_return)
            in_trade = False
            cum_return = 0.0

    # Close any open trade at end
    if in_trade:
        trade_returns.append(cum_return)

    return trade_returns


# ═══════════════════════════════════════════════════════════════════════════
# FULL BACKTEST FOR ONE STOCK (convenience function)
# ═══════════════════════════════════════════════════════════════════════════

def backtest_stock(
    func,
    feature_matrix: np.ndarray,
    close_prices: np.ndarray,
    cfg: GPConfig = None,
) -> BacktestResult:
    """
    End-to-end backtest: generate signals → run backtest.

    Parameters
    ----------
    func : callable
        Compiled GP function.
    feature_matrix : np.ndarray
        Shape (n_bars, n_features).
    close_prices : np.ndarray
        Shape (n_bars,).
    cfg : GPConfig
        Configuration.

    Returns
    -------
    BacktestResult
    """
    signals = generate_signals_vectorised(func, feature_matrix)
    return run_backtest(signals, close_prices, cfg)


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-STOCK BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def backtest_multi_stock(
    func,
    stock_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cfg: GPConfig = None,
) -> Dict[str, BacktestResult]:
    """
    Run backtest across multiple stocks.

    Parameters
    ----------
    func : callable
        Compiled GP function.
    stock_data : dict
        {symbol: (feature_matrix, close_prices)}
    cfg : GPConfig
        Configuration.

    Returns
    -------
    dict
        {symbol: BacktestResult}
    """
    results = {}
    for symbol, (features, prices) in stock_data.items():
        try:
            results[symbol] = backtest_stock(func, features, prices, cfg)
        except Exception as exc:
            results[symbol] = BacktestResult()  # Empty/failed result
    return results