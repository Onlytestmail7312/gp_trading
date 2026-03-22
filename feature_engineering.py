"""
feature_engineering.py -- compute DAILY features from 1-minute OHLCV data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from utils import get_logger
from config import DAILY_FEATURES

log = get_logger()


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("1D").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    daily = daily[daily["volume"] > 0].copy()
    return daily


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _macd(s: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = _ema(s, fast)
    ema_slow    = _ema(s, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_pct(s: pd.Series, n: int = 20) -> pd.Series:
    sma   = _sma(s, n)
    std   = s.rolling(n).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (s - lower) / (upper - lower + 1e-8)


def compute_daily_features(
    daily: pd.DataFrame,
    nifty_daily: Optional[pd.DataFrame] = None,
    symbol: str = "",
) -> pd.DataFrame:
    df = daily.copy()
    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]

    # Returns
    df["ret_1d"]  = c.pct_change(1)
    df["ret_5d"]  = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)
    df["ret_20d"] = c.pct_change(20)

    # Trend
    df["close_vs_sma10"]  = c / _sma(c, 10)  - 1
    df["close_vs_sma20"]  = c / _sma(c, 20)  - 1
    df["close_vs_sma50"]  = c / _sma(c, 50)  - 1
    df["close_vs_sma200"] = c / _sma(c, 200) - 1

    # Momentum
    df["rsi_14"] = _rsi(c, 14) / 100.0
    _, sig, hist = _macd(c)
    df["macd_signal"] = sig  / (c + 1e-8)
    df["macd_hist"]   = hist / (c + 1e-8)

    # Volatility
    df["atr_pct_14"] = _atr(h, l, c, 14) / (c + 1e-8)
    df["bb_pct"]     = _bollinger_pct(c, 20)

    # Volume
    vol_sma = _sma(v.astype(float), 20)
    df["volume_rel_20"] = v / (vol_sma + 1e-8)

    # Nifty / Relative strength
    if nifty_daily is not None:
        nc = nifty_daily["close"].reindex(df.index, method="ffill")
        df["nifty_ret_5d"]   = nc.pct_change(5)
        df["nifty_ret_20d"]  = nc.pct_change(20)
        df["nifty_vs_sma50"] = nc / _sma(nc, 50) - 1
        df["rs_5d"]  = df["ret_5d"]  - nc.pct_change(5)
        df["rs_20d"] = df["ret_20d"] - nc.pct_change(20)
    else:
        for col in ["nifty_ret_5d","nifty_ret_20d","nifty_vs_sma50","rs_5d","rs_20d"]:
            df[col] = 0.0

    n_before = len(df)
    df = df.dropna(subset=DAILY_FEATURES)
    n_after  = len(df)

    if symbol:
        log.info(f"  {symbol}: {n_after} daily bars (dropped {n_before - n_after} NaN rows)")

    return df


def build_daily_features(
    df_1m: pd.DataFrame,
    nifty_1m: Optional[pd.DataFrame] = None,
    symbol: str = "",
) -> pd.DataFrame:
    daily = resample_to_daily(df_1m)
    nifty_daily = resample_to_daily(nifty_1m) if nifty_1m is not None else None

    feat_df = compute_daily_features(daily, nifty_daily, symbol)
    feat_df["symbol"] = symbol
    return feat_df
