from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Optional
from utils import get_logger
from config import DAILY_FEATURES

log = get_logger()


def resample_to_daily(df: pd.DataFrame, require_volume: bool = True) -> pd.DataFrame:
    df2 = df.copy()
    if df2.index.tz is not None:
        df2.index = df2.index.normalize()
    daily = df2.resample("1D").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })
    daily = daily.dropna(subset=["close"])
    if require_volume:
        daily = daily[daily["volume"] > 0].copy()
    else:
        daily = daily[daily["close"] > 0].copy()
    return daily


def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def _atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _macd(s, fast=12, slow=26, signal=9):
    ema_fast    = _ema(s, fast)
    ema_slow    = _ema(s, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger_pct(s, n=20):
    sma   = _sma(s, n)
    std   = s.rolling(n).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (s - lower) / (upper - lower + 1e-8)


def compute_daily_features(
    daily: pd.DataFrame,
    nifty_close: Optional[pd.Series] = None,
    symbol: str = "",
) -> pd.DataFrame:
    df = daily.copy()
    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]

    df["ret_1d"]  = c.pct_change(1)
    df["ret_5d"]  = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)
    df["ret_20d"] = c.pct_change(20)

    df["close_vs_sma10"]  = c / _sma(c, 10)  - 1
    df["close_vs_sma20"]  = c / _sma(c, 20)  - 1
    df["close_vs_sma50"]  = c / _sma(c, 50)  - 1
    df["close_vs_sma200"] = c / _sma(c, 200) - 1

    df["rsi_14"] = _rsi(c, 14) / 100.0
    _, sig, hist = _macd(c)
    df["macd_signal"] = sig  / (c + 1e-8)
    df["macd_hist"]   = hist / (c + 1e-8)

    df["atr_pct_14"] = _atr(h, l, c, 14) / (c + 1e-8)
    df["bb_pct"]     = _bollinger_pct(c, 20)

    vol_sma = _sma(v.astype(float), 20)
    df["volume_rel_20"] = v / (vol_sma + 1e-8)

    if nifty_close is not None:
        nc = nifty_close
        matched = nc.notna().sum()
        log.info(f"  {symbol}: nifty matched {matched}/{len(df)} dates")
        nc = nc.ffill().bfill()
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
    # Resample stock to daily
    daily = resample_to_daily(df_1m)

    nifty_close = None
    if nifty_1m is not None:
        # Resample nifty to daily
        nifty_daily = resample_to_daily(nifty_1m, require_volume=False)

        # Convert both indexes to plain date strings for alignment
        nc = nifty_daily["close"].copy()

        # Build a simple dict: date_string -> close_price
        nifty_dict = {str(idx)[:10]: val for idx, val in nc.items()}

        # Map stock dates to nifty prices
        aligned_values = []
        for idx in daily.index:
            date_str = str(idx)[:10]
            aligned_values.append(nifty_dict.get(date_str, np.nan))

        nc_aligned = pd.Series(aligned_values, index=daily.index)

        # Forward fill then backward fill missing values
        nc_aligned = nc_aligned.ffill().bfill()

        matched = (~nc_aligned.isna()).sum()
        log.info(f"  {symbol}: nifty matched {matched}/{len(daily)} dates")

        MIN_NIFTY_MATCH_PCT = 0.8
        match_pct = matched / max(len(daily), 1)
        if match_pct >= MIN_NIFTY_MATCH_PCT:
            nifty_close = nc_aligned
        else:
            log.warning(
                f"Nifty match rate too low "
                f"({match_pct:.1%}), setting Nifty features to zero"
            )
            nifty_close = None

    feat_df = compute_daily_features(daily, nifty_close, symbol)
    feat_df["symbol"] = symbol
    return feat_df
