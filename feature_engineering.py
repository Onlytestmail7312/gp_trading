"""
Feature engineering for GP trading system.

Computes normalized technical features from 1-minute OHLCV bars.
All features are ratios / returns / z-scores — no raw price levels.
"""

import numpy as np
import pandas as pd
from typing import Dict

from gp_system_complete.config import (
    BARS_PER_DAY,
    RETURN_PERIODS,
    SMA_PERIODS_BARS,
    ATR_PERIOD_BARS,
    VOLATILITY_PERIOD,
    VOLUME_REL_PERIODS,
    RANGE_PCT_PERIODS,
    TREND_50D_BARS,
    TREND_20D_BARS,
    FORWARD_RETURN_PERIODS,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MIN,
    SESSION_MINUTES,
    EPSILON,
)
from gp_system_complete.utils import get_logger


# ═══════════════════════════════════════════════════════════════════
# RETURN FEATURES
# ═══════════════════════════════════════════════════════════════════
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lagged return features.
    ret_n = close / close.shift(n) - 1
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for n in RETURN_PERIODS:
        out[f"ret_{n}m"] = close / close.shift(n) - 1
    return out


# ═══════════════════════════════════════════════════════════════════
# TREND / POSITION FEATURES
# ═══════════════════════════════════════════════════════════════════
def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trend-position features using SMAs.
    All expressed as ratios (normalized).
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    smas = {}
    for name, period in SMA_PERIODS_BARS.items():
        smas[name] = close.rolling(window=period, min_periods=max(period // 2, 1)).mean()

    # close vs SMA ratios
    out["close_vs_sma20"] = close / smas["sma20"].replace(0, np.nan) - 1
    out["close_vs_sma50"] = close / smas["sma50"].replace(0, np.nan) - 1
    out["close_vs_sma100"] = close / smas["sma100"].replace(0, np.nan) - 1
    out["close_vs_sma200"] = close / smas["sma200"].replace(0, np.nan) - 1

    # SMA cross ratios
    out["sma20_vs_sma50"] = smas["sma20"] / smas["sma50"].replace(0, np.nan) - 1
    out["sma50_vs_sma200"] = smas["sma50"] / smas["sma200"].replace(0, np.nan) - 1

    # 50-day trend return (uses daily close proxy: last bar each day)
    out["trend_50d_return"] = close / close.shift(TREND_50D_BARS) - 1

    return out


# ═══════════════════════════════════════════════════════════════════
# VOLATILITY FEATURES
# ═══════════════════════════════════════════════════════════════════
def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility features: ATR%, range%, realized vol.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    out = pd.DataFrame(index=df.index)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR as percentage of close
    atr = tr.rolling(window=ATR_PERIOD_BARS, min_periods=max(ATR_PERIOD_BARS // 2, 1)).mean()
    out["atr_pct"] = atr / close.replace(0, np.nan)

    # Range percentage
    for p in RANGE_PCT_PERIODS:
        period_bars = p * BARS_PER_DAY
        rolling_high = high.rolling(window=period_bars, min_periods=max(period_bars // 2, 1)).max()
        rolling_low = low.rolling(window=period_bars, min_periods=max(period_bars // 2, 1)).min()
        out[f"range_pct_{p}"] = (rolling_high - rolling_low) / close.replace(0, np.nan)

    # Realized volatility (std of 1m returns over N days)
    ret_1m = close.pct_change()
    out["volatility_20"] = ret_1m.rolling(
        window=VOLATILITY_PERIOD, min_periods=max(VOLATILITY_PERIOD // 2, 1)
    ).std() * np.sqrt(BARS_PER_DAY)  # annualize roughly

    return out


# ═══════════════════════════════════════════════════════════════════
# VOLUME FEATURES
# ═══════════════════════════════════════════════════════════════════
def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative volume features.
    """
    volume = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)

    for p in VOLUME_REL_PERIODS:
        period_bars = p * BARS_PER_DAY
        mean_vol = volume.rolling(
            window=period_bars, min_periods=max(period_bars // 2, 1)
        ).mean()
        out[f"volume_rel_{p}"] = volume / mean_vol.replace(0, np.nan)

    return out


# ═══════════════════════════════════════════════════════════════════
# SESSION / TIME FEATURES
# ═══════════════════════════════════════════════════════════════════
def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session/time features from the timestamp index.
    """
    idx = df.index
    out = pd.DataFrame(index=idx)

    # Convert to IST if UTC
    if hasattr(idx, "tz") and idx.tz is not None:
        ist_idx = idx.tz_convert("Asia/Kolkata")
    else:
        ist_idx = idx

    # Minutes from market open
    minutes_from_open = (
        (ist_idx.hour - MARKET_OPEN_HOUR) * 60
        + (ist_idx.minute - MARKET_OPEN_MIN)
    )
    out["minutes_from_open_norm"] = minutes_from_open / SESSION_MINUTES
    out["minutes_to_close_norm"] = 1.0 - out["minutes_from_open_norm"]

    # Day of week (0=Monday, 4=Friday) normalized to [0, 1]
    out["day_of_week"] = ist_idx.dayofweek / 4.0

    # Opening/closing phase flags (first/last 15 minutes)
    out["is_opening_phase"] = (minutes_from_open <= 15).astype(float)
    out["is_closing_phase"] = (minutes_from_open >= SESSION_MINUTES - 15).astype(float)

    return out


# ═══════════════════════════════════════════════════════════════════
# FORWARD RETURNS (for diagnostics)
# ═══════════════════════════════════════════════════════════════════
def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward returns for evaluation / diagnostics.
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for n in FORWARD_RETURN_PERIODS:
        out[f"fwd_ret_{n}m"] = close.shift(-n) / close - 1
    return out


# ═══════════════════════════════════════════════════════════════════
# NIFTY CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════════
def compute_nifty_features(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NIFTY context features (same logic as stock, prefixed).
    """
    close = nifty_df["close"]
    out = pd.DataFrame(index=nifty_df.index)

    # Returns
    for n in [1, 5, 15, 60]:
        out[f"nifty_ret_{n}m"] = close / close.shift(n) - 1

    # Trend
    sma50 = close.rolling(
        window=SMA_PERIODS_BARS["sma50"],
        min_periods=max(SMA_PERIODS_BARS["sma50"] // 2, 1),
    ).mean()
    sma200 = close.rolling(
        window=SMA_PERIODS_BARS["sma200"],
        min_periods=max(SMA_PERIODS_BARS["sma200"] // 2, 1),
    ).mean()

    out["nifty_close_vs_sma50"] = close / sma50.replace(0, np.nan) - 1
    out["nifty_close_vs_sma200"] = close / sma200.replace(0, np.nan) - 1

    # Market return 20d
    out["market_return_20d"] = close / close.shift(TREND_20D_BARS) - 1

    return out


# ═══════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH
# ═══════════════════════════════════════════════════════════════════
def compute_relative_strength(
    stock_df: pd.DataFrame,
    nifty_features: pd.DataFrame,
    stock_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute stock vs NIFTY relative strength features.
    """
    out = pd.DataFrame(index=stock_df.index)

    # Align NIFTY features to stock index
    nifty_aligned = nifty_features.reindex(stock_df.index, method="ffill")

    # Short-term RS
    if "ret_5m" in stock_returns.columns and "nifty_ret_5m" in nifty_aligned.columns:
        out["rs_5m"] = stock_returns["ret_5m"] - nifty_aligned["nifty_ret_5m"]
    if "ret_15m" in stock_returns.columns and "nifty_ret_15m" in nifty_aligned.columns:
        out["rs_15m"] = stock_returns["ret_15m"] - nifty_aligned["nifty_ret_15m"]
    if "ret_60m" in stock_returns.columns and "nifty_ret_60m" in nifty_aligned.columns:
        out["rs_60m"] = stock_returns["ret_60m"] - nifty_aligned["nifty_ret_60m"]

    # 20-day RS
    stock_close = stock_df["close"]
    stock_20d = stock_close / stock_close.shift(TREND_20D_BARS) - 1
    nifty_20d = nifty_aligned.get("market_return_20d", pd.Series(0, index=stock_df.index))
    out["rs_20d"] = stock_20d - nifty_20d

    return out


# ═══════════════════════════════════════════════════════════════════
# MASTER: BUILD ALL FEATURES FOR ONE STOCK
# ═══════════════════════════════════════════════════════════════════
def build_features_for_stock(
    stock_df: pd.DataFrame,
    nifty_features: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Build the complete feature set for a single stock.

    Parameters
    ----------
    stock_df : DataFrame with OHLCV for one stock (ts index)
    nifty_features : DataFrame of precomputed NIFTY features (ts index)
    symbol : stock symbol string

    Returns
    -------
    DataFrame with OHLCV + all features + forward returns + symbol column
    """
    log = get_logger()
    log.info(f"Computing features for {symbol}...")

    # Stock features
    returns = compute_returns(stock_df)
    trend = compute_trend_features(stock_df)
    vol = compute_volatility_features(stock_df)
    volume = compute_volume_features(stock_df)
    time_feat = compute_time_features(stock_df)
    fwd = compute_forward_returns(stock_df)

    # Relative strength
    rs = compute_relative_strength(stock_df, nifty_features, returns)

    # Align NIFTY features
    nifty_aligned = nifty_features.reindex(stock_df.index, method="ffill")

    # Combine everything
    result = pd.concat(
        [
            stock_df,           # raw OHLCV
            returns,            # ret_1m, ret_5m, ...
            trend,              # close_vs_sma20, ...
            vol,                # atr_pct, range_pct_5, ...
            volume,             # volume_rel_5, ...
            nifty_aligned,      # nifty_ret_5m, ...
            rs,                 # rs_5m, rs_15m, ...
            time_feat,          # minutes_from_open_norm, ...
            fwd,                # fwd_ret_5m, ...
        ],
        axis=1,
    )

    result["symbol"] = symbol

    # Drop rows with NaN in critical features (due to warm-up)
    initial_rows = len(result)
    critical_cols = [c for c in result.columns if c not in ["symbol"] and not c.startswith("fwd_")]
    result = result.dropna(subset=critical_cols)
    dropped = initial_rows - len(result)

    log.info(
        f"  {symbol}: {len(result):,} rows after warm-up "
        f"(dropped {dropped:,} NaN rows)"
    )

    return result