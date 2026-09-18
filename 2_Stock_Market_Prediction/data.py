"""Real market data acquisition + feature engineering.

Data comes from Yahoo Finance through `yfinance`. If the network or the
provider is unavailable we raise a clear error rather than silently inventing
numbers - this project never fabricates prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from config import BENCHMARK, DAYS


def fetch_prices(ticker: str, days: int = DAYS) -> pd.DataFrame:
    """Download real OHLCV history for `ticker`."""
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {ticker!r}. Check the ticker and your "
            "internet connection (Yahoo Finance may be rate limiting).")
    # yfinance sometimes returns a MultiIndex when several tickers are asked
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_benchmark(days: int = DAYS) -> pd.DataFrame | None:
    """Download the market benchmark (e.g. SPY) for relative reporting."""
    try:
        return fetch_prices(BENCHMARK, days)
    except Exception:
        return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features that are *known at prediction time* (no look-ahead)."""
    out = df.copy()
    close = out["Close"]
    ret = close.pct_change()

    out["ret_1"] = ret
    out["ret_5"] = close.pct_change(5)
    out["sma_10"] = close.rolling(10).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["vol_20"] = ret.rolling(20).std()
    out["mom_10"] = close - close.shift(10)

    # RSI(14) - a real momentum indicator, computed causally
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD
    out["macd"] = out["ema_12"] - close.ewm(span=26, adjust=False).mean()

    # volatility-scaled distance from the moving average (mean reversion)
    out["z_sma20"] = (close - out["sma_20"]) / out["sma_20"]

    out["target"] = close.shift(-1)               # next-day close
    out["target_ret"] = close.pct_change().shift(-1)  # next-day return
    return out.dropna()

# Convenience: what we actually regress on (returns extrapolate far better
# than raw prices when using tree models).
TARGET_RET = "target_ret"


FEATURE_COLS = ["ret_1", "ret_5", "sma_10", "sma_20", "sma_50", "ema_12",
                "vol_20", "mom_10", "rsi_14", "macd", "z_sma20"]
