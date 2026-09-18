"""Evaluation metrics and a realistic trading backtest.

Kept free of any heavy ML dependency so it can be unit-tested in isolation.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------ price metrics ---
def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / np.where(actual == 0, np.nan, actual))) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2)) or 1e-12
    r2 = float(1 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def directional_accuracy(actual, predicted) -> float:
    """Fraction of times the model got the *direction* of change right."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 2:
        return 0.0
    a_dir = np.sign(np.diff(actual))
    p_dir = np.sign(np.diff(predicted))
    return float(np.mean(a_dir == p_dir))


# --------------------------------------------------------------- backtest ----
def backtest(actual, predicted, cost=0.0005, threshold=0.0) -> dict:
    """Long-only strategy: hold when the model predicts a rise above `threshold`.

    Returns strategy return, buy & hold return, Sharpe (annualised, ~252 days)
    and maximum drawdown. Trades incur `cost` each time we flip position.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 3:
        return {"strategy_ret": 0.0, "buyhold_ret": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "n_trades": 0}

    prev_close = actual[:-1]
    next_ret = np.diff(actual) / prev_close           # realised next-day return
    expected = (predicted[1:] - prev_close) / prev_close
    position = (expected > threshold).astype(float)    # 1 = long, 0 = flat

    # transaction cost charged whenever the position changes
    trades = np.abs(np.diff(np.concatenate([[0.0], position])))
    strat_ret = position * next_ret - trades * cost

    equity = np.cumprod(1.0 + strat_ret)
    bh_equity = np.cumprod(1.0 + next_ret)

    def sharpe(returns):
        sd = np.std(returns)
        return float(np.mean(returns) / sd * np.sqrt(252)) if sd > 0 else 0.0

    def max_dd(equity_curve):
        peak = np.maximum.accumulate(equity_curve)
        dd = equity_curve / peak - 1.0
        return float(dd.min())

    return {
        "strategy_ret": float(equity[-1] - 1.0),
        "buyhold_ret": float(bh_equity[-1] - 1.0),
        "sharpe": sharpe(strat_ret),
        "max_drawdown": max_dd(equity),
        "n_trades": int(trades.sum()),
    }


def confidence_from_error(predicted, rmse, z=1.0):
    """Simple prediction band ± z*RMSE around a point forecast."""
    predicted = np.asarray(predicted, dtype=float)
    return predicted - z * rmse, predicted + z * rmse
