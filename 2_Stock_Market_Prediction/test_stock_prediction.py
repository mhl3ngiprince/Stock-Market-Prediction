"""Tests for the stock pipeline - run with:  pytest -q"""
import tempfile

import numpy as np
import pandas as pd
import pytest

import database as db
import metrics as M


# ----------------------------------------------------------------- metrics ---
def test_regression_metrics_perfect():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    m = M.regression_metrics(a, a)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_imperfect():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 2.5, 3.5])
    m = M.regression_metrics(a, p)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(0.5)


def test_directional_accuracy():
    # perfectly rising series, correctly predicted direction
    a = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 2.1, 3.1, 4.1])
    assert M.directional_accuracy(a, p) == pytest.approx(1.0)


def test_backtest_flat_when_no_signal():
    a = np.array([100.0, 101.0, 102.0, 103.0])
    p = a.copy()  # predicts "no change" vs prev -> below threshold
    bt = M.backtest(a, p, threshold=0.05)
    assert bt["strategy_ret"] == pytest.approx(0.0, abs=1e-9)
    assert bt["buyhold_ret"] > 0


def test_backtest_captures_uptrend():
    a = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
    p = np.array([100.0, 103.0, 105.0, 107.0, 109.0])  # predicts rises
    bt = M.backtest(a, p, cost=0.0)
    assert bt["strategy_ret"] > 0
    assert bt["n_trades"] >= 1


def test_confidence_band_width():
    lo, hi = M.confidence_from_error(np.array([10.0, 20.0]), rmse=1.0, z=2.0)
    assert lo.tolist() == [8.0, 18.0]
    assert hi.tolist() == [12.0, 22.0]


# ---------------------------------------------------------------- database ---
@pytest.fixture()
def conn():
    return db.get_conn(tempfile.mktemp(suffix=".db"))


def _sample_df():
    idx = pd.date_range("2024-01-01", periods=5)
    return pd.DataFrame({"Open": np.arange(5.0),
                         "High": np.arange(5.0) + 1,
                         "Low": np.arange(5.0) - 1,
                         "Close": np.arange(5.0) + 0.5,
                         "Volume": np.full(5, 1000.0)}, index=idx)


def test_save_and_read_prices(conn):
    n = db.save_prices(conn, "TEST", _sample_df())
    assert n == 5
    rows = db.latest_prices(conn, "TEST", 10)
    assert len(rows) == 5
    assert rows[0]["close"] == pytest.approx(0.5)


def test_save_run_and_signal(conn):
    db.save_run(conn, "TEST", "next_close", "gbdt",
                {"n_train": 100, "n_test": 20, "mae": 1.0, "rmse": 1.5,
                 "r2": 0.9, "mape": 1.0, "dir_acc": 0.55, "strategy_ret": 0.1,
                 "buyhold_ret": 0.2, "sharpe": 1.2, "max_drawdown": -0.1})
    db.save_signal(conn, "TEST", "2024-01-05", "BUY", 0.8, 0.02, "test")
    run = conn.execute("SELECT * FROM runs WHERE ticker='TEST'").fetchone()
    sig = conn.execute("SELECT * FROM signals WHERE ticker='TEST'").fetchone()
    assert run["dir_acc"] == pytest.approx(0.55)
    assert sig["action"] == "BUY"


def test_prediction_roundtrip(conn):
    db.save_prediction(conn, "TEST", "2024-01-05", "gbdt", 101.0, 100.0, 1.0, 0.9)
    row = conn.execute("SELECT * FROM predictions").fetchone()
    assert row["predicted"] == pytest.approx(101.0)


# -------------------------------------------------------------------- data ---
def test_add_features_no_lookahead():
    """Features must not use future prices; target is the only forward value."""
    import data as data_mod
    idx = pd.date_range("2023-01-01", periods=120)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 120))
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": np.full(120, 1e6)}, index=idx)
    feats = data_mod.add_features(df)
    assert "rsi_14" in feats.columns
    assert "target_ret" in feats.columns
    # target is next-day return: pick a timestamp and compare against the raw
    # close series at that timestamp and the one after it.
    ts = feats.index[len(feats) // 2]
    pos = list(df.index).index(ts)
    expected = close[pos + 1] / close[pos] - 1
    assert feats.loc[ts, "target_ret"] == pytest.approx(expected, rel=1e-9)


def test_backend_pick_falls_back():
    import models
    assert models.pick_backend("gbdt") == "gbdt"
    # lstm requested without tensorflow available should downgrade to gbdt
    if not models.tensorflow_available():
        assert models.pick_backend("lstm") == "gbdt"
