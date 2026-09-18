"""SQLite persistence for the stock pipeline.

Placeholder lists are generated from the column names so the number of ``?``
always matches the number of columns - a class of bug that is otherwise easy
to introduce by hand.

Tables
------
prices       : real OHLCV history (one row per ticker per day)
predictions  : every out-of-sample forecast with its actual value
runs         : one row per training run with aggregate + risk metrics
signals      : actionable trading signals derived from forecasts
"""
import sqlite3

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices(
    ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
    close REAL, volume REAL, PRIMARY KEY(ticker,date));

CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY, ticker TEXT, date TEXT, model TEXT,
    predicted REAL, actual REAL, mae REAL, r2 REAL,
    created_at TEXT DEFAULT (datetime('now')));
CREATE INDEX IF NOT EXISTS idx_pred ON predictions(ticker,date);

CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY, ticker TEXT, model TEXT, backend TEXT,
    n_train INT, n_test INT,
    mae REAL, rmse REAL, r2 REAL, mape REAL,
    dir_acc REAL, strategy_ret REAL, buyhold_ret REAL, sharpe REAL,
    max_drawdown REAL,
    created_at TEXT DEFAULT (datetime('now')));

CREATE TABLE IF NOT EXISTS signals(
    id INTEGER PRIMARY KEY, ticker TEXT, date TEXT,
    action TEXT, confidence REAL, expected_move REAL, reason TEXT,
    created_at TEXT DEFAULT (datetime('now')));
CREATE INDEX IF NOT EXISTS idx_sig ON signals(ticker,date);
"""


def _q(n: int) -> str:
    """Produce exactly n '?' placeholders (','.join avoids miscounting)."""
    return ",".join(["?"] * n)


def get_conn(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_prices(conn, ticker, df):
    """Upsert real OHLCV rows. `df` is a pandas frame indexed by date."""
    rows = []
    for d, r in df.iterrows():
        rows.append((ticker, str(d)[:10],
                     float(r["Open"]), float(r["High"]), float(r["Low"]),
                     float(r["Close"]), float(r.get("Volume", 0) or 0)))
    sql = ("INSERT OR REPLACE INTO prices"
           "(ticker,date,open,high,low,close,volume) VALUES(" + _q(7) + ")")
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def save_prediction(conn, ticker, date, model, pred, actual, mae=0.0, r2=0.0):
    sql = ("INSERT INTO predictions(ticker,date,model,predicted,actual,mae,r2)"
           " VALUES(" + _q(7) + ")")
    conn.execute(sql, (ticker, date, model, float(pred), float(actual),
                       float(mae), float(r2)))
    conn.commit()


_RUN_COLS = ["ticker", "model", "backend", "n_train", "n_test", "mae", "rmse",
             "r2", "mape", "dir_acc", "strategy_ret", "buyhold_ret", "sharpe",
             "max_drawdown"]


def save_run(conn, ticker, model, backend, metrics: dict):
    vals = [ticker, model, backend,
            metrics.get("n_train"), metrics.get("n_test"),
            metrics.get("mae"), metrics.get("rmse"), metrics.get("r2"),
            metrics.get("mape"), metrics.get("dir_acc"),
            metrics.get("strategy_ret"), metrics.get("buyhold_ret"),
            metrics.get("sharpe"), metrics.get("max_drawdown")]
    sql = ("INSERT INTO runs(" + ",".join(_RUN_COLS) + ") VALUES("
           + _q(len(_RUN_COLS)) + ")")
    conn.execute(sql, vals)
    conn.commit()


def save_signal(conn, ticker, date, action, confidence, expected_move, reason=""):
    cols = ["ticker", "date", "action", "confidence", "expected_move", "reason"]
    sql = ("INSERT INTO signals(" + ",".join(cols) + ") VALUES("
           + _q(len(cols)) + ")")
    conn.execute(sql, (ticker, date, action, float(confidence),
                       float(expected_move), reason))
    conn.commit()


def latest_prices(conn, ticker, limit=200):
    return conn.execute(
        "SELECT date, close FROM prices WHERE ticker=?"
        " ORDER BY date DESC LIMIT ?", (ticker, limit)).fetchall()[::-1]
