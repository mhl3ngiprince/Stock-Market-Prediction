"""Server-rendered dashboard for the stock prediction pipeline.

No emoji (SVG icons only). Every figure comes from the real `runs`, `signals`,
`predictions` and `prices` tables written by the pipeline.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_shared"))

import dashboard_kit as kit   # noqa: E402
import database as db         # noqa: E402


def _fmt_pct(v):
    return "-" if v is None else f"{v * 100:+.1f}%"


def render(conn=None) -> str:
    conn = conn or db.get_conn()

    runs = conn.execute(
        "SELECT ticker, backend, mae, rmse, r2, mape, dir_acc, strategy_ret,"
        " buyhold_ret, sharpe, max_drawdown, n_test, created_at FROM runs"
        " ORDER BY created_at DESC").fetchall()
    signals = conn.execute(
        "SELECT ticker, date, action, confidence, expected_move, reason"
        " FROM signals ORDER BY id DESC LIMIT 15").fetchall()
    price_count = conn.execute("SELECT COUNT(*) n FROM prices").fetchone()["n"]
    pred_count = conn.execute("SELECT COUNT(*) n FROM predictions").fetchone()["n"]
    tickers = [r for r in conn.execute(
        "SELECT DISTINCT ticker FROM prices ORDER BY ticker")]
    ticker_names = [t["ticker"] for t in tickers]

    # strategy vs buy-and-hold for the most recent run per ticker
    latest = {}
    for r in runs:
        latest.setdefault(r["ticker"], r)
    labels = list(latest.keys())
    strat = [(latest[t]["strategy_ret"] or 0) * 100 for t in labels]
    hold = [(latest[t]["buyhold_ret"] or 0) * 100 for t in labels]

    cards = [
        kit.card("Coverage", "database",
                 kit.stat("Tickers", len(ticker_names), "", "trend")
                 + kit.stat("Stored price rows", price_count, "", "database")
                 + kit.stat("Forecasts stored", pred_count, "", "signal"),
                 note="Source: prices / predictions tables (real Yahoo Finance)"),
        kit.card("Latest metrics by ticker", "chart",
                 kit.table(
                     ["ticker", "backend", "MAE", "RMSE", "R2", "dir acc",
                      "sharpe", "max DD"],
                     [(r["ticker"], r["backend"],
                       f"{r['mae']:.2f}" if r["mae"] is not None else "-",
                       f"{r['rmse']:.2f}" if r["rmse"] is not None else "-",
                       f"{r['r2']:.3f}" if r["r2"] is not None else "-",
                       f"{r['dir_acc']:.2f}" if r["dir_acc"] is not None else "-",
                       f"{r['sharpe']:.2f}" if r["sharpe"] is not None else "-",
                       _fmt_pct(r["max_drawdown"]) )
                      for r in list(latest.values())]),
                 span=2,
                 note="Source: runs table (out-of-sample test split)"),
        kit.card("Strategy vs buy & hold (%)", "trend",
                 kit.bar_chart(labels, strat, "#22c55e")
                 + '<p class="muted">Buy &amp; hold: '
                 + ", ".join(f"{t} {h:+.1f}%" for t, h in zip(labels, hold))
                 + "</p>",
                 note="Source: runs table (cost-aware backtest)"),
        kit.card("Latest signals", "signal",
                 kit.table(["ticker", "date", "action", "conf", "exp move"],
                           [(s["ticker"], s["date"], s["action"],
                             f"{s['confidence']:.2f}"
                             if s["confidence"] is not None else "-",
                             _fmt_pct(s["expected_move"]))
                            for s in signals]),
                 span=2,
                 note="Source: signals table (derived from forecasts)"),
        kit.card("Model runs", "list",
                 kit.table(["ticker", "backend", "n test", "created"],
                           [(r["ticker"], r["backend"], r["n_test"],
                             (r["created_at"] or "")[:16])
                            for r in runs[:12]]),
                 note="Source: runs table"),
    ]
    return kit.page("Stock Market Prediction",
                    "Next-day forecasts, backtests and signals",
                    "".join(cards),
                    footer=("Tickers: " + ", ".join(ticker_names)
                            + "  |  data source: Yahoo Finance via yfinance"))
