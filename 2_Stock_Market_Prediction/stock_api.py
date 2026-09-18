"""FastAPI service for the stock prediction pipeline.

Run:  python pipeline.py --api      (docs at http://127.0.0.1:8002/docs)
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import database as db
import pipeline

app = FastAPI(title="Stock Prediction API", version="2.0.0")


class ForecastRequest(BaseModel):
    ticker: str
    backend: Optional[str] = None
    days: Optional[int] = None
    epochs: Optional[int] = None


@app.get("/health")
def health():
    conn = db.get_conn()
    n = conn.execute("SELECT COUNT(*) c FROM prices").fetchone()["c"]
    return {"status": "ok", "prices_stored": n,
            "tensorflow": pipeline.models.tensorflow_available()}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    """Fetch real data, train, evaluate and produce a signal for one ticker."""
    try:
        metrics = pipeline.predict_ticker(req.ticker.upper(), backend=req.backend,
                                          days=req.days, epochs=req.epochs)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    conn = db.get_conn()
    sig = conn.execute(
        "SELECT action, confidence, expected_move, reason FROM signals"
        " WHERE ticker=? ORDER BY id DESC LIMIT 1", (req.ticker.upper(),)).fetchone()
    return {"ticker": req.ticker.upper(), "metrics": metrics,
            "signal": dict(sig) if sig else None}


@app.get("/metrics")
def metrics():
    conn = db.get_conn()
    df = pd.read_sql(
        "SELECT ticker, backend, mae, r2, dir_acc, strategy_ret, buyhold_ret,"
        " sharpe, max_drawdown, created_at FROM runs ORDER BY created_at DESC",
        conn)
    return {"runs": df.to_dict(orient="records")}


@app.get("/signals")
def signals():
    conn = db.get_conn()
    df = pd.read_sql(
        "SELECT ticker, date, action, confidence, expected_move, reason"
        " FROM signals ORDER BY id DESC LIMIT 50", conn)
    return {"signals": df.to_dict(orient="records")}


@app.get("/prices/{ticker}")
def prices(ticker: str, limit: int = 120):
    conn = db.get_conn()
    rows = db.latest_prices(conn, ticker.upper(), limit)
    return {"ticker": ticker.upper(),
            "prices": [{"date": r["date"], "close": r["close"]} for r in rows]}


@app.get("/predictions/{ticker}")
def predictions(ticker: str, limit: int = 100):
    conn = db.get_conn()
    df = pd.read_sql(
        "SELECT date, model, predicted, actual FROM predictions"
        " WHERE ticker=? ORDER BY date DESC LIMIT ?", conn,
        params=(ticker.upper(), limit))
    return {"ticker": ticker.upper(),
            "predictions": df.to_dict(orient="records")}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Server-rendered dashboard (SVG icons, real data only)."""
    import dashboard as dash
    return dash.render(db.get_conn())
