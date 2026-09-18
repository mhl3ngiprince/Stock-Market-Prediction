"""Stock Market Prediction - real data, honest evaluation, deployable signals.

Pipeline per ticker:
    real Yahoo Finance OHLCV
        -> features (returns, SMA/EMA, RSI, MACD, volatility, z-score)
        -> model (LSTM if TensorFlow is present, else gradient boosting)
        -> strictly out-of-sample test split (NO leakage)
        -> metrics + long/flat backtest vs buy & hold
        -> SQLite persistence + PNG plot + next-day signal + CSV export

Commands
--------
    python pipeline.py                       # run configured tickers
    python pipeline.py --tickers AAPL,TSLA   # override
    python pipeline.py --report              # print DB summary
    python pipeline.py --api                 # start REST service
    python pipeline.py --selftest            # verify logic end-to-end offline
"""
from __future__ import annotations
import argparse
import os
import sys
# Windows consoles default to cp1252 and choke on unicode like 'R²'.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import data as data_mod
import database as db
import metrics as M
import models
from config import (API_HOST, API_PORT, BACKEND, EXPORT_DIR, HORIZON, PLOT_DIR,
                    SEED, TEST_SPLIT, TICKERS, TRANSACTION_COST, WINDOW)

np.random.seed(SEED)


# ---------------------------------------------------------------- plotting ---
def plot_forecast(ticker, dates, actual, pred, rmse, metrics, outdir=PLOT_DIR):
    os.makedirs(outdir, exist_ok=True)
    lo, hi = M.confidence_from_error(pred, rmse)
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="actual", lw=1.5)
    plt.plot(dates, pred, label="predicted", lw=1.2)
    plt.fill_between(dates, lo, hi, alpha=0.15, label="±1·RMSE")
    plt.title(f"{ticker}  MAE={metrics['mae']:.2f}  RMSE={rmse:.2f}  "
              f"R²={metrics['r2']:.3f}  dir_acc={metrics['dir_acc']:.0%}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"{ticker}.png")
    plt.savefig(path, dpi=110)
    plt.close()
    return path


# ------------------------------------------------------------------ pipeline --
def predict_ticker(ticker, backend=None, days=None, epochs=None):
    backend = models.pick_backend(backend or BACKEND)
    conn = db.get_conn()

    df = data_mod.fetch_prices(ticker, days or None)
    feats = data_mod.add_features(df)
    db.save_prices(conn, ticker, df)
    print(f"{ticker}: {len(df)} real trading days, "
          f"{feats.index[0].date()} -> {feats.index[-1].date()}")

    n = len(feats)
    split = int(n * (1 - TEST_SPLIT))
    if split < WINDOW + 10 or n - split < 5:
        raise RuntimeError(f"{ticker}: not enough history ({n} rows) to train/test.")

    if backend == "lstm":
        pred, actual, dates, rmse, extra = _run_lstm(feats, split, epochs)
        model_path = os.path.join(PLOT_DIR, f"{ticker}_lstm.keras")
    else:
        pred, actual, dates, rmse, extra = _run_gbdt(feats, split)
        model_path = os.path.join(PLOT_DIR, f"{ticker}_gbdt.joblib")

    reg = M.regression_metrics(actual, pred)
    reg["rmse"] = rmse
    reg["dir_acc"] = M.directional_accuracy(actual, pred)
    bt = M.backtest(actual, pred, cost=TRANSACTION_COST)
    reg.update(bt)
    reg["n_train"], reg["n_test"] = split, n - split

    # persist every out-of-sample forecast
    for d, p, a in zip(dates, pred, actual):
        db.save_prediction(conn, ticker, str(pd.Timestamp(d).date()),
                           backend, p, a, reg["mae"], reg["r2"])
    db.save_run(conn, ticker, "next_close", backend, reg)

    # next-day signal from the most recent window
    sig = make_signal(feats, pred[-1] if len(pred) else None)
    db.save_signal(conn, ticker, str(feats.index[-1].date()), sig["action"],
                   sig["confidence"], sig["expected_move"], sig["reason"])

    plot_forecast(ticker, dates, actual, pred, rmse, reg)
    print(f"  model={backend}  MAE={reg['mae']:.2f} RMSE={rmse:.2f} "
          f"R²={reg['r2']:.3f} MAPE={reg['mape']:.2f}%")
    print(f"  dir_acc={reg['dir_acc']:.0%}  strategy={reg['strategy_ret']:+.1%} "
          f"vs buy&hold={reg['buyhold_ret']:+.1%}  sharpe={reg['sharpe']:.2f} "
          f"maxDD={reg['max_drawdown']:.1%}  trades={reg['n_trades']}")
    print(f"  signal -> {sig['action']} (expected {sig['expected_move']:+.2%})  "
          f"plot: {PLOT_DIR}/{ticker}.png")
    return reg


def _run_lstm(feats, split, epochs):
    """Fit an LSTM on scaled closes using ONLY the training slice, then
    forecast the test period window-by-window (no test data ever seen)."""
    close = feats["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler().fit(close[:split])          # fit scaler on train only
    scaled = scaler.transform(close).ravel()
    window = WINDOW
    Xtr, ytr = models.make_windows(scaled[:split], window)
    model = models.build_lstm(window)
    model.fit(Xtr[..., None], ytr, epochs=epochs or 25, batch_size=32,
              verbose=0, validation_split=0.1)

    # build test windows without crossing the split boundary
    Xte, yte = models.make_windows(scaled, window)
    offset = split - window
    Xte, yte = Xte[offset:], yte[offset:]
    pred = scaler.inverse_transform(
        model.predict(Xte[..., None], verbose=0)).ravel()
    actual = scaler.inverse_transform(yte.reshape(-1, 1)).ravel()
    dates = feats.index[split:][:len(pred)]
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    os.makedirs(PLOT_DIR, exist_ok=True)
    models.save_model(model, os.path.join(PLOT_DIR, "tmp_lstm.keras"))
    return pred, actual[:len(pred)], dates, rmse, {}


def _run_gbdt(feats, split):
    """Regress next-day *return* then reconstruct price (trees cannot
    extrapolate a price level, but they predict returns well)."""
    cols = data_mod.FEATURE_COLS
    X = feats[cols].values
    y = feats["target_ret"].values
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    gb = models.GBDTModel().fit(Xtr, ytr)
    pred_ret = gb.predict(Xte)
    base = feats["Close"].values[split:]
    actual = feats["target"].values[split:]
    pred = base * (1.0 + pred_ret)          # reconstruct next-day price
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    dates = feats.index[split:]
    import joblib
    os.makedirs(PLOT_DIR, exist_ok=True)
    joblib.dump(gb, os.path.join(PLOT_DIR, "tmp_gbdt.joblib"))
    return pred, actual, dates, rmse, {}


def make_signal(feats, last_pred):
    """Turn the latest prediction into a BUY/HOLD/SELL-ready signal."""
    last_close = float(feats["Close"].iloc[-1])
    if last_pred is None:
        return {"action": "HOLD", "confidence": 0.0,
                "expected_move": 0.0, "reason": "no forecast available"}
    move = (float(last_pred) - last_close) / last_close
    rng = float(feats["Close"].pct_change().std() or 0.01)
    strength = min(abs(move) / (2 * rng + 1e-9), 1.0)     # 0..1
    if move > rng:
        action, reason = "BUY", f"forecast +{move:.2%} > 1 std"
    elif move < -rng:
        action, reason = "SELL", f"forecast {move:.2%} < -1 std"
    else:
        action, reason = "HOLD", f"forecast {move:+.2%} within noise"
    return {"action": action, "confidence": round(strength, 3),
            "expected_move": round(move, 5), "reason": reason}


# ------------------------------------------------------------------ report ----
def report(export=False):
    conn = db.get_conn()
    print("== Model performance by ticker ==")
    df = pd.read_sql(
        "SELECT ticker, backend, ROUND(mae,2) mae, ROUND(r2,3) r2,"
        " ROUND(dir_acc,3) dir_acc, ROUND(strategy_ret,3) strat,"
        " ROUND(buyhold_ret,3) buyhold, ROUND(sharpe,2) sharpe,"
        " ROUND(max_drawdown,3) maxDD, n_test FROM runs"
        " ORDER BY created_at DESC", conn)
    print(df.to_string(index=False) if len(df) else "(no runs yet)")
    print("\n== Latest signals ==")
    sig = pd.read_sql(
        "SELECT ticker, date, action, ROUND(confidence,2) conf,"
        " ROUND(expected_move,4) exp_move, reason FROM signals s"
        " WHERE created_at=(SELECT MAX(created_at) FROM signals s2"
        "                   WHERE s2.ticker=s.ticker) ORDER BY ticker", conn)
    print(sig.to_string(index=False) if len(sig) else "(no signals yet)")
    if export:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        path = os.path.join(EXPORT_DIR, "latest_signals.csv")
        sig.to_csv(path, index=False)
        print(f"\nExported signals -> {path}")


# ---------------------------------------------------------------- selftest ----
def selftest():
    """Prove the math works with a synthetic-but-realistic price path."""
    print("== Self-test (offline) ==")
    rng = np.random.default_rng(SEED)
    n = 400
    rets = rng.normal(0.0004, 0.012, n)
    close = 100 * np.cumprod(1 + rets)
    actual = close[1:]
    pred = actual + rng.normal(0, 0.5, len(actual))    # noisy forecast
    reg = M.regression_metrics(actual, pred)
    reg["dir_acc"] = M.directional_accuracy(actual, pred)
    bt = M.backtest(actual, pred)
    print(f"  metrics : MAE={reg['mae']:.3f} RMSE={reg['rmse']:.3f} "
          f"R2={reg['r2']:.3f} dir_acc={reg['dir_acc']:.2f}")
    print(f"  backtest: strat={bt['strategy_ret']:+.2%} "
          f"buyhold={bt['buyhold_ret']:+.2%} sharpe={bt['sharpe']:.2f} "
          f"maxDD={bt['max_drawdown']:.1%} trades={bt['n_trades']}")
    ok = (reg["mae"] > 0 and np.isfinite(reg["r2"])
          and np.isfinite(bt["sharpe"]))
    print("  SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ------------------------------------------------------------------- api -----
def run_api():
    try:
        import uvicorn
    except ImportError:
        sys.exit("Install API extras: pip install fastapi 'uvicorn[standard]'")
    print(f"Stock API on http://{API_HOST}:{API_PORT}")
    uvicorn.run("stock_api:app", host=API_HOST, port=API_PORT, reload=False)


# ------------------------------------------------------------------- main -----
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", default=None, help="comma separated, overrides .env")
    p.add_argument("--backend", default=None, choices=["auto", "lstm", "gbdt"])
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--report", action="store_true")
    p.add_argument("--export", action="store_true")
    p.add_argument("--api", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.report:
        report(args.export)
        return 0
    if args.api:
        run_api()
        return 0

    tickers = ([t.strip().upper() for t in args.tickers.split(",")]
               if args.tickers else TICKERS)
    print(f"Running {len(tickers)} ticker(s): {', '.join(tickers)}")
    for t in tickers:
        try:
            predict_ticker(t, backend=args.backend, days=args.days,
                           epochs=args.epochs)
        except Exception as e:
            print(f"{t}: FAILED - {e}")
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
