# 2 · Stock Market Prediction - Real Data, Honest Evaluation, Actionable Signals

Forecasts the **next-day close** from real Yahoo Finance data with a strictly
out-of-sample test split, then scores the forecast the way a trader cares about:
directional accuracy and a cost-aware backtest against buy & hold.

**No simulated prices.** If Yahoo Finance is unreachable the pipeline raises a
clear error rather than inventing data.

## What's real here

| Concern | What this does |
|--------|-----------------|
| Data | Real OHLCV via `yfinance`, stored in SQLite (`prices`) |
| Features | Returns (1/5d), SMA 10/20/50, EMA, RSI-14, MACD, volatility, z-score - all causal (no look-ahead) |
| Target | Next-day **return** (model form that actually extrapolates), price reconstructed |
| Models | **LSTM** (TensorFlow) *or* **gradient boosting** (scikit-learn), auto-selected |
| Validation | Chronological train/test split, no shuffling, no leakage |
| Metrics | MAE, RMSE, MAPE, R², **directional accuracy** |
| Backtest | Long/flat strategy with **transaction costs**, vs buy & hold, Sharpe, max drawdown, #trades |
| Signals | BUY / HOLD / SELL with confidence and expected move |
| Outputs | PNG plot with ±RMSE band, SQLite tables, CSV export |
| Integration | REST API (forecast, metrics, signals, prices, predictions) |
| Tests | `pytest` suite for metrics, backtest, features, DB |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env                 # edit tickers if you like

python pipeline.py --selftest        # verify the math with no network
python pipeline.py                   # real data -> models -> metrics -> DB
python pipeline.py --report --export # metrics + signals table + CSV
python pipeline.py --api             # http://127.0.0.1:8002  (docs at /docs)
```

### Examples
```bash
python pipeline.py --tickers AAPL,TSLA --days 1200 --backend gbdt
python pipeline.py --tickers NVDA --backend lstm --epochs 30
```

## REST API
| Method | Path | Purpose |
|-------|------|---------|
| GET  | `/health` | service status |
| POST | `/forecast` | fetch + train + evaluate + signal for a ticker |
| GET  | `/metrics` | all stored run metrics |
| GET  | `/signals` | latest signals |
| GET  | `/prices/{ticker}` | stored OHLCV closes |
| GET  | `/predictions/{ticker}` | out-of-sample forecasts |

## Interpreting the numbers (read this)
* **R²** near 1 on a *price level* mostly means "prices are autocorrelated",
  not that the model is clever. The metric that matters is **directional
  accuracy** and whether the **strategy beats buy & hold after costs**.
* Directional accuracy hovering around 50% is normal and honest for daily
  equities - do not trust any demo that claims 99%.
* A strategy can beat buy & hold by *avoiding* drawdowns even with modest
  accuracy; check `maxDD` and `sharpe` together.

## Tables
`prices` · `predictions` · `runs` (metrics) · `signals`

## Deeper modelling ideas
* Add macro features (rates, VIX) and cross-asset signals.
* Walk-forward retraining instead of a single split.
* Probabilistic forecasts (quantile regression) for risk sizing.
* Position sizing from volatility instead of a binary long/flat.

> Not financial advice. Markets are noisy; treat this as a research tool.

## Web dashboard

Every project ships a server-rendered **web dashboard** (a real HTML page).

* **No emoji** ? all icons are inline **SVG** (defined in `../_shared/dashboard_kit.py`).
* **Real data only** ? every card is labelled with its data source, and the
  page reads the same SQLite tables the pipeline writes.
* **No CDN / no JavaScript required** ? charts are plain inline SVG.

Open it by running the project's API and visiting `/dashboard`:

    python <api-entrypoint>            # start the service
    # then open http://127.0.0.1:<port>/dashboard
