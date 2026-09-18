"""Forecasting models.

Two interchangeable backends:

* ``lstm`` - a Keras LSTM over a sliding window of closing prices.
* ``gbdt`` - scikit-learn gradient boosting over the engineered features.

The pipeline picks whichever is available (`BACKEND=auto`), so the project runs
everywhere: with TensorFlow installed you get the LSTM, without it you still
get a real, trained model instead of a crash.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from config import EPOCHS, SEED, WINDOW


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


# ------------------------------------------------------------------- LSTM -----
def build_lstm(window=WINDOW):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.random.set_seed(SEED)
    model = models.Sequential([
        layers.Input((window, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def make_windows(series: np.ndarray, window: int = WINDOW):
    """(X, y) sliding windows over a scaled 1-D price series."""
    X = np.stack([series[i - window:i] for i in range(window, len(series))])
    y = series[window:]
    return X, y


def train_lstm(scaled_close: np.ndarray, epochs: int = EPOCHS, window: int = WINDOW):
    """Train an LSTM on a scaled close-price series; return (model, X, y)."""
    X, y = make_windows(scaled_close, window)
    model = build_lstm(window)
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)
    return model, X, y


# ------------------------------------------------------------------- GBDT -----
class GBDTModel:
    """Gradient-boosted trees over engineered features (no deep-learning dep)."""

    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, random_state=SEED)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return getattr(self.model, "feature_importances_", None)


def pick_backend(preference: str = "auto") -> str:
    pref = (preference or "auto").lower()
    if pref in ("lstm", "gbdt"):
        if pref == "lstm" and not tensorflow_available():
            print("  (lstm requested but TensorFlow missing -> using gbdt)")
            return "gbdt"
        return pref
    return "lstm" if tensorflow_available() else "gbdt"


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if hasattr(model, "save"):                 # keras
        model.save(path)
    else:
        import joblib
        joblib.dump(model, path)


def load_model(path: str):
    if path.endswith(".keras") or path.endswith(".h5"):
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    import joblib
    return joblib.load(path)
