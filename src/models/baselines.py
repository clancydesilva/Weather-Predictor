"""
src/models/baselines.py
───────────────────────
Naive baselines that every trained model must beat.

PersistenceBaseline  — predict that the next hour = the current hour.
                       This is the absolute floor. Any model that cannot
                       beat it is useless.

ClimatologyBaseline  — for each (hour, month) bin, predict the historical
                       mean from the training set. This tests whether the
                       model learns anything beyond "it rains more in winter
                       at 3am than in summer at noon".

Both expose the same interface as TwoStagePipeline so they can be evaluated
with the same evaluate.py functions.
"""

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """
    Predicts rain_occurred[t] = rain_occurred[t-1]  (lag-1 persistence).
    Predicts rainfall_log1p[t] = log1p(rainfall_mm[t-1]).

    No fitting required — uses the lag feature columns already in X.
    """
    name = "PersistenceBaseline"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return the lag-1 rain occurrence as a probability.
        Values are binary (0.0 or 1.0) since persistence gives no gradation.
        """
        return X["rain_occurred_lag_1h"].values.astype(float)

    def predict_rain_amount(self, X: pd.DataFrame) -> np.ndarray:
        """Return log1p of lag-1 rainfall as the amount prediction."""
        return np.log1p(np.clip(X["rainfall_lag_1h"].values, 0, None))

    def predict(self, X: pd.DataFrame) -> dict:
        """Match TwoStagePipeline.predict() interface."""
        rain_prob = self.predict_proba(X)
        reg_out   = self.predict_rain_amount(X)
        return {
            "rain_probability": rain_prob,
            "rain_flag":        (rain_prob >= 0.5).astype(int),
            "rainfall_log1p":   reg_out,
            "rainfall_mm":      np.expm1(reg_out),
        }


class ClimatologyBaseline:
    """
    For each (hour-of-day, month) bin, predict:
      - rain_probability : mean historical rain_occurred rate in that bin
      - rainfall amount  : mean historical log1p rainfall in that bin (wet hours only)

    Fit on training data only. At prediction time, look up the (hour, month)
    from the index of X.
    """
    name = "ClimatologyBaseline"

    # Overall fallback values used when a (hour, month) bin is unseen in training
    _FALLBACK_PROB   = 0.172   # ≈ overall wet fraction from audit
    _FALLBACK_AMOUNT = 0.0

    def __init__(self):
        self._clim_prob:   pd.Series | None = None
        self._clim_amount: pd.Series | None = None
        self._is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "ClimatologyBaseline":
        """
        Compute per-bin historical mean from training data.

        Parameters
        ----------
        df_train : pd.DataFrame  — must be indexed by datetime, must contain
                   rain_occurred and rainfall_log1p columns.
        """
        keys = pd.MultiIndex.from_arrays(
            [df_train.index.hour, df_train.index.month],
            names=["hour", "month"],
        )

        self._clim_prob = df_train["rain_occurred"].groupby(keys).mean()

        wet = df_train[df_train["rain_occurred"] == 1]
        wet_keys = pd.MultiIndex.from_arrays(
            [wet.index.hour, wet.index.month],
            names=["hour", "month"],
        )
        self._clim_amount = wet["rainfall_log1p"].groupby(wet_keys).mean()

        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Call fit() before predict_proba()"
        keys = list(zip(X.index.hour, X.index.month))
        return np.array([
            self._clim_prob.get(k, self._FALLBACK_PROB) for k in keys
        ])

    def predict_rain_amount(self, X: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Call fit() before predict_rain_amount()"
        keys = list(zip(X.index.hour, X.index.month))
        return np.array([
            self._clim_amount.get(k, self._FALLBACK_AMOUNT) for k in keys
        ])

    def predict(self, X: pd.DataFrame) -> dict:
        """Match TwoStagePipeline.predict() interface."""
        rain_prob = self.predict_proba(X)
        reg_out   = self.predict_rain_amount(X)
        return {
            "rain_probability": rain_prob,
            "rain_flag":        (rain_prob >= 0.5).astype(int),
            "rainfall_log1p":   reg_out,
            "rainfall_mm":      np.expm1(reg_out),
        }
