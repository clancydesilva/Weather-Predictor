"""
src/models/pipeline.py
──────────────────────
TwoStagePipeline — the mandatory architecture for all rainfall models.

Why two stages are required
───────────────────────────
Cork rainfall is 82.8% dry hours. A single regression model trained on all
hours learns that predicting near-zero always produces a low loss. It
converges to "almost never rains" and never improves.

The fix: split the problem into two separate questions.
  Stage A (classifier): Will it rain at all?       → P(rain_occurred)
  Stage B (regressor) : If so, how much?           → log1p(rainfall_mm)
                        trained ONLY on wet hours

Final prediction = P(rain) × expm1(Stage B output)

This forces the regressor to learn the shape of rainfall amounts
without being distracted by the 82.8% zeros.

Public API:
    TwoStagePipeline(classifier, regressor, name)
        .fit(X_train, y_train, X_val, y_val)  -> self
        .predict(X)                            -> dict
        .predict_proba(X)                      -> np.ndarray
"""

import numpy as np
import pandas as pd

from src.evaluate import find_optimal_threshold
from src.config import FEATURE_COLUMNS, TARGET_BINARY, TARGET_REGRESSION


class TwoStagePipeline:
    """
    Two-stage weather forecast pipeline wrapping any sklearn-compatible
    classifier and regressor.

    Parameters
    ----------
    classifier : sklearn-compatible classifier with predict_proba()
    regressor  : sklearn-compatible regressor with predict()
    name       : human-readable identifier used in metrics.json and logs
    """

    def __init__(self, classifier, regressor, name: str = "TwoStagePipeline"):
        self.classifier  = classifier
        self.regressor   = regressor
        self.name        = name
        self.threshold   = 0.5       # updated by fit() when val data is provided
        self._is_fitted  = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> "TwoStagePipeline":
        """
        Fit both stages.

        Parameters
        ----------
        X_train : feature DataFrame — columns must include FEATURE_COLUMNS
        y_train : target DataFrame — must contain TARGET_BINARY and TARGET_REGRESSION
        X_val   : optional — if provided, threshold is calibrated on this set
        y_val   : optional — required if X_val is provided

        Stage A trains on ALL training hours.
        Stage B trains ONLY on wet hours (rain_occurred == 1).
        Threshold is calibrated on the validation set only.
        """
        _check_columns(X_train, y_train)

        # Stage A: binary classifier on all hours
        clf_fit_kwargs = _early_stop_kwargs(self.classifier, X_val, y_val, TARGET_BINARY)
        self.classifier.fit(X_train[FEATURE_COLUMNS], y_train[TARGET_BINARY], **clf_fit_kwargs)

        # Stage B: regressor on wet hours only
        wet_mask = y_train[TARGET_BINARY] == 1
        X_wet    = X_train.loc[wet_mask, FEATURE_COLUMNS]
        y_wet    = y_train.loc[wet_mask, TARGET_REGRESSION]

        reg_fit_kwargs = _early_stop_kwargs(self.regressor, X_val, y_val, TARGET_REGRESSION, wet_only=True)
        self.regressor.fit(X_wet, y_wet, **reg_fit_kwargs)

        # Calibrate decision threshold on validation set (never on test)
        if X_val is not None and y_val is not None:
            val_prob = self.classifier.predict_proba(X_val[FEATURE_COLUMNS])[:, 1]
            self.threshold = find_optimal_threshold(
                y_val[TARGET_BINARY].values, val_prob
            )
            print(f"  [{self.name}] Optimal threshold = {self.threshold:.4f}")

        self._is_fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Run both stages and return a combined prediction dict.

        Returns
        -------
        dict with keys:
            rain_probability : float array [0, 1]   — Stage A output
            rain_flag        : int array {0, 1}      — threshold-applied binary
            rainfall_log1p   : float array           — raw Stage B output
            rainfall_mm      : float array >= 0      — combined two-stage output
        """
        assert self._is_fitted, f"{self.name}: call fit() before predict()"

        X_feat = X[FEATURE_COLUMNS]

        # Stage A
        rain_prob = self.classifier.predict_proba(X_feat)[:, 1]

        # Stage B (run on all rows — multiply by probability to suppress dry predictions)
        reg_out = self.regressor.predict(X_feat)
        reg_out = np.clip(reg_out, 0, None)   # predictions cannot be negative log1p

        # Final: P(rain) × amount — probability acts as a continuous gate
        rainfall_mm = rain_prob * np.expm1(reg_out)

        return {
            "rain_probability": rain_prob,
            "rain_flag":        (rain_prob >= self.threshold).astype(int),
            "rainfall_log1p":   reg_out,
            "rainfall_mm":      rainfall_mm,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Convenience passthrough — returns Stage A probability only."""
        assert self._is_fitted, f"{self.name}: call fit() before predict_proba()"
        return self.classifier.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"TwoStagePipeline(name={self.name!r}, status={status}, "
            f"threshold={self.threshold:.4f})"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_columns(X: pd.DataFrame, y: pd.DataFrame) -> None:
    """Raise informative errors if required columns are absent."""
    missing_feat = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing_feat:
        raise ValueError(f"X is missing feature columns: {missing_feat[:5]}...")

    for col in (TARGET_BINARY, TARGET_REGRESSION):
        if col not in y.columns:
            raise ValueError(
                f"y is missing target column '{col}'. "
                f"Present columns: {list(y.columns)}"
            )


def _early_stop_kwargs(
    estimator,
    X_val: pd.DataFrame | None,
    y_val: pd.DataFrame | None,
    target_col: str,
    wet_only: bool = False,
) -> dict:
    """
    Build fit() kwargs for early stopping if the estimator supports it.
    XGBoost and LightGBM both support eval_set for early stopping.
    Sklearn estimators without this param get an empty dict.
    """
    if X_val is None or y_val is None:
        return {}

    # Check if the estimator's fit() accepts eval_set
    import inspect
    sig = inspect.signature(estimator.fit)
    if "eval_set" not in sig.parameters:
        return {}

    if wet_only:
        val_wet_mask = y_val[TARGET_BINARY] == 1
        X_v = X_val.loc[val_wet_mask, FEATURE_COLUMNS]
        y_v = y_val.loc[val_wet_mask, target_col]
    else:
        X_v = X_val[FEATURE_COLUMNS]
        y_v = y_val[target_col]

    return {"eval_set": [(X_v, y_v)]}
