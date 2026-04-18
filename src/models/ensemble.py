"""
src/models/ensemble.py
──────────────────────
Soft-vote ensemble combining XGBoost and LightGBM TwoStagePipelines.

Why ensemble?
─────────────
XGBoost and LightGBM make different errors due to their different tree-
building strategies (level-wise vs leaf-wise). Averaging their probabilities
smooths out individual model noise and almost always beats either alone,
at near-zero extra cost (both models are already trained).

Strategy
────────
Classifier:  Soft-vote — weighted average of P(rain) from each pipeline.
Regressor:   Weighted average of rainfall_mm — weights are inverse val-MAE
             (lower MAE = better = higher weight). Computed in train.py and
             passed in at construction time.

Weight computation (done in train.py, not here):
    w_xgb  = (1 / mae_xgb) / (1/mae_xgb + 1/mae_lgb)
    w_lgbm = (1 / mae_lgb) / (1/mae_xgb + 1/mae_lgb)

Meta-learner fallback: if soft-vote underperforms either individual model
by > 3% F1, a LogisticRegression stacking meta-learner can be swapped in
(see IMPLEMENTATION PLAN Phase 2.6). That path is not implemented here —
evaluate soft-vote first, add complexity only if needed.

Public API:
    SoftVoteEnsemble(pipelines, weights)
        .predict(X) -> dict
"""

import numpy as np
import pandas as pd

from src.models.pipeline import TwoStagePipeline


class SoftVoteEnsemble:
    """
    Soft-voting ensemble over a list of fitted TwoStagePipelines.

    Parameters
    ----------
    pipelines : list[TwoStagePipeline]
        Fitted pipelines to ensemble. All must be fitted before passing in.
    weights   : list[float] | None
        Weight for each pipeline. Must sum to 1.0. If None, equal weights
        are used. Compute from inverse val-MAE in train.py.
    threshold : float
        Decision threshold for rain_flag. Calibrate on validation set.
    """

    def __init__(
        self,
        pipelines: list[TwoStagePipeline],
        weights: list[float] | None = None,
        threshold: float = 0.5,
        name: str = "SoftVoteEnsemble",
    ):
        if not pipelines:
            raise ValueError("At least one pipeline is required.")

        self.pipelines = pipelines
        self.name      = name
        self.threshold = threshold

        if weights is None:
            n = len(pipelines)
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != len(pipelines):
                raise ValueError(
                    f"len(weights)={len(weights)} must equal len(pipelines)={len(pipelines)}"
                )
            total = sum(weights)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
            self.weights = weights

        # Verify all pipelines are fitted
        for p in self.pipelines:
            if not p._is_fitted:
                raise ValueError(f"Pipeline '{p.name}' must be fitted before ensembling.")

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Run all pipelines and return weighted-average predictions.

        Returns
        -------
        dict with keys:
            rain_probability : weighted mean P(rain) across all pipelines
            rain_flag        : threshold-applied binary (0 or 1)
            rainfall_mm      : weighted mean of individual pipeline rainfall_mm
            individual_probs : list of per-pipeline rain_probability arrays
        """
        all_probs   = []
        all_amounts = []

        for pipe, w in zip(self.pipelines, self.weights):
            result = pipe.predict(X)
            all_probs.append(result["rain_probability"] * w)
            all_amounts.append(result["rainfall_mm"] * w)

        ensemble_prob   = np.sum(all_probs,   axis=0)
        ensemble_amount = np.sum(all_amounts, axis=0)

        return {
            "rain_probability":  ensemble_prob,
            "rain_flag":         (ensemble_prob >= self.threshold).astype(int),
            "rainfall_mm":       ensemble_amount,
            "individual_probs":  [p / w for p, w in zip(all_probs, self.weights)],
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Convenience — returns ensemble rain_probability only."""
        return self.predict(X)["rain_probability"]

    def __repr__(self) -> str:
        names = [p.name for p in self.pipelines]
        return (
            f"SoftVoteEnsemble(pipelines={names}, "
            f"weights={[round(w, 3) for w in self.weights]}, "
            f"threshold={self.threshold:.4f})"
        )


def compute_inverse_mae_weights(val_maes: dict[str, float]) -> dict[str, float]:
    """
    Compute inverse-MAE weights for the regressor ensemble.

    Lower MAE = better model = higher weight.

    Parameters
    ----------
    val_maes : dict mapping model name -> val_mae_mm
               e.g. {"XGBoost_TwoStage": 0.91, "LightGBM_TwoStage": 0.88}

    Returns
    -------
    dict mapping model name -> weight (values sum to 1.0)

    Example
    -------
    >>> compute_inverse_mae_weights({"xgb": 0.91, "lgbm": 0.88})
    {"xgb": 0.491, "lgbm": 0.509}
    """
    inv_maes = {name: 1.0 / mae for name, mae in val_maes.items()}
    total    = sum(inv_maes.values())
    return {name: inv / total for name, inv in inv_maes.items()}
