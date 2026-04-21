"""
src/evaluate.py
───────────────
All metric functions used to evaluate models. Define these before writing
any model code — models are measured by these functions, not the other way.

Key design decisions:
  - find_optimal_threshold() runs on VALIDATION set only, never test.
  - evaluate_regressor() always inverse-transforms log1p → mm before reporting.
  - check_f1_gate() is the single place the PyTorch gate condition is enforced.

Public API:
    find_optimal_threshold(y_true, y_prob)  -> float
    evaluate_classifier(y_true, y_prob, threshold, split_name) -> dict
    evaluate_regressor(y_true_log, y_pred_log, split_name) -> dict
    evaluate_continuous(y_true, y_pred, variable, split_name) -> dict
    check_f1_gate(metrics, split) -> None
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
    r2_score,
)

from src.config import MIN_F1_RAIN, TARGET_F1_RAIN, TARGET_MAE_RAIN_MM


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the probability threshold that maximises F1 score.

    IMPORTANT: Call this on the VALIDATION set only. Apply the returned
    threshold to the test set without re-running this function on test data.
    Re-optimising on test would inflate reported metrics.

    Parameters
    ----------
    y_true : array of int {0, 1}
    y_prob : array of float [0, 1]  — predicted probabilities for class 1

    Returns
    -------
    float : threshold in [0, 1] that maximises val F1
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one fewer threshold than precision/recall values
    f1_scores = (
        2 * (precisions[:-1] * recalls[:-1])
        / (precisions[:-1] + recalls[:-1] + 1e-9)
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def evaluate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "val",
) -> dict:
    """
    Compute classifier metrics for rain occurrence prediction.

    Parameters
    ----------
    y_true     : array of int {0, 1}
    y_prob     : array of float [0, 1] — predicted probabilities
    threshold  : from find_optimal_threshold() on val set
    split_name : prefix for returned metric keys (e.g. "val", "test")

    Returns
    -------
    dict with keys: {split}_roc_auc, {split}_f1, {split}_precision,
                    {split}_recall, {split}_avg_precision
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        f"{split_name}_roc_auc":       float(roc_auc_score(y_true, y_prob)),
        f"{split_name}_f1":            float(f1_score(y_true, y_pred, zero_division=0)),
        f"{split_name}_precision":     float(
            f1_score(y_true, y_pred, zero_division=0, average=None)[1]
            if y_pred.sum() > 0 else 0.0
        ),
        f"{split_name}_recall":        float(
            (y_true[y_pred == 1].sum() / y_true.sum()) if y_true.sum() > 0 else 0.0
        ),
        f"{split_name}_avg_precision": float(average_precision_score(y_true, y_prob)),
        f"{split_name}_threshold":     round(threshold, 4),
    }


def evaluate_regressor(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    split_name: str = "val",
) -> dict:
    """
    Compute regression metrics on log1p-transformed rainfall.

    Metrics are reported in both log space (for loss comparison) and
    inverse-transformed mm (for human interpretability).

    IMPORTANT: Only call this on rows where rain actually occurred
    (y_true > 0). Including dry hours inflates R² and hides poor performance.

    Parameters
    ----------
    y_true_log : log1p-transformed true rainfall
    y_pred_log : log1p-transformed predicted rainfall
    split_name : prefix for returned metric keys

    Returns
    -------
    dict with keys: {split}_mae_log, {split}_rmse_log, {split}_mae_mm,
                    {split}_rmse_mm, {split}_r2_mm
    """
    # Clip to 0 before inverse-transform — predictions can go slightly negative
    y_true_mm = np.expm1(np.clip(y_true_log, 0, None))
    y_pred_mm = np.expm1(np.clip(y_pred_log, 0, None))

    return {
        f"{split_name}_mae_log":  float(mean_absolute_error(y_true_log, y_pred_log)),
        f"{split_name}_rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        f"{split_name}_mae_mm":   float(mean_absolute_error(y_true_mm, y_pred_mm)),
        f"{split_name}_rmse_mm":  float(np.sqrt(mean_squared_error(y_true_mm, y_pred_mm))),
        f"{split_name}_r2_mm":    float(r2_score(y_true_mm, y_pred_mm)),
    }


def evaluate_continuous(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable: str = "temp_c",
    split_name: str = "val",
) -> dict:
    """
    Standard regression metrics for continuous variables (temp, wind, humidity).
    No log transform — these are already on a sensible scale.

    Returns
    -------
    dict with keys: {split}_{variable}_mae, {split}_{variable}_rmse,
                    {split}_{variable}_r2
    """
    return {
        f"{split_name}_{variable}_mae":  float(mean_absolute_error(y_true, y_pred)),
        f"{split_name}_{variable}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{split_name}_{variable}_r2":   float(r2_score(y_true, y_pred)),
    }


def check_f1_gate(metrics: dict, split: str = "test") -> None:
    """
    Check whether the rain occurrence F1 score meets the product gates
    defined in config.py. Prints a clear summary to stdout.

    Gates (from config):
        MIN_F1_RAIN    = 0.55  — below this, stop and investigate
        TARGET_F1_RAIN = 0.70  — at or above this, PyTorch phase is optional

    Parameters
    ----------
    metrics : dict — typically the full metrics dict from train.py
    split   : which split to check (default "test")
    """
    f1 = metrics.get(f"{split}_f1", 0.0)
    mae = metrics.get(f"{split}_mae_mm", float("inf"))

    print(f"\n{'-' * 50}")
    print(f"  F1 GATE CHECK  ({split} set)")
    print(f"{'-' * 50}")
    print(f"  Rain occurrence F1 : {f1:.4f}  (target >= {TARGET_F1_RAIN})")
    print(f"  Rainfall MAE (mm)  : {mae:.4f}  (target < {TARGET_MAE_RAIN_MM})")

    if f1 < MIN_F1_RAIN:
        print(f"\n  BELOW FLOOR - F1 < {MIN_F1_RAIN}. Stop and investigate class imbalance.")
    elif f1 >= TARGET_F1_RAIN and mae < TARGET_MAE_RAIN_MM:
        print(f"\n  GATE PASSED - Gradient boosting v1 ships. PyTorch is optional.")
    elif f1 >= TARGET_F1_RAIN:
        print(f"\n  F1 target met, but MAE >= {TARGET_MAE_RAIN_MM}mm. Tune regressor further.")
    else:
        print(f"\n  Above floor, below target. Tune or proceed to ensemble.")
    print(f"{'-' * 50}\n")


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    """
    Proportion of the top-K predictions by probability that are correct.
    Used for onset/offset evaluation (Phase 3) where we care most about
    the highest-confidence predictions.

    Parameters
    ----------
    y_true : array of int {0, 1}
    y_prob : array of float [0, 1]
    k      : number of top predictions to evaluate

    Returns
    -------
    float : fraction of top-K predictions that are true positives
    """
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_prob)[-k:]
    return float(y_true[top_k_idx].mean())
