"""
src/models/xgb_model.py
───────────────────────
XGBoost two-stage rainfall model.

Key decisions
─────────────
- Classifier uses scale_pos_weight = XGB_SCALE_POS_WEIGHT (4.82)
  Formula: n_dry / n_wet = 460,388 / 95,549 = 4.82
  This tells XGBoost that each wet-hour error should cost 4.82× more
  than a dry-hour error, counteracting the class imbalance.

- eval_metric = ['auc', 'aucpr']:
  aucpr (area under precision-recall) is more informative than roc_auc
  for imbalanced classes. roc_auc can look good even when the model
  mostly ignores the minority class.

- tree_method = 'hist': fastest CPU tree builder. Automatically uses
  GPU if one is available (no code change needed).

- early_stopping_rounds = 50: stops adding trees when validation metric
  hasn't improved for 50 rounds. Prevents overfitting and saves time.

- Regressor uses objective='reg:squarederror' on log1p targets.
  The log1p transform already handles the right-skewed distribution,
  so a standard squared-error loss is appropriate.

Public API:
    build_xgb_pipeline(**kwargs) -> TwoStagePipeline
    XGB_PARAM_GRID               -> dict  (for hyperparameter sweep)
"""

import xgboost as xgb

from src.config import XGB_SCALE_POS_WEIGHT
from src.models.pipeline import TwoStagePipeline

# ── Hyperparameter sweep grid (used by train.py) ──────────────────────────────
# Cartesian product of these values = 27 combinations.
# train.py iterates with early stopping so n_estimators is a ceiling, not fixed.
XGB_PARAM_GRID = {
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators":  [300, 600, 1000],
}


def build_xgb_pipeline(
    n_estimators: int = 600,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> TwoStagePipeline:
    """
    Build a TwoStagePipeline backed by XGBoost for both stages.

    Parameters
    ----------
    n_estimators          : max trees (early stopping may use fewer)
    max_depth             : max tree depth — deeper = more complex, more overfit risk
    learning_rate         : shrinkage per step — lower = slower but more robust
    subsample             : fraction of rows sampled per tree (row subsampling)
    colsample_bytree      : fraction of features sampled per tree
    early_stopping_rounds : stop if val metric doesn't improve for this many rounds
    random_state          : seed for reproducibility

    Returns
    -------
    TwoStagePipeline wrapping XGBClassifier + XGBRegressor
    """
    classifier = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,   # 4.82 = n_dry / n_wet from audit
        objective="binary:logistic",
        eval_metric=["auc", "aucpr"],
        tree_method="hist",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        verbosity=0,
    )

    regressor = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        verbosity=0,
    )

    return TwoStagePipeline(
        classifier=classifier,
        regressor=regressor,
        name="XGBoost_TwoStage",
    )
