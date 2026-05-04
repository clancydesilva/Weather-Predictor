"""
src/models/lgbm_model.py
────────────────────────
LightGBM two-stage rainfall model.

Key differences from XGBoost
─────────────────────────────
- is_unbalance=True : LightGBM's equivalent of scale_pos_weight.
  Automatically reweights classes by their inverse frequency.
  Combined with XGB_SCALE_POS_WEIGHT in the ensemble gives belt-and-braces
  imbalance handling across both frameworks.

- num_leaves instead of max_depth: LightGBM grows leaf-wise (best-first)
  rather than level-wise. num_leaves=63 roughly corresponds to max_depth=6
  but gives LightGBM more flexibility in tree shape.

- verbose=-1 : suppresses LightGBM's very copious training output.

- Trains significantly faster than XGBoost in practice (~3-5× on CPU)
  due to its histogram-based algorithm and better cache utilisation.

- colsample_by_tree (LightGBM spelling) vs colsample_bytree (XGBoost).

Public API:
    build_lgbm_pipeline(**kwargs) -> TwoStagePipeline
    LGBM_PARAM_GRID               -> dict  (for hyperparameter sweep)
"""

import lightgbm as lgb

from src.models.pipeline import TwoStagePipeline

# ── Hyperparameter sweep grid (used by train.py) ──────────────────────────────
LGBM_PARAM_GRID = {
    "num_leaves":    [31, 63, 127],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators":  [300, 600, 1000],
}


def build_lgbm_pipeline(
    n_estimators: int = 600,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> TwoStagePipeline:
    """
    Build a TwoStagePipeline backed by LightGBM for both stages.

    Parameters
    ----------
    n_estimators          : max boosting rounds (early stopping may use fewer)
    num_leaves            : max leaves per tree (controls model complexity)
    learning_rate         : shrinkage per step
    subsample             : row subsampling fraction per tree (bagging_fraction)
    colsample_bytree      : feature subsampling fraction per tree
    early_stopping_rounds : stop if val metric doesn't improve for this many rounds
    random_state          : seed for reproducibility

    Returns
    -------
    TwoStagePipeline wrapping LGBMClassifier + LGBMRegressor
    """
    classifier = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_by_tree=colsample_bytree,
        is_unbalance=True,                 # auto-reweight by inverse class frequency
        objective="binary",
        metric=["auc", "average_precision"],
        verbose=-1,
        random_state=random_state,
    )
    # Store early_stopping_rounds so pipeline._early_stop_kwargs can read it
    classifier._early_stopping_rounds = early_stopping_rounds

    regressor = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_by_tree=colsample_bytree,
        objective="regression",
        metric="mae",
        verbose=-1,
        random_state=random_state,
    )
    regressor._early_stopping_rounds = early_stopping_rounds

    return TwoStagePipeline(
        classifier=classifier,
        regressor=regressor,
        name="LightGBM_TwoStage",
    )
