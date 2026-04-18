"""
src/train.py
────────────
Full training orchestrator. Runs the complete Phase 2 pipeline in order:

  1.  Load hourly_features.parquet
  2.  Time-based split (assert no leakage)
  3.  Evaluate Persistence baseline
  4.  Evaluate Climatology baseline
  5.  Train XGBoost (default params or --sweep for grid search)
  6.  Train LightGBM (default params or --sweep for grid search)
  7.  Build SoftVoteEnsemble with inverse-MAE weights
  8.  Evaluate ensemble on TEST SET
  9.  Check F1 gate
  10. Save models to models/  (keep last 3 versions)
  11. Write all metrics to results/metrics.json

Usage
-----
    # Fast run with default hyperparameters (~5–10 min):
    python src/train.py

    # Full hyperparameter sweep (27 combos per model, ~1–2 hrs):
    python src/train.py --sweep

Output
------
    models/XGBoost_TwoStage_latest.joblib
    models/LightGBM_TwoStage_latest.joblib
    models/ensemble_latest.joblib
    results/metrics.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.config import (
    FEATURE_COLUMNS,
    METRICS_JSON,
    MODELS_DIR,
    RESULTS_DIR,
    TARGET_BINARY,
    TARGET_REGRESSION,
    TRAIN_END_DATE,
    VAL_END_DATE,
    TEST_START_DATE,
)
from src.evaluate import (
    check_f1_gate,
    evaluate_classifier,
    evaluate_regressor,
    find_optimal_threshold,
)
from src.models.baselines import ClimatologyBaseline, PersistenceBaseline
from src.models.ensemble import SoftVoteEnsemble, compute_inverse_mae_weights
from src.models.lgbm_model import LGBM_PARAM_GRID, build_lgbm_pipeline
from src.models.xgb_model import XGB_PARAM_GRID, build_xgb_pipeline


# ── Split ─────────────────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the feature DataFrame into train / val / test using config dates.
    Asserts strict temporal ordering — no row overlaps.
    """
    train = df.loc[:TRAIN_END_DATE]
    val   = df.loc[TRAIN_END_DATE:VAL_END_DATE].iloc[1:]   # exclude boundary row
    test  = df.loc[TEST_START_DATE:]

    assert train.index.max() < val.index.min(),  "Train/val boundary overlap!"
    assert val.index.max()   < test.index.min(), "Val/test boundary overlap!"
    assert len(train) > 0 and len(val) > 0 and len(test) > 0, "Empty split!"

    print(f"  Train: {len(train):>7,} rows  ({train.index.min().date()} to {train.index.max().date()})")
    print(f"  Val:   {len(val):>7,} rows  ({val.index.min().date()} to {val.index.max().date()})")
    print(f"  Test:  {len(test):>7,} rows  ({test.index.min().date()} to {test.index.max().date()})")
    return train, val, test


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the two target columns as a separate DataFrame."""
    return df[[TARGET_BINARY, TARGET_REGRESSION]]


# ── Model saving ──────────────────────────────────────────────────────────────

def save_model(obj, name: str) -> Path:
    """
    Save a model to models/<name>_<timestamp>.joblib and models/<name>_latest.joblib.
    Returns the timestamped path.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODELS_DIR / f"{name}_{ts}.joblib"
    joblib.dump(obj, path)
    joblib.dump(obj, MODELS_DIR / f"{name}_latest.joblib")
    print(f"  Saved: {path.name}")
    prune_old_versions(name, keep=3)
    return path


def prune_old_versions(model_name: str, keep: int = 3) -> None:
    """
    Keep only the `keep` most recent timestamped model files.
    Uses pathlib glob + mtime sort — no additional library needed.
    """
    versioned = sorted(
        MODELS_DIR.glob(f"{model_name}_2*.joblib"),  # timestamp starts with year
        key=lambda p: p.stat().st_mtime,
    )
    for old in versioned[:-keep]:
        old.unlink()
        print(f"  Pruned: {old.name}")


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _eval_pipeline(pipe, X: pd.DataFrame, y: pd.DataFrame, split: str) -> dict:
    """
    Run predict() on a fitted pipeline and return classifier + regressor metrics.
    Regressor metrics are computed on wet hours only.
    """
    out = pipe.predict(X)

    # Use pip's calibrated threshold if it has one, else default
    threshold = getattr(pipe, "threshold", 0.5)

    clf_m = evaluate_classifier(
        y[TARGET_BINARY].values, out["rain_probability"], threshold, split
    )

    # Regressor: wet hours only
    wet = y[TARGET_BINARY].values == 1
    if wet.sum() > 0:
        reg_m = evaluate_regressor(
            y.loc[y[TARGET_BINARY] == 1, TARGET_REGRESSION].values,
            out["rainfall_log1p"][wet],
            split,
        )
    else:
        reg_m = {}

    return {**clf_m, **reg_m}


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Hyperparameter sweep ──────────────────────────────────────────────────────

def _sweep(
    builder_fn,
    param_grid: dict,
    X_train, y_train,
    X_val,   y_val,
    model_label: str,
) -> tuple:
    """
    Grid search over param_grid. Trains each combination with early stopping
    on the val set. Returns (best_pipeline, best_val_metrics, all_results).
    """
    results = []
    best_f1 = -1.0
    best_pipe = None
    best_metrics = {}

    combos = list(ParameterGrid(param_grid))
    print(f"  Sweeping {len(combos)} combinations for {model_label}...")

    for i, params in enumerate(combos, 1):
        t0   = time.perf_counter()
        pipe = builder_fn(**params)
        pipe.fit(X_train, y_train, X_val, y_val)
        elapsed = time.perf_counter() - t0

        m = _eval_pipeline(pipe, X_val, y_val, "val")
        f1  = m.get("val_f1", 0.0)
        mae = m.get("val_mae_mm", float("inf"))

        results.append({"params": params, "val_f1": f1, "val_mae_mm": mae})
        print(f"    [{i:02d}/{len(combos)}] {params}  F1={f1:.4f}  MAE={mae:.4f}mm  ({elapsed:.1f}s)")

        if f1 > best_f1:
            best_f1      = f1
            best_pipe    = pipe
            best_metrics = m

    print(f"  Best {model_label}: F1={best_f1:.4f}  params={best_pipe}")
    return best_pipe, best_metrics, results


# ── Main ──────────────────────────────────────────────────────────────────────

def train(run_sweep: bool = False) -> dict:
    """
    Full training run. Returns the complete metrics dict.

    Parameters
    ----------
    run_sweep : if True, run the full hyperparameter grid.
                if False, use the default (good) hyperparameters — much faster.
    """
    all_metrics: dict = {"run_timestamp": datetime.now().isoformat(), "sweep": run_sweep}

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _section("1. LOADING DATA")
    from src.config import FEATURES_PARQUET
    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    _section("2. TIME-BASED SPLIT")
    train_df, val_df, test_df = time_split(df)

    y_train = make_targets(train_df)
    y_val   = make_targets(val_df)
    y_test  = make_targets(test_df)

    # ── 3. Persistence baseline ───────────────────────────────────────────────
    _section("3. PERSISTENCE BASELINE")
    pb = PersistenceBaseline()
    pb_val  = _eval_pipeline(pb, val_df,  y_val,  "val")
    pb_test = _eval_pipeline(pb, test_df, y_test, "test")
    print(f"  Val  F1={pb_val.get('val_f1', 0):.4f}  MAE={pb_val.get('val_mae_mm', 0):.4f}mm")
    print(f"  Test F1={pb_test.get('test_f1', 0):.4f}  MAE={pb_test.get('test_mae_mm', 0):.4f}mm")
    all_metrics["persistence"] = {**pb_val, **pb_test}

    # ── 4. Climatology baseline ───────────────────────────────────────────────
    _section("4. CLIMATOLOGY BASELINE")
    cb = ClimatologyBaseline().fit(train_df)
    cb_val  = _eval_pipeline(cb, val_df,  y_val,  "val")
    cb_test = _eval_pipeline(cb, test_df, y_test, "test")
    print(f"  Val  F1={cb_val.get('val_f1', 0):.4f}  MAE={cb_val.get('val_mae_mm', 0):.4f}mm")
    print(f"  Test F1={cb_test.get('test_f1', 0):.4f}  MAE={cb_test.get('test_mae_mm', 0):.4f}mm")
    all_metrics["climatology"] = {**cb_val, **cb_test}

    # ── 5. XGBoost ────────────────────────────────────────────────────────────
    _section("5. XGBOOST TWO-STAGE")
    if run_sweep:
        xgb_pipe, xgb_val_m, xgb_sweep = _sweep(
            build_xgb_pipeline, XGB_PARAM_GRID,
            train_df, y_train, val_df, y_val, "XGBoost"
        )
        all_metrics["xgb_sweep"] = xgb_sweep
    else:
        print("  Training with default hyperparameters (use --sweep for grid search)")
        xgb_pipe = build_xgb_pipeline()
        xgb_pipe.fit(train_df, y_train, val_df, y_val)
        xgb_val_m = _eval_pipeline(xgb_pipe, val_df, y_val, "val")

    xgb_test_m = _eval_pipeline(xgb_pipe, test_df, y_test, "test")
    print(f"  Val  F1={xgb_val_m.get('val_f1', 0):.4f}  MAE={xgb_val_m.get('val_mae_mm', 0):.4f}mm")
    print(f"  Test F1={xgb_test_m.get('test_f1', 0):.4f}  MAE={xgb_test_m.get('test_mae_mm', 0):.4f}mm")
    all_metrics["xgboost"] = {**xgb_val_m, **xgb_test_m}
    save_model(xgb_pipe, "XGBoost_TwoStage")

    # ── 6. LightGBM ───────────────────────────────────────────────────────────
    _section("6. LIGHTGBM TWO-STAGE")
    if run_sweep:
        lgbm_pipe, lgbm_val_m, lgbm_sweep = _sweep(
            build_lgbm_pipeline, LGBM_PARAM_GRID,
            train_df, y_train, val_df, y_val, "LightGBM"
        )
        all_metrics["lgbm_sweep"] = lgbm_sweep
    else:
        print("  Training with default hyperparameters (use --sweep for grid search)")
        lgbm_pipe = build_lgbm_pipeline()
        lgbm_pipe.fit(train_df, y_train, val_df, y_val)
        lgbm_val_m = _eval_pipeline(lgbm_pipe, val_df, y_val, "val")

    lgbm_test_m = _eval_pipeline(lgbm_pipe, test_df, y_test, "test")
    print(f"  Val  F1={lgbm_val_m.get('val_f1', 0):.4f}  MAE={lgbm_val_m.get('val_mae_mm', 0):.4f}mm")
    print(f"  Test F1={lgbm_test_m.get('test_f1', 0):.4f}  MAE={lgbm_test_m.get('test_mae_mm', 0):.4f}mm")
    all_metrics["lightgbm"] = {**lgbm_val_m, **lgbm_test_m}
    save_model(lgbm_pipe, "LightGBM_TwoStage")

    # ── 7. Ensemble ───────────────────────────────────────────────────────────
    _section("7. SOFT-VOTE ENSEMBLE")
    val_maes = {
        xgb_pipe.name:  xgb_val_m.get("val_mae_mm", 1.0),
        lgbm_pipe.name: lgbm_val_m.get("val_mae_mm", 1.0),
    }
    weights_dict = compute_inverse_mae_weights(val_maes)
    weights_list = [weights_dict[xgb_pipe.name], weights_dict[lgbm_pipe.name]]
    print(f"  Ensemble weights: XGB={weights_list[0]:.4f}  LGBM={weights_list[1]:.4f}")

    # Calibrate ensemble threshold on val set
    ensemble = SoftVoteEnsemble(
        pipelines=[xgb_pipe, lgbm_pipe],
        weights=weights_list,
        threshold=0.5,
        name="SoftVoteEnsemble",
    )
    val_probs = ensemble.predict_proba(val_df)
    ens_threshold = find_optimal_threshold(y_val[TARGET_BINARY].values, val_probs)
    ensemble.threshold = ens_threshold
    print(f"  Ensemble threshold (from val): {ens_threshold:.4f}")

    # ── 8. Evaluate ensemble on test ──────────────────────────────────────────
    _section("8. ENSEMBLE TEST EVALUATION")
    ens_out = ensemble.predict(test_df)
    ens_test_m = evaluate_classifier(
        y_test[TARGET_BINARY].values, ens_out["rain_probability"],
        ensemble.threshold, "test"
    )
    wet_test = y_test[TARGET_BINARY].values == 1
    ens_reg_m = evaluate_regressor(
        y_test.loc[y_test[TARGET_BINARY] == 1, TARGET_REGRESSION].values,
        np.log1p(np.clip(ens_out["rainfall_mm"][wet_test], 0, None)),
        "test"
    )
    ens_test_full = {**ens_test_m, **ens_reg_m}
    all_metrics["ensemble"] = ens_test_full

    print(f"  Test F1={ens_test_full.get('test_f1', 0):.4f}  "
          f"MAE={ens_test_full.get('test_mae_mm', 0):.4f}mm  "
          f"ROC-AUC={ens_test_full.get('test_roc_auc', 0):.4f}")

    # Comparison vs baselines
    pb_f1  = pb_test.get("test_f1", 0)
    cb_f1  = cb_test.get("test_f1", 0)
    ens_f1 = ens_test_full.get("test_f1", 0)
    print(f"\n  Baseline comparison:")
    print(f"    Persistence F1  : {pb_f1:.4f}")
    print(f"    Climatology F1  : {cb_f1:.4f}")
    print(f"    Ensemble F1     : {ens_f1:.4f}  {'BEATS BASELINES' if ens_f1 > max(pb_f1, cb_f1) else 'DOES NOT BEAT BASELINES'}")

    save_model(ensemble, "ensemble")

    # ── 9. F1 gate ────────────────────────────────────────────────────────────
    _section("9. F1 GATE CHECK")
    check_f1_gate(ens_test_full, split="test")

    # ── 10. Write metrics.json ────────────────────────────────────────────────
    _section("10. WRITING METRICS")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  Written: {METRICS_JSON}")

    _section("TRAINING COMPLETE")
    return all_metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cork City weather forecasting models.")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run full hyperparameter grid search (27 combos per model, slow)."
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    metrics = train(run_sweep=args.sweep)
    elapsed = time.perf_counter() - t_start
    print(f"\n  Total training time: {elapsed/60:.1f} minutes")
