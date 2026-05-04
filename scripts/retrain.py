"""
scripts/retrain.py
──────────────────
Nightly retraining orchestrator for the Cork City Weather Predictor.

Pipeline (in order):
  1. Fetch new rows from Met Éireann (exits early if none)
  2. Re-run data cleaning  (src/data/clean.py)
  3. Re-run feature engineering (src/data/features.py)
  4. Retrain XGBoost + LightGBM + ensemble  (src/train.py)
  5. Promotion gate: new val_F1 >= current val_F1 - 0.02
     - PASS → save models, update metrics.json
     - FAIL → discard new models, keep existing production versions
  6. Retrain onset/offset classifiers on updated parquet
  7. Prune old model versions (keep 3 most recent)

Usage
-----
    python scripts/retrain.py               # full retrain if new data
    python scripts/retrain.py --force       # retrain even with no new rows
    python scripts/retrain.py --skip-fetch  # skip fetch (use existing parquet)

Exit codes
----------
    0 — success (including graceful no-op when up to date)
    1 — retrain failed or promotion gate rejected
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FEATURES_PARQUET, METRICS_JSON, MODELS_DIR, RESULTS_DIR
from src.logger import get_logger

log = get_logger(__name__, log_file="pipeline.log")


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _load_current_val_f1() -> float:
    """
    Read the current production ensemble's val F1 from metrics.json.
    Returns 0.0 if the file doesn't exist (first-ever train).
    """
    if not METRICS_JSON.exists():
        return 0.0
    with open(METRICS_JSON) as f:
        metrics = json.load(f)
    # Look in ensemble section first, then fall back
    ens = metrics.get("ensemble", {})
    return float(ens.get("val_f1", ens.get("test_f1", 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly retrain orchestrator.")
    parser.add_argument("--force",       action="store_true",
                        help="Retrain even if no new rows were fetched.")
    parser.add_argument("--skip-fetch",  action="store_true",
                        help="Skip fetch step, use existing parquet.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    log.info("Retrain started")

    # ── 1. Fetch new data ─────────────────────────────────────────────────────
    new_rows = 0
    if not args.skip_fetch:
        _section("1. FETCHING LATEST DATA")
        from scripts.fetch_latest import fetch_and_append
        new_rows = fetch_and_append()
        if new_rows < 0:
            log.error("Fetch failed — aborting retrain.")
            sys.exit(1)
        if new_rows == 0 and not args.force:
            log.info("No new data — already up to date. Use --force to retrain anyway.")
            sys.exit(0)
        log.info(f"New rows fetched: {new_rows}")
    else:
        log.info("[--skip-fetch] Using existing raw CSV.")
        new_rows = -1  # sentinel: "unknown, but proceeding"

    # ── 2. Clean data ─────────────────────────────────────────────────────────
    _section("2. CLEANING DATA")
    from src.data.clean import clean_pipeline
    clean_pipeline()

    # ── 3. Feature engineering ────────────────────────────────────────────────
    _section("3. BUILDING FEATURES")
    from src.data.features import build_features
    build_features()

    # ── 4. Load current production F1 (before overwriting) ───────────────────
    _section("4. PROMOTION GATE BASELINE")
    current_val_f1 = _load_current_val_f1()
    f1_floor = current_val_f1 - 0.02
    print(f"  Current production val F1 : {current_val_f1:.4f}")
    print(f"  Promotion floor (F1 - 2%) : {f1_floor:.4f}")

    # ── 5. Retrain models ─────────────────────────────────────────────────────
    _section("5. RETRAINING ENSEMBLE")
    from src.train import train
    new_metrics = train(run_sweep=False)

    new_val_f1 = (
        new_metrics.get("ensemble", {}).get("val_f1")
        or new_metrics.get("xgboost",  {}).get("val_f1", 0.0)
    )
    log.info(f"New model val F1: {new_val_f1:.4f}")

    # ── 6. Promotion gate ─────────────────────────────────────────────────────
    _section("6. PROMOTION GATE")
    if new_val_f1 >= f1_floor:
        log.info(f"PASSED ({new_val_f1:.4f} >= {f1_floor:.4f}) — promoting to production.")
        promoted = True
    else:
        log.warning(f"FAILED ({new_val_f1:.4f} < {f1_floor:.4f}) — discarding new models.")
        promoted = False

    if not promoted:
        # Roll back: reload the previous _latest.joblib files (already on disk
        # from the last successful run — train.py overwrote them, so we need
        # to restore from the second-most-recent timestamped version).
        _rollback_latest_models()
        sys.exit(1)

    # ── 7. Retrain onset/offset classifiers ───────────────────────────────────
    _section("7. RETRAINING ONSET/OFFSET CLASSIFIERS")
    import pandas as pd
    from src.train import save_model, time_split
    from src.models.onset_offset import train_onset_offset

    df = pd.read_parquet(FEATURES_PARQUET)
    train_df, val_df, test_df = time_split(df)
    onset_clf, offset_clf, oo_metrics = train_onset_offset(train_df, val_df, test_df)
    save_model(onset_clf,  "onset_classifier")
    save_model(offset_clf, "offset_classifier")

    # Append onset/offset metrics to metrics.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON) as f:
        all_metrics = json.load(f)
    all_metrics["onset_offset"] = {
        "trained_at": datetime.now().isoformat(),
        **oo_metrics,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    _section("RETRAIN COMPLETE")
    log.info(f"New rows fetched   : {new_rows if new_rows >= 0 else 'N/A (skip-fetch)'}")
    log.info(f"New ensemble val F1: {new_val_f1:.4f}")
    log.info(f"Onset  test F1     : {oo_metrics.get('onset_test_f1', 0):.4f}")
    log.info(f"Offset test F1     : {oo_metrics.get('offset_test_f1', 0):.4f}")
    log.info(f"Total time         : {elapsed/60:.1f} minutes")


def _rollback_latest_models() -> None:
    """
    After a failed promotion gate, restore *_latest.joblib from the
    second-most-recent timestamped version (the previous production model).
    """
    import joblib

    model_names = ["ensemble", "XGBoost_TwoStage", "LightGBM_TwoStage"]
    for name in model_names:
        versioned = sorted(
            MODELS_DIR.glob(f"{name}_2*.joblib"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(versioned) >= 2:
            prev = versioned[-2]   # second-most-recent = previous production
            latest = MODELS_DIR / f"{name}_latest.joblib"
            import shutil
            shutil.copy2(prev, latest)
            print(f"  Rolled back {name}_latest → {prev.name}")
        else:
            print(f"  WARNING: No previous version to roll back to for {name}")


if __name__ == "__main__":
    main()
