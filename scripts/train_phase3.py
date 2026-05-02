"""
scripts/train_phase3.py
───────────────────────
Phase 3 training orchestrator: onset and offset transition classifiers.

Designed to run *after* Phase 2 (src/train.py) has already produced
the feature parquet and ensemble model.  Does NOT re-train XGBoost or
LightGBM — it loads the existing parquet, trains the two onset/offset
classifiers, saves them, and appends their metrics to results/metrics.json.

Usage
-----
    python scripts/train_phase3.py

Output
------
    models/onset_classifier_latest.joblib
    models/offset_classifier_latest.joblib
    results/metrics.json  (onset_offset section appended/updated)
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    FEATURES_PARQUET,
    METRICS_JSON,
    MODELS_DIR,
    RESULTS_DIR,
    TRAIN_END_DATE,
    VAL_END_DATE,
    TEST_START_DATE,
)
from src.models.onset_offset import (
    TARGET_ONSET,
    TARGET_OFFSET,
    train_onset_offset,
)
from src.train import save_model, time_split


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    t_start = time.perf_counter()
    print(f"Phase 3 — Onset/Offset Classifier Training")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Load features ──────────────────────────────────────────────────────
    _section("1. LOADING FEATURES")
    if not FEATURES_PARQUET.exists():
        print(f"ERROR: {FEATURES_PARQUET} not found. Run 'python -m src.train' first.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns")

    # Verify onset/offset columns exist
    for col in (TARGET_ONSET, TARGET_OFFSET):
        if col not in df.columns:
            print(f"ERROR: Column '{col}' missing from parquet.")
            print("       Re-run 'python -m src.data.features' to regenerate.")
            sys.exit(1)

    # ── 2. Split ──────────────────────────────────────────────────────────────
    _section("2. TIME-BASED SPLIT")
    train_df, val_df, test_df = time_split(df)

    # ── 3. Label counts ───────────────────────────────────────────────────────
    _section("3. LABEL COUNTS")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for col in (TARGET_ONSET, TARGET_OFFSET):
            n   = split_df[col].sum()
            pct = 100 * split_df[col].mean()
            print(f"  {split_name:5}  {col:15}: {n:,} events  ({pct:.2f}%)")

    # ── 4. Train classifiers ──────────────────────────────────────────────────
    _section("4. TRAINING ONSET AND OFFSET CLASSIFIERS")
    onset_clf, offset_clf, metrics = train_onset_offset(train_df, val_df, test_df)

    # ── 5. Save models ────────────────────────────────────────────────────────
    _section("5. SAVING MODELS")
    save_model(onset_clf,  "onset_classifier")
    save_model(offset_clf, "offset_classifier")

    # ── 6. Sample predictions on first week of test ───────────────────────────
    _section("6. SAMPLE PREDICTIONS (first 168h of test set)")
    sample = test_df.iloc[:168]

    onset_events  = onset_clf.predict_events(sample)
    offset_events = offset_clf.predict_events(sample)

    print(f"\n  Onset events predicted  : {len(onset_events)}")
    for e in onset_events[:8]:
        print(f"    {e['datetime']}  confidence={e['confidence']:.3f}  [{e['event']}]")

    print(f"\n  Offset events predicted : {len(offset_events)}")
    for e in offset_events[:8]:
        print(f"    {e['datetime']}  confidence={e['confidence']:.3f}  [{e['event']}]")

    # ── 7. Update metrics.json ────────────────────────────────────────────────
    _section("7. UPDATING METRICS")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing metrics if they exist, otherwise start fresh
    existing: dict = {}
    if METRICS_JSON.exists():
        with open(METRICS_JSON) as f:
            existing = json.load(f)

    existing["onset_offset"] = {
        "trained_at": datetime.now().isoformat(),
        **metrics,
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"  Updated: {METRICS_JSON}")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    _section("PHASE 3 COMPLETE")
    print(f"  Onset  test F1 : {metrics.get('onset_test_f1', 0):.4f}")
    print(f"  Offset test F1 : {metrics.get('offset_test_f1', 0):.4f}")
    print(f"  Onset  P@3     : {metrics.get('onset_test_precision_at_3', 0):.4f}")
    print(f"  Offset P@3     : {metrics.get('offset_test_precision_at_3', 0):.4f}")
    print(f"\n  Models saved to: {MODELS_DIR}")
    print(f"  Total time     : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
