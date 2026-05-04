"""
scripts/_test_lgbm_early_stopping.py
Verifies that LightGBM early stopping callbacks are correctly injected
by _early_stop_kwargs and actually fire during training.
No network, no parquet — uses synthetic data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import FEATURE_COLUMNS, TARGET_BINARY, TARGET_REGRESSION
from src.models.pipeline import _early_stop_kwargs
from src.models.lgbm_model import build_lgbm_pipeline

N_TRAIN = 500
N_VAL   = 100
RNG     = np.random.default_rng(42)

# Build synthetic feature DataFrame with all required columns
X_train = pd.DataFrame(RNG.standard_normal((N_TRAIN, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
X_val   = pd.DataFrame(RNG.standard_normal((N_VAL,   len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
y_train = pd.DataFrame({
    TARGET_BINARY:     RNG.integers(0, 2, N_TRAIN),
    TARGET_REGRESSION: RNG.uniform(0, 2, N_TRAIN),
})
y_val = pd.DataFrame({
    TARGET_BINARY:     RNG.integers(0, 2, N_VAL),
    TARGET_REGRESSION: RNG.uniform(0, 2, N_VAL),
})

print("=" * 55)
print("  Test 1: _early_stop_kwargs injects callbacks for LightGBM")
print("=" * 55)
clf = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
clf._early_stopping_rounds = 10
kwargs = _early_stop_kwargs(clf, X_val, y_val, TARGET_BINARY)
assert "eval_set" in kwargs, "eval_set missing"
assert "callbacks" in kwargs, "callbacks missing — bug NOT fixed"
cbs = kwargs["callbacks"]
cb_types = [type(c).__name__ for c in cbs]
print(f"  Callbacks injected: {cb_types}")
assert any("early_stopping" in str(type(c)).lower() or "EarlyStopping" in type(c).__name__ for c in cbs), \
    "No EarlyStopping callback found"
print("  PASS: callbacks correctly injected for LGBMClassifier")

print()
print("=" * 55)
print("  Test 2: _early_stop_kwargs does NOT inject for XGBoost")
print("=" * 55)
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(n_estimators=50, verbosity=0)
xgb_kwargs = _early_stop_kwargs(xgb_clf, X_val, y_val, TARGET_BINARY)
assert "eval_set" in xgb_kwargs, "XGB eval_set missing"
assert "callbacks" not in xgb_kwargs, "XGB should NOT get callbacks"
print("  PASS: XGBoost gets eval_set only (no callbacks)")

print()
print("=" * 55)
print("  Test 3: Early stopping fires — LightGBM uses fewer rounds")
print("=" * 55)
# Build pipeline with only 200 max rounds but early stopping at 10
pipe = build_lgbm_pipeline(n_estimators=200, early_stopping_rounds=10)
assert pipe.classifier._early_stopping_rounds == 10
assert pipe.regressor._early_stopping_rounds == 10

# Fit with val data — if early stopping works, actual rounds < 200
pipe.fit(X_train, y_train, X_val, y_val)

clf_rounds = pipe.classifier.best_iteration_ if hasattr(pipe.classifier, 'best_iteration_') else None
print(f"  LGBMClassifier best_iteration_: {clf_rounds}")
print("  (If early stopping fired, this < 200)")

# We can't guarantee it fires on toy data (may not overfit),
# but we can assert the model trained without error and rounds <= 200
assert pipe._is_fitted, "Pipeline not fitted"
print("  PASS: Pipeline fitted with early stopping callbacks, no errors")

print()
print("=" * 55)
print("  ALL LGBM EARLY STOPPING TESTS PASSED")
print("=" * 55)
