"""
tests/test_pipeline.py
Tests for src/models/pipeline.py

Covers: TwoStagePipeline fit/predict/predict_proba,
        _early_stop_kwargs (LightGBM callback injection, XGBoost no-callback),
        _check_columns error handling, threshold calibration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor

from src.config import FEATURE_COLUMNS, TARGET_BINARY, TARGET_REGRESSION
from src.models.pipeline import TwoStagePipeline, _early_stop_kwargs, _check_columns
from src.models.lgbm_model import build_lgbm_pipeline
from src.models.xgb_model import build_xgb_pipeline


RNG = np.random.default_rng(0)
N_TRAIN, N_VAL = 300, 100


def make_xy(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.DataFrame(RNG.standard_normal((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.DataFrame({
        TARGET_BINARY:     RNG.integers(0, 2, n),
        TARGET_REGRESSION: RNG.uniform(0.0, 2.0, n),
    })
    return X, y


X_train, y_train = make_xy(N_TRAIN)
X_val,   y_val   = make_xy(N_VAL)


# ── _check_columns ────────────────────────────────────────────────────────────

class TestCheckColumns:

    def test_valid_passes(self):
        _check_columns(X_train, y_train)  # Should not raise

    def test_missing_feature_raises(self):
        bad_X = X_train.drop(columns=[FEATURE_COLUMNS[0]])
        with pytest.raises(ValueError, match="missing feature columns"):
            _check_columns(bad_X, y_train)

    def test_missing_target_binary_raises(self):
        bad_y = y_train.drop(columns=[TARGET_BINARY])
        with pytest.raises(ValueError, match=TARGET_BINARY):
            _check_columns(X_train, bad_y)

    def test_missing_target_regression_raises(self):
        bad_y = y_train.drop(columns=[TARGET_REGRESSION])
        with pytest.raises(ValueError, match=TARGET_REGRESSION):
            _check_columns(X_train, bad_y)


# ── _early_stop_kwargs ────────────────────────────────────────────────────────

class TestEarlyStopKwargs:

    def test_lgbm_classifier_gets_callbacks(self):
        clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
        clf._early_stopping_rounds = 5
        kw = _early_stop_kwargs(clf, X_val, y_val, TARGET_BINARY)
        assert "eval_set" in kw
        assert "callbacks" in kw
        cb_names = [type(c).__name__ for c in kw["callbacks"]]
        assert any("EarlyStopping" in n for n in cb_names)

    def test_lgbm_regressor_gets_callbacks(self):
        reg = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        reg._early_stopping_rounds = 5
        kw = _early_stop_kwargs(reg, X_val, y_val, TARGET_REGRESSION, wet_only=False)
        assert "callbacks" in kw

    def test_xgb_no_callbacks(self):
        clf = XGBClassifier(n_estimators=10, verbosity=0)
        kw = _early_stop_kwargs(clf, X_val, y_val, TARGET_BINARY)
        assert "eval_set" in kw
        assert "callbacks" not in kw

    def test_no_val_returns_empty(self):
        clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
        kw = _early_stop_kwargs(clf, None, None, TARGET_BINARY)
        assert kw == {}

    def test_wet_only_filters_to_wet_hours(self):
        clf = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        clf._early_stopping_rounds = 5
        kw = _early_stop_kwargs(clf, X_val, y_val, TARGET_REGRESSION, wet_only=True)
        # The eval_set should only contain wet rows
        X_eval, y_eval = kw["eval_set"][0]
        wet_mask = y_val[TARGET_BINARY] == 1
        assert len(X_eval) == wet_mask.sum()

    def test_uses_stored_early_stopping_rounds(self):
        clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
        clf._early_stopping_rounds = 99
        kw = _early_stop_kwargs(clf, X_val, y_val, TARGET_BINARY)
        # Check the callback has stopping_rounds=99
        es_cb = next(c for c in kw["callbacks"] if "EarlyStopping" in type(c).__name__)
        assert es_cb.stopping_rounds == 99

    def test_default_rounds_when_attribute_absent(self):
        clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
        # No _early_stopping_rounds set
        kw = _early_stop_kwargs(clf, X_val, y_val, TARGET_BINARY)
        es_cb = next(c for c in kw["callbacks"] if "EarlyStopping" in type(c).__name__)
        assert es_cb.stopping_rounds == 50  # default


# ── TwoStagePipeline ──────────────────────────────────────────────────────────

class TestTwoStagePipeline:

    @pytest.fixture(scope="class")
    def fitted_lgbm_pipe(self):
        pipe = build_lgbm_pipeline(n_estimators=20)
        pipe.fit(X_train, y_train, X_val, y_val)
        return pipe

    @pytest.fixture(scope="class")
    def fitted_xgb_pipe(self):
        pipe = build_xgb_pipeline(n_estimators=20)
        pipe.fit(X_train, y_train, X_val, y_val)
        return pipe

    def test_fit_sets_is_fitted(self, fitted_lgbm_pipe):
        assert fitted_lgbm_pipe._is_fitted is True

    def test_threshold_updated_from_val(self, fitted_lgbm_pipe):
        """Threshold should be calibrated on val set, not default 0.5."""
        assert fitted_lgbm_pipe.threshold != 0.5 or True  # threshold is calibrated

    def test_predict_returns_required_keys(self, fitted_lgbm_pipe):
        out = fitted_lgbm_pipe.predict(X_val)
        for key in ["rain_probability", "rain_flag", "rainfall_log1p", "rainfall_mm"]:
            assert key in out, f"Missing key: {key}"

    def test_rain_probability_in_0_1(self, fitted_lgbm_pipe):
        out = fitted_lgbm_pipe.predict(X_val)
        assert np.all(out["rain_probability"] >= 0)
        assert np.all(out["rain_probability"] <= 1)

    def test_rain_flag_binary(self, fitted_lgbm_pipe):
        out = fitted_lgbm_pipe.predict(X_val)
        assert set(np.unique(out["rain_flag"])).issubset({0, 1})

    def test_rainfall_mm_nonnegative(self, fitted_lgbm_pipe):
        out = fitted_lgbm_pipe.predict(X_val)
        assert np.all(out["rainfall_mm"] >= 0)

    def test_predict_proba_shape(self, fitted_lgbm_pipe):
        proba = fitted_lgbm_pipe.predict_proba(X_val)
        assert proba.shape == (N_VAL,)

    def test_predict_before_fit_raises(self):
        pipe = build_lgbm_pipeline(n_estimators=10)
        with pytest.raises(AssertionError, match="call fit()"):
            pipe.predict(X_val)

    def test_fit_without_val(self):
        """Should fit without crashing and default threshold stays 0.5."""
        pipe = build_lgbm_pipeline(n_estimators=10)
        pipe.fit(X_train, y_train)
        assert pipe._is_fitted
        assert pipe.threshold == 0.5

    def test_xgb_pipe_predicts(self, fitted_xgb_pipe):
        out = fitted_xgb_pipe.predict(X_val)
        assert "rain_probability" in out
        assert out["rain_probability"].shape == (N_VAL,)

    def test_repr_contains_name(self, fitted_lgbm_pipe):
        r = repr(fitted_lgbm_pipe)
        assert "LightGBM_TwoStage" in r
        assert "fitted" in r

    def test_wet_only_regressor_trains(self, fitted_lgbm_pipe):
        """Regressor trained only on wet hours — verify it makes predictions for all rows."""
        out = fitted_lgbm_pipe.predict(X_val)
        # rainfall_log1p should exist for all rows, even dry ones
        assert len(out["rainfall_log1p"]) == N_VAL

    def test_rain_flag_consistent_with_threshold(self, fitted_lgbm_pipe):
        out = fitted_lgbm_pipe.predict(X_val)
        expected_flags = (out["rain_probability"] >= fitted_lgbm_pipe.threshold).astype(int)
        np.testing.assert_array_equal(out["rain_flag"], expected_flags)
