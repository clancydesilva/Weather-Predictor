"""
tests/test_ensemble.py
Tests for src/models/ensemble.py

Covers: SoftVoteEnsemble, compute_inverse_mae_weights
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_BINARY, TARGET_REGRESSION
from src.models.ensemble import SoftVoteEnsemble, compute_inverse_mae_weights
from src.models.lgbm_model import build_lgbm_pipeline
from src.models.xgb_model import build_xgb_pipeline


RNG = np.random.default_rng(7)
N_TRAIN, N_VAL = 200, 60


def make_xy(n):
    X = pd.DataFrame(RNG.standard_normal((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.DataFrame({
        TARGET_BINARY:     RNG.integers(0, 2, n),
        TARGET_REGRESSION: RNG.uniform(0, 2, n),
    })
    return X, y


X_train, y_train = make_xy(N_TRAIN)
X_val,   y_val   = make_xy(N_VAL)


@pytest.fixture(scope="module")
def fitted_pipes():
    lgbm = build_lgbm_pipeline(n_estimators=15)
    lgbm.fit(X_train, y_train, X_val, y_val)
    xgb = build_xgb_pipeline(n_estimators=15)
    xgb.fit(X_train, y_train, X_val, y_val)
    return lgbm, xgb


@pytest.fixture(scope="module")
def ensemble(fitted_pipes):
    lgbm, xgb = fitted_pipes
    ens = SoftVoteEnsemble(
        pipelines=[lgbm, xgb],
        weights=[0.5, 0.5],
        threshold=0.5,
        name="TestEnsemble",
    )
    return ens


# ── compute_inverse_mae_weights ───────────────────────────────────────────────

class TestComputeInverseMAEWeights:

    def test_lower_mae_gets_higher_weight(self):
        weights = compute_inverse_mae_weights({"A": 0.2, "B": 0.5})
        assert weights["A"] > weights["B"]

    def test_weights_sum_to_one(self):
        weights = compute_inverse_mae_weights({"A": 0.3, "B": 0.4, "C": 0.5})
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_equal_mae_equal_weights(self):
        weights = compute_inverse_mae_weights({"A": 0.5, "B": 0.5})
        assert weights["A"] == pytest.approx(weights["B"])

    def test_single_model(self):
        weights = compute_inverse_mae_weights({"A": 0.3})
        assert weights["A"] == pytest.approx(1.0)

    def test_all_keys_present(self):
        mae_dict = {"XGB": 0.3, "LGBM": 0.4}
        weights = compute_inverse_mae_weights(mae_dict)
        assert set(weights.keys()) == set(mae_dict.keys())


# ── SoftVoteEnsemble ──────────────────────────────────────────────────────────

class TestSoftVoteEnsemble:

    def test_predict_proba_shape(self, ensemble):
        proba = ensemble.predict_proba(X_val)
        assert proba.shape == (N_VAL,)

    def test_predict_proba_in_0_1(self, ensemble):
        proba = ensemble.predict_proba(X_val)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_returns_keys(self, ensemble):
        out = ensemble.predict(X_val)
        for key in ["rain_probability", "rain_flag", "rainfall_mm"]:
            assert key in out

    def test_rain_flag_consistent_with_threshold(self, ensemble):
        out = ensemble.predict(X_val)
        expected = (out["rain_probability"] >= ensemble.threshold).astype(int)
        np.testing.assert_array_equal(out["rain_flag"], expected)

    def test_rainfall_mm_nonnegative(self, ensemble):
        out = ensemble.predict(X_val)
        assert np.all(out["rainfall_mm"] >= 0)

    def test_weights_applied(self, fitted_pipes):
        """Check that changing weights changes probabilities."""
        lgbm, xgb = fitted_pipes
        ens_equal = SoftVoteEnsemble([lgbm, xgb], [0.5, 0.5], threshold=0.5)
        ens_lgbm  = SoftVoteEnsemble([lgbm, xgb], [0.9, 0.1], threshold=0.5)
        p_equal = ens_equal.predict_proba(X_val)
        p_lgbm  = ens_lgbm.predict_proba(X_val)
        # They should differ when models disagree
        assert not np.allclose(p_equal, p_lgbm), "Weights have no effect — bug!"

    def test_custom_threshold(self, fitted_pipes):
        lgbm, xgb = fitted_pipes
        ens_low  = SoftVoteEnsemble([lgbm, xgb], [0.5, 0.5], threshold=0.1)
        ens_high = SoftVoteEnsemble([lgbm, xgb], [0.5, 0.5], threshold=0.9)
        out_low  = ens_low.predict(X_val)
        out_high = ens_high.predict(X_val)
        # Low threshold -> more rain flags; high threshold -> fewer
        assert out_low["rain_flag"].sum() >= out_high["rain_flag"].sum()

    def test_single_pipeline_ensemble(self, fitted_pipes):
        lgbm, _ = fitted_pipes
        ens = SoftVoteEnsemble([lgbm], [1.0], threshold=0.5)
        out = ens.predict(X_val)
        assert "rain_probability" in out
