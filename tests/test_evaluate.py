"""
tests/test_evaluate.py
Tests for src/evaluate.py

Covers: find_optimal_threshold, evaluate_classifier, evaluate_regressor,
        evaluate_continuous, check_f1_gate, precision_at_k
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from src.evaluate import (
    find_optimal_threshold,
    evaluate_classifier,
    evaluate_regressor,
    evaluate_continuous,
    check_f1_gate,
    precision_at_k,
)


RNG = np.random.default_rng(42)


# ── find_optimal_threshold ────────────────────────────────────────────────────

class TestFindOptimalThreshold:

    def test_returns_float(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        t = find_optimal_threshold(y, p)
        assert isinstance(t, float)

    def test_threshold_in_0_1(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        t = find_optimal_threshold(y, p)
        assert 0.0 <= t <= 1.0

    def test_perfect_classifier_high_threshold(self):
        """Perfect probabilities -> optimal threshold between positives and negatives."""
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
        t = find_optimal_threshold(y, p)
        assert 0.2 <= t <= 0.9, f"Expected threshold between classes, got {t}"

    def test_always_rain_gives_some_threshold(self):
        """Extreme case: all positive labels."""
        y = np.array([1, 1, 1, 1])
        p = np.array([0.8, 0.9, 0.7, 0.6])
        t = find_optimal_threshold(y, p)
        assert 0.0 <= t <= 1.0

    def test_large_dataset_stable(self):
        y = RNG.integers(0, 2, 1000)
        p = RNG.uniform(0, 1, 1000)
        t = find_optimal_threshold(y, p)
        assert 0.0 <= t <= 1.0


# ── evaluate_classifier ───────────────────────────────────────────────────────

class TestEvaluateClassifier:

    def _make_data(self, n=200, noise=0.2):
        y = RNG.integers(0, 2, n)
        p = np.clip(y + RNG.normal(0, noise, n), 0, 1)
        return y, p

    def test_keys_present(self):
        y, p = self._make_data()
        m = evaluate_classifier(y, p, 0.5, "val")
        for key in ["val_roc_auc", "val_f1", "val_precision", "val_recall",
                    "val_avg_precision", "val_threshold"]:
            assert key in m, f"Missing key: {key}"

    def test_split_prefix_test(self):
        y, p = self._make_data()
        m = evaluate_classifier(y, p, 0.5, "test")
        assert "test_f1" in m
        assert "val_f1" not in m

    def test_roc_auc_in_range(self):
        y, p = self._make_data()
        m = evaluate_classifier(y, p, 0.5)
        assert 0.0 <= m["val_roc_auc"] <= 1.0

    def test_f1_in_range(self):
        y, p = self._make_data()
        m = evaluate_classifier(y, p, 0.5)
        assert 0.0 <= m["val_f1"] <= 1.0

    def test_perfect_classifier(self):
        """Perfect predictions should give F1=1.0, AUC=1.0."""
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        m = evaluate_classifier(y, p, 0.5)
        assert m["val_f1"] == 1.0
        assert m["val_roc_auc"] == 1.0

    def test_all_zeros_no_crash(self):
        """Model predicts all zeros — zero_division guard should prevent crash."""
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.1, 0.2, 0.2])
        m = evaluate_classifier(y, p, 0.9)  # high threshold -> all predicted 0
        assert m["val_f1"] == 0.0

    def test_threshold_persisted_in_output(self):
        y, p = self._make_data()
        m = evaluate_classifier(y, p, 0.37)
        assert m["val_threshold"] == round(0.37, 4)


# ── evaluate_regressor ────────────────────────────────────────────────────────

class TestEvaluateRegressor:

    def test_keys_present(self):
        y = np.log1p(RNG.uniform(0.1, 5, 100))
        p = y + RNG.normal(0, 0.1, 100)
        m = evaluate_regressor(y, p, "val")
        for key in ["val_mae_log", "val_rmse_log", "val_mae_mm", "val_rmse_mm", "val_r2_mm"]:
            assert key in m, f"Missing key: {key}"

    def test_perfect_regressor(self):
        y = np.log1p(np.array([0.5, 1.0, 2.0, 0.3]))
        m = evaluate_regressor(y, y, "val")
        assert m["val_mae_log"] == pytest.approx(0.0, abs=1e-9)
        assert m["val_mae_mm"] == pytest.approx(0.0, abs=1e-9)
        assert m["val_r2_mm"] == pytest.approx(1.0, abs=1e-6)

    def test_inverse_transform_applied(self):
        """mae_mm should be in physical mm scale, not log scale."""
        y = np.log1p(np.array([1.0, 2.0, 3.0]))
        p = y + 0.5  # half-log offset
        m = evaluate_regressor(y, p, "val")
        assert m["val_mae_mm"] > m["val_mae_log"]  # mm values > log values

    def test_negative_predictions_clipped(self):
        """Negative predictions in log space should be clipped, not crash."""
        y = np.array([0.5, 1.0, 0.8])
        p = np.array([-0.1, -0.5, 0.8])  # some negative log predictions
        m = evaluate_regressor(y, p, "val")
        assert isinstance(m["val_mae_mm"], float)

    def test_split_prefix(self):
        y = np.log1p(np.array([0.5, 1.0]))
        m = evaluate_regressor(y, y, "test")
        assert "test_mae_mm" in m
        assert "val_mae_mm" not in m


# ── evaluate_continuous ───────────────────────────────────────────────────────

class TestEvaluateContinuous:

    def test_keys_present(self):
        y = RNG.uniform(5, 20, 50)
        p = y + RNG.normal(0, 0.5, 50)
        m = evaluate_continuous(y, p, "temp_c", "val")
        for key in ["val_temp_c_mae", "val_temp_c_rmse", "val_temp_c_r2"]:
            assert key in m

    def test_perfect_continuous(self):
        y = np.array([10.0, 12.0, 15.0])
        m = evaluate_continuous(y, y, "temp_c", "val")
        assert m["val_temp_c_mae"] == pytest.approx(0.0)
        assert m["val_temp_c_r2"] == pytest.approx(1.0)

    def test_custom_variable_name(self):
        y = RNG.uniform(0, 100, 30)
        m = evaluate_continuous(y, y, "humidity_pct", "test")
        assert "test_humidity_pct_mae" in m


# ── check_f1_gate ─────────────────────────────────────────────────────────────

class TestCheckF1Gate:

    def test_no_crash_above_target(self, capsys):
        check_f1_gate({"test_f1": 0.75, "test_mae_mm": 0.4}, "test")
        out = capsys.readouterr().out
        assert "GATE PASSED" in out

    def test_no_crash_below_floor(self, capsys):
        check_f1_gate({"test_f1": 0.40, "test_mae_mm": 0.4}, "test")
        out = capsys.readouterr().out
        assert "BELOW FLOOR" in out

    def test_f1_above_target_but_mae_high(self, capsys):
        check_f1_gate({"test_f1": 0.75, "test_mae_mm": 2.0}, "test")
        out = capsys.readouterr().out
        assert "MAE" in out

    def test_above_floor_below_target(self, capsys):
        check_f1_gate({"test_f1": 0.62, "test_mae_mm": 0.4}, "test")
        out = capsys.readouterr().out
        assert "floor" in out.lower() or "target" in out.lower()

    def test_missing_f1_uses_zero(self, capsys):
        """Missing f1 key should default to 0.0 and not crash."""
        check_f1_gate({}, "test")
        # Should not raise

    def test_val_split(self, capsys):
        check_f1_gate({"val_f1": 0.75, "val_mae_mm": 0.4}, "val")
        out = capsys.readouterr().out
        assert "val" in out.lower()


# ── precision_at_k ────────────────────────────────────────────────────────────

class TestPrecisionAtK:

    def test_perfect_top3(self):
        """Top 3 by prob are all true positives."""
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.95])
        assert precision_at_k(y, p, k=3) == pytest.approx(1.0)

    def test_zero_top3(self):
        """Top 3 by prob are all false positives."""
        y = np.array([1, 1, 1, 0, 0, 0])
        p = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.95])
        assert precision_at_k(y, p, k=3) == pytest.approx(0.0)

    def test_mixed_top3(self):
        """Top 3 has 2 TP and 1 FP."""
        y = np.array([0, 1, 1, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.85, 0.1, 0.1, 0.8])
        # Top 3 by prob: indices 0, 2, 5 -> y=[0,1,1] -> 2/3
        assert precision_at_k(y, p, k=3) == pytest.approx(2/3)

    def test_k_larger_than_n(self):
        """k > len(y) should clamp to len(y) and not crash."""
        y = np.array([1, 0, 1])
        p = np.array([0.8, 0.3, 0.9])
        result = precision_at_k(y, p, k=10)
        assert 0.0 <= result <= 1.0

    def test_k1(self):
        """k=1: only top prediction matters."""
        y = np.array([0, 1, 0])
        p = np.array([0.1, 0.9, 0.2])
        assert precision_at_k(y, p, k=1) == pytest.approx(1.0)

    def test_returns_float(self):
        y = np.array([0, 1, 1])
        p = np.array([0.3, 0.7, 0.8])
        assert isinstance(precision_at_k(y, p, k=2), float)
