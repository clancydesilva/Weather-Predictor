"""
tests/test_predict.py
Tests for src/predict.py

Covers: get_forecast_window, generate_forecast — all paths,
        edge cases, schema validation of output structure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.config import FEATURE_COLUMNS, TARGET_BINARY, TARGET_REGRESSION
from src.predict import get_forecast_window, generate_forecast


RNG = np.random.default_rng(99)


def make_feature_df(n: int = 48) -> pd.DataFrame:
    """Create a fake feature DataFrame with realistic structure."""
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    data = {col: RNG.standard_normal(n) for col in FEATURE_COLUMNS}
    # Add raw met columns used in generate_forecast
    data["temp_c"]       = RNG.uniform(8, 20, n)
    data["wind_speed_kmh"] = RNG.uniform(5, 40, n)
    data["humidity_pct"]  = RNG.uniform(50, 90, n)
    data["pressure_hpa"]  = RNG.uniform(990, 1030, n)
    data["wind_dir_deg"]  = RNG.uniform(0, 360, n)
    return pd.DataFrame(data, index=idx)


def make_mock_ensemble(n_hours: int = 24):
    ens = MagicMock()
    ens.predict.return_value = {
        "rain_probability": RNG.uniform(0, 1, n_hours),
        "rain_flag":        RNG.integers(0, 2, n_hours),
        "rainfall_mm":      RNG.uniform(0, 2, n_hours),
    }
    return ens


def make_mock_onset_clf():
    clf = MagicMock()
    clf.predict_events.return_value = [
        {"event": "onset", "datetime": "2025-01-01T06:00:00",
         "confidence": 0.8, "message": "Rain expected to start around 06:00"}
    ]
    return clf


def make_mock_offset_clf():
    clf = MagicMock()
    clf.predict_events.return_value = []
    return clf


# ── get_forecast_window ───────────────────────────────────────────────────────

class TestGetForecastWindow:

    def test_returns_last_n_hours(self):
        df = make_feature_df(100)
        window = get_forecast_window(df, n_hours=24)
        assert len(window) == 24

    def test_returns_last_rows(self):
        df = make_feature_df(100)
        window = get_forecast_window(df, n_hours=10)
        pd.testing.assert_frame_equal(window, df.iloc[-10:])

    def test_exact_size_works(self):
        df = make_feature_df(24)
        window = get_forecast_window(df, n_hours=24)
        assert len(window) == 24

    def test_too_few_rows_raises(self):
        df = make_feature_df(10)
        with pytest.raises(ValueError, match="at least 24 rows"):
            get_forecast_window(df, n_hours=24)

    def test_one_row_window(self):
        df = make_feature_df(50)
        window = get_forecast_window(df, n_hours=1)
        assert len(window) == 1
        pd.testing.assert_frame_equal(window, df.iloc[-1:])

    def test_preserves_index(self):
        df = make_feature_df(50)
        window = get_forecast_window(df, n_hours=24)
        assert window.index[-1] == df.index[-1]

    def test_returns_dataframe(self):
        df = make_feature_df(50)
        result = get_forecast_window(df, n_hours=10)
        assert isinstance(result, pd.DataFrame)


# ── generate_forecast ─────────────────────────────────────────────────────────

class TestGenerateForecast:

    @pytest.fixture
    def forecast_inputs(self):
        n = 24
        window = make_feature_df(n).iloc[:n]
        ensemble = make_mock_ensemble(n)
        onset_clf = make_mock_onset_clf()
        offset_clf = make_mock_offset_clf()
        return window, ensemble, onset_clf, offset_clf

    def test_returns_top_level_keys(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        for key in ["hours", "onset_events", "offset_events", "daily_summary"]:
            assert key in result, f"Missing top-level key: {key}"

    def test_hours_count(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        assert len(result["hours"]) == 24

    def test_hour_entry_keys(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        h = result["hours"][0]
        for key in ["datetime", "temp_c", "feels_like_c", "rain_probability",
                    "rain_flag", "rainfall_mm", "wind_speed_kmh", "wind_dir_deg",
                    "humidity_pct", "pressure_hpa", "comfort_score",
                    "umbrella_risk", "clothing"]:
            assert key in h, f"Missing hour key: {key}"

    def test_rain_probability_in_range(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        for h in result["hours"]:
            assert 0.0 <= h["rain_probability"] <= 1.0

    def test_rainfall_mm_nonnegative(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        for h in result["hours"]:
            assert h["rainfall_mm"] >= 0.0

    def test_rain_flag_binary(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        for h in result["hours"]:
            assert h["rain_flag"] in (0, 1)

    def test_comfort_score_in_range(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        for h in result["hours"]:
            assert 0.0 <= h["comfort_score"] <= 10.0

    def test_daily_summary_keys(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        ds = result["daily_summary"]
        for key in ["max_temp_c", "min_temp_c", "total_rainfall_mm",
                    "peak_rain_probability", "avg_comfort_score",
                    "rain_hours", "forecast_hours"]:
            assert key in ds

    def test_daily_summary_math(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        ds = result["daily_summary"]
        hours = result["hours"]
        assert ds["max_temp_c"] == max(h["temp_c"] for h in hours)
        assert ds["min_temp_c"] == min(h["temp_c"] for h in hours)
        assert ds["forecast_hours"] == 24
        assert ds["rain_hours"] == sum(h["rain_flag"] for h in hours)

    def test_onset_events_passed_through(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        assert len(result["onset_events"]) == 1  # from mock

    def test_offset_events_empty(self, forecast_inputs):
        window, ens, onset, offset = forecast_inputs
        result = generate_forecast(window, ens, onset, offset)
        assert result["offset_events"] == []

    def test_wind_dir_nan_handled(self, forecast_inputs):
        """wind_dir_deg NaN should not crash (even though parquet has no NaNs)."""
        window, ens, onset, offset = forecast_inputs
        window = window.copy()
        window["wind_dir_deg"] = float("nan")
        result = generate_forecast(window, ens, onset, offset)
        for h in result["hours"]:
            assert h["wind_dir_deg"] == 0  # coerced to 0

    def test_8h_window(self):
        """Non-24h windows should work."""
        n = 8
        window = make_feature_df(50).iloc[:n]
        ens = make_mock_ensemble(n)
        onset = make_mock_onset_clf()
        offset = make_mock_offset_clf()
        result = generate_forecast(window, ens, onset, offset)
        assert len(result["hours"]) == 8
        assert result["daily_summary"]["forecast_hours"] == 8
