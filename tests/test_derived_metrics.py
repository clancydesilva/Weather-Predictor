"""
tests/test_derived_metrics.py
Tests for src/models/derived_metrics.py

Covers: apparent_temperature, outdoor_comfort_score, umbrella_inversion_risk,
        clothing_recommendation, derive_all — all boundary conditions and
        physical plausibility checks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import math
from src.models.derived_metrics import (
    apparent_temperature,
    outdoor_comfort_score,
    clothing_recommendation,
    umbrella_inversion_risk,
    derive_all,
)


# ── apparent_temperature ──────────────────────────────────────────────────────

class TestApparentTemperature:

    def test_cold_wind_chill_lower_than_actual(self):
        """Wind chill in cold weather should make it feel colder."""
        feels_like = apparent_temperature(temp_c=2.0, humidity_pct=80, wind_kmh=30)
        assert feels_like < 2.0, "Wind chill should make it feel colder"

    def test_cold_calm_wind_close_to_actual(self):
        """Calm cold: minimal wind chill, very close to actual temp."""
        feels_like = apparent_temperature(temp_c=5.0, humidity_pct=80, wind_kmh=0)
        assert abs(feels_like - 5.0) < 5.0, "Calm cold should be close to actual"

    def test_hot_humid_higher_than_actual(self):
        """Hot and humid: feels hotter than actual."""
        feels_like = apparent_temperature(temp_c=30.0, humidity_pct=90, wind_kmh=5)
        assert feels_like > 30.0, "Hot + humid should feel hotter"

    def test_hot_dry_lower_than_actual(self):
        """Hot and dry with wind: evaporative cooling."""
        feels_like = apparent_temperature(temp_c=30.0, humidity_pct=20, wind_kmh=20)
        assert feels_like < 30.0, "Hot + dry + wind should feel cooler"

    def test_mild_returns_actual(self):
        """In mild range (10-25°C), actual temp is returned as-is."""
        for temp in [10.0, 15.0, 18.0, 24.9]:
            feels_like = apparent_temperature(temp_c=temp, humidity_pct=60, wind_kmh=10)
            assert feels_like == temp, f"Mild {temp}°C should return actual temp"

    def test_returns_float(self):
        result = apparent_temperature(5.0, 70, 20)
        assert isinstance(result, float)

    def test_1dp_precision(self):
        """Result should be rounded to 1 decimal place."""
        result = apparent_temperature(2.0, 80, 25)
        assert result == round(result, 1), "Should be rounded to 1 dp"

    def test_negative_wind_clamped(self):
        """Negative wind speed is physically impossible — should not crash."""
        result = apparent_temperature(5.0, 80, -10)
        assert isinstance(result, float)

    def test_freezing(self):
        """Extreme cold and high wind — should be much below actual."""
        result = apparent_temperature(temp_c=-10.0, humidity_pct=70, wind_kmh=50)
        assert result < -10.0

    def test_boundary_exactly_10(self):
        """temp_c == 10 is in the mild zone (not wind chill)."""
        result = apparent_temperature(temp_c=10.0, humidity_pct=60, wind_kmh=30)
        assert result == 10.0

    def test_boundary_exactly_25(self):
        """temp_c == 25 is in the mild zone (not heat index)."""
        result = apparent_temperature(temp_c=25.0, humidity_pct=60, wind_kmh=5)
        assert result == 25.0


# ── outdoor_comfort_score ─────────────────────────────────────────────────────

class TestOutdoorComfortScore:

    def test_perfect_day(self):
        """Ideal Cork day: 18°C, calm, no rain, 50% humidity = near 10."""
        score = outdoor_comfort_score(temp_c=18, wind_kmh=0, rain_prob=0.0, humidity_pct=50)
        assert score >= 9.0, f"Perfect day should score >= 9, got {score}"

    def test_miserable_day(self):
        """Stormy cold wet day = low score."""
        score = outdoor_comfort_score(temp_c=3, wind_kmh=60, rain_prob=0.95, humidity_pct=95)
        assert score <= 3.0, f"Miserable day should score <= 3, got {score}"

    def test_range_0_to_10(self):
        """Score must always be in [0, 10]."""
        test_cases = [
            (18, 0, 0, 50),
            (3, 80, 1.0, 100),
            (-10, 100, 1.0, 100),
            (40, 0, 0, 0),
        ]
        for temp, wind, rain, hum in test_cases:
            s = outdoor_comfort_score(temp, wind, rain, hum)
            assert 0.0 <= s <= 10.0, f"Score {s} out of range for ({temp},{wind},{rain},{hum})"

    def test_rain_lowers_score(self):
        """Higher rain probability should decrease comfort score."""
        base = outdoor_comfort_score(15, 10, 0.0, 60)
        rainy = outdoor_comfort_score(15, 10, 0.9, 60)
        assert rainy < base, "Rain should lower comfort"

    def test_wind_lowers_score(self):
        """High wind should lower comfort score."""
        calm = outdoor_comfort_score(15, 0, 0.0, 60)
        windy = outdoor_comfort_score(15, 55, 0.0, 60)
        assert windy < calm, "Wind should lower comfort"

    def test_returns_float(self):
        assert isinstance(outdoor_comfort_score(15, 10, 0.3, 60), float)

    def test_1dp_precision(self):
        score = outdoor_comfort_score(15, 10, 0.3, 60)
        assert score == round(score, 1)

    def test_cert_rain_gives_zero_rain_component(self):
        """P(rain)=1.0 should kill the 30% rain weight."""
        s = outdoor_comfort_score(18, 0, 1.0, 50)
        # Rain score = 0, rest = 50% of 10 + 15% of 10 + 5% of 10 = 7.0
        assert s <= 7.5


# ── umbrella_inversion_risk ───────────────────────────────────────────────────

class TestUmbrellaInversionRisk:

    def test_high_wind_high_rain(self):
        assert umbrella_inversion_risk(40, 0.8) is True

    def test_low_wind_high_rain(self):
        assert umbrella_inversion_risk(20, 0.8) is False

    def test_high_wind_low_rain(self):
        assert umbrella_inversion_risk(40, 0.2) is False

    def test_boundary_wind_35(self):
        """Exactly 35 km/h is NOT above 35 — should be False."""
        assert umbrella_inversion_risk(35.0, 0.8) is False

    def test_boundary_wind_above_35(self):
        assert umbrella_inversion_risk(35.1, 0.8) is True

    def test_boundary_rain_05(self):
        """Exactly 0.5 rain prob is NOT above 0.5 — should be False."""
        assert umbrella_inversion_risk(40, 0.5) is False

    def test_returns_bool(self):
        result = umbrella_inversion_risk(30, 0.7)
        assert isinstance(result, bool)


# ── clothing_recommendation ───────────────────────────────────────────────────

class TestClothingRecommendation:

    def test_freezing_gets_heavy_coat(self):
        rec = clothing_recommendation(temp_c=2, rain_prob=0.0, wind_kmh=5)
        assert "Heavy coat" in rec["items"]
        assert "Gloves" in rec["items"]
        assert "Scarf" in rec["items"]

    def test_cool_gets_jacket(self):
        rec = clothing_recommendation(temp_c=8, rain_prob=0.0, wind_kmh=5)
        assert "Jacket" in rec["items"]
        assert "Jumper" in rec["items"]

    def test_mild_gets_light_jacket(self):
        rec = clothing_recommendation(temp_c=15, rain_prob=0.0, wind_kmh=5)
        assert "Light jacket" in rec["items"]

    def test_warm_gets_tshirt(self):
        rec = clothing_recommendation(temp_c=22, rain_prob=0.0, wind_kmh=5)
        assert "T-shirt / light layers" in rec["items"]

    def test_rain_no_wind_gives_umbrella(self):
        rec = clothing_recommendation(temp_c=15, rain_prob=0.8, wind_kmh=10)
        assert rec["umbrella"] is True
        assert rec["waterproof"] is False
        assert "Umbrella" in rec["items"]

    def test_rain_high_wind_gives_waterproof(self):
        rec = clothing_recommendation(temp_c=15, rain_prob=0.8, wind_kmh=40)
        assert rec["waterproof"] is True
        assert rec["umbrella"] is False
        assert "Waterproof jacket" in rec["items"]
        assert "Umbrella" not in rec["items"]

    def test_no_rain_no_umbrella(self):
        rec = clothing_recommendation(temp_c=15, rain_prob=0.2, wind_kmh=10)
        assert rec["umbrella"] is False
        assert rec["waterproof"] is False
        assert "Umbrella" not in rec["items"]

    def test_very_windy_adds_wind_layer(self):
        rec = clothing_recommendation(temp_c=15, rain_prob=0.0, wind_kmh=45)
        assert "Wind-resistant outer layer" in rec["items"]

    def test_wind_under_40_no_wind_layer(self):
        rec = clothing_recommendation(temp_c=15, rain_prob=0.0, wind_kmh=39)
        assert "Wind-resistant outer layer" not in rec["items"]

    def test_returns_dict_structure(self):
        rec = clothing_recommendation(temp_c=15, rain_prob=0.5, wind_kmh=20)
        assert "items" in rec
        assert "confidence" in rec
        assert "umbrella" in rec
        assert "waterproof" in rec
        assert isinstance(rec["items"], list)

    def test_confidence_shows_percentage(self):
        rec = clothing_recommendation(temp_c=15, rain_prob=0.75, wind_kmh=10)
        assert "75%" in rec["confidence"]

    def test_boundary_temp_exactly_5(self):
        """temp_c == 5 is NOT < 5, should give Jacket/Jumper not Heavy coat."""
        rec = clothing_recommendation(temp_c=5, rain_prob=0.0, wind_kmh=5)
        assert "Heavy coat" not in rec["items"]
        assert "Jacket" in rec["items"]

    def test_boundary_temp_exactly_12(self):
        """temp_c == 12 is NOT < 12, should give Light jacket."""
        rec = clothing_recommendation(temp_c=12, rain_prob=0.0, wind_kmh=5)
        assert "Jacket" not in rec["items"] or "Light jacket" in rec["items"]

    def test_rain_prob_exactly_05_no_umbrella(self):
        """rain_prob == 0.5 is NOT > 0.5 — no umbrella."""
        rec = clothing_recommendation(temp_c=15, rain_prob=0.5, wind_kmh=10)
        assert rec["umbrella"] is False


# ── derive_all ────────────────────────────────────────────────────────────────

class TestDeriveAll:

    def test_returns_all_keys(self):
        result = derive_all(15, 10, 0.3, 65)
        assert "feels_like_c" in result
        assert "comfort_score" in result
        assert "umbrella_risk" in result
        assert "clothing" in result

    def test_feels_like_is_float(self):
        result = derive_all(8, 20, 0.6, 70)
        assert isinstance(result["feels_like_c"], float)

    def test_comfort_in_range(self):
        result = derive_all(15, 10, 0.3, 65)
        assert 0.0 <= result["comfort_score"] <= 10.0

    def test_umbrella_risk_is_bool(self):
        result = derive_all(15, 40, 0.8, 70)
        assert isinstance(result["umbrella_risk"], bool)

    def test_clothing_is_dict(self):
        result = derive_all(15, 10, 0.3, 65)
        assert isinstance(result["clothing"], dict)

    def test_consistent_with_individual_functions(self):
        """derive_all should match calling functions individually."""
        t, w, r, h = 8.0, 25.0, 0.7, 75.0
        individual = {
            "feels_like_c":  apparent_temperature(t, h, w),
            "comfort_score": outdoor_comfort_score(t, w, r, h),
            "umbrella_risk": umbrella_inversion_risk(w, r),
            "clothing":      clothing_recommendation(t, r, w),
        }
        combined = derive_all(t, w, r, h)
        assert combined["feels_like_c"] == individual["feels_like_c"]
        assert combined["comfort_score"] == individual["comfort_score"]
        assert combined["umbrella_risk"] == individual["umbrella_risk"]
