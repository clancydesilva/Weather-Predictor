"""
src/models/derived_metrics.py
─────────────────────────────
Human-readable comfort and clothing metrics derived from model output.

These are post-processing functions applied AFTER the ensemble predicts
rain probability and rainfall amount. They require no additional ML model —
they are deterministic formulae from meteorology / ergonomics literature.

Formulae used:
  - Apparent temperature: Australian BOM (wind chill + heat index blend)
  - Outdoor comfort score: Weighted Gaussian composite (0–10)
  - Clothing recommendation: Rule-based on temp + rain + wind
  - Umbrella inversion risk: Wind speed threshold (35 km/h)

Public API:
    apparent_temperature(temp_c, humidity_pct, wind_kmh)   -> float
    outdoor_comfort_score(temp_c, wind_kmh, rain_prob, humidity_pct) -> float
    clothing_recommendation(temp_c, rain_prob, wind_kmh)   -> dict
    umbrella_inversion_risk(wind_kmh, rain_prob)            -> bool
    derive_all(temp_c, wind_kmh, rain_prob, humidity_pct)  -> dict
"""

import numpy as np


def apparent_temperature(
    temp_c: float,
    humidity_pct: float,
    wind_kmh: float,
) -> float:
    """
    Apparent temperature (feels-like) in °C.

    For temp < 10°C: Simplified Siple-Passel wind chill dominates.
    For temp > 25°C: Steadman 1979 heat index dominates.
    Between 10–25°C: actual temperature (neither effect strong enough to matter).

    Parameters
    ----------
    temp_c       : air temperature in Celsius
    humidity_pct : relative humidity 0–100
    wind_kmh     : wind speed in km/h

    Returns
    -------
    float : feels-like temperature in Celsius, rounded to 1 dp
    """
    wind_kmh = max(wind_kmh, 0.0)

    if temp_c < 10:
        # Wind chill: only meaningful above ~5 km/h
        v = max(wind_kmh, 5.0)
        feels_like = (
            13.12
            + 0.6215 * temp_c
            - 11.37 * (v ** 0.16)
            + 0.3965 * temp_c * (v ** 0.16)
        )
    elif temp_c > 25:
        # Heat index — vapour pressure term drives the stickiness penalty
        e = (humidity_pct / 100.0) * 6.105 * np.exp(
            17.27 * temp_c / (237.7 + temp_c)
        )
        feels_like = temp_c + 0.33 * e - 0.70 * (wind_kmh / 3.6) - 4.00
    else:
        feels_like = temp_c

    return round(float(feels_like), 1)


def outdoor_comfort_score(
    temp_c: float,
    wind_kmh: float,
    rain_prob: float,
    humidity_pct: float,
) -> float:
    """
    Composite outdoor comfort score from 0 (unbearable) to 10 (perfect).

    Formula weights (tuned for Cork's Atlantic climate):
      50% — Temperature (Gaussian peak at 18°C, σ=8°C)
      30% — Rain probability (linear: 0 rain = 10, certain rain = 0)
      15% — Wind speed (linear: calm = 10, ≥ 60 km/h = 0)
       5% — Humidity (Gaussian peak at 50%, σ=30%)

    Returns
    -------
    float : comfort score 0.0–10.0, rounded to 1 dp
    """
    # Temperature: Gaussian centred on 18°C, σ=8°C
    temp_score = 10.0 * np.exp(-0.5 * ((temp_c - 18.0) / 8.0) ** 2)

    # Rain probability: linear inverse
    rain_score = 10.0 * (1.0 - float(np.clip(rain_prob, 0, 1)))

    # Wind speed: linear inverse, capped at 60 km/h
    wind_score = float(max(0.0, 10.0 * (1.0 - wind_kmh / 60.0)))

    # Humidity: Gaussian centred on 50%, σ=30%
    humidity_score = 10.0 * np.exp(-0.5 * ((humidity_pct - 50.0) / 30.0) ** 2)

    score = (
        0.50 * temp_score
        + 0.30 * rain_score
        + 0.15 * wind_score
        + 0.05 * humidity_score
    )
    return round(float(np.clip(score, 0.0, 10.0)), 1)


def umbrella_inversion_risk(wind_kmh: float, rain_prob: float) -> bool:
    """
    True if wind is strong enough to invert a standard umbrella AND rain is likely.

    Threshold: wind > 35 km/h AND P(rain) > 0.5.
    At 35 km/h (Beaufort 5, fresh breeze) standard umbrellas frequently invert.
    In that case a waterproof jacket is the better recommendation.

    Returns
    -------
    bool
    """
    return wind_kmh > 35.0 and rain_prob > 0.5


def clothing_recommendation(
    temp_c: float,
    rain_prob: float,
    wind_kmh: float,
) -> dict:
    """
    Rule-based clothing recommendation for Cork conditions.

    Returns
    -------
    dict:
        items       : list[str] — ordered clothing items
        confidence  : str       — human-readable rain summary
        umbrella    : bool      — True if umbrella recommended (no inversion risk)
        waterproof  : bool      — True if waterproof jacket recommended instead
    """
    items: list[str] = []

    # Temperature layer
    if temp_c < 5:
        items.extend(["Heavy coat", "Gloves", "Scarf"])
    elif temp_c < 12:
        items.extend(["Jacket", "Jumper"])
    elif temp_c < 18:
        items.append("Light jacket")
    else:
        items.append("T-shirt / light layers")

    # Rain layer
    inversion = umbrella_inversion_risk(wind_kmh, rain_prob)
    umbrella   = rain_prob > 0.5 and not inversion
    waterproof = rain_prob > 0.5 and inversion

    if waterproof:
        items.append("Waterproof jacket")
    elif umbrella:
        items.append("Umbrella")

    # Extra wind layer
    if wind_kmh > 40:
        items.append("Wind-resistant outer layer")

    return {
        "items":      items,
        "confidence": f"{int(rain_prob * 100)}% rain chance",
        "umbrella":   umbrella,
        "waterproof": waterproof,
    }


def derive_all(
    temp_c: float,
    wind_kmh: float,
    rain_prob: float,
    humidity_pct: float,
) -> dict:
    """
    Convenience wrapper — compute all derived metrics in one call.

    Returns
    -------
    dict:
        feels_like_c   : float
        comfort_score  : float
        umbrella_risk  : bool
        clothing       : dict (from clothing_recommendation)
    """
    return {
        "feels_like_c":  apparent_temperature(temp_c, humidity_pct, wind_kmh),
        "comfort_score": outdoor_comfort_score(temp_c, wind_kmh, rain_prob, humidity_pct),
        "umbrella_risk": umbrella_inversion_risk(wind_kmh, rain_prob),
        "clothing":      clothing_recommendation(temp_c, rain_prob, wind_kmh),
    }
