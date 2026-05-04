"""
src/predict.py
──────────────
Inference layer: takes the last 72h of observations and produces a 24h
hourly forecast, including rain probability, rainfall amount, onset/offset
events, derived comfort metrics, and clothing recommendations.

This is the bridge between the trained models and the API. The API calls
`generate_forecast()` — it never touches model files or parquet directly.

Design decisions:
  - All models are passed in as arguments (no global state here).
    The API's lifespan handler loads models once on startup and passes them in.
  - Input is a slice of hourly_features.parquet (already-engineered features).
    The API reads the last 72h from the live parquet, which is updated by
    fetch_latest.py / retrain.py in Phase 5.
  - Output is a list of dicts (one per forecast hour) plus a summary.
    The API's Pydantic schemas validate and serialise this output.

Public API:
    generate_forecast(feature_df, ensemble, onset_clf, offset_clf) -> dict
    get_forecast_window(features_df, n_hours=24) -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from src.models.derived_metrics import derive_all
from src.models.onset_offset import OnsetOffsetClassifier
from src.models.ensemble import SoftVoteEnsemble


def get_forecast_window(
    features_df: pd.DataFrame,
    n_hours: int = 24,
) -> pd.DataFrame:
    """
    Slice the last `n_hours` rows from a feature DataFrame.

    In production the feature DataFrame is built from the live-updated parquet.
    The last row is the most recent observation; the previous 71 rows provide
    the lag/rolling context the models need.

    Parameters
    ----------
    features_df : full hourly_features.parquet DataFrame (or subset)
    n_hours     : number of forecast hours to return (default 24)

    Returns
    -------
    pd.DataFrame : last n_hours rows — the forecast window
    """
    if len(features_df) < n_hours:
        raise ValueError(
            f"Need at least {n_hours} rows for a {n_hours}h forecast, "
            f"got {len(features_df)}."
        )
    return features_df.iloc[-n_hours:]


def generate_forecast(
    forecast_window: pd.DataFrame,
    ensemble: SoftVoteEnsemble,
    onset_clf: OnsetOffsetClassifier,
    offset_clf: OnsetOffsetClassifier,
) -> dict:
    """
    Run all models on a forecast window and return a structured forecast dict.

    Parameters
    ----------
    forecast_window : pd.DataFrame
        Exactly 24 rows (or however many hours you want to forecast).
        Must contain all FEATURE_COLUMNS plus raw met fields for derived metrics.
    ensemble        : fitted SoftVoteEnsemble
    onset_clf       : fitted OnsetOffsetClassifier (event_type='onset')
    offset_clf      : fitted OnsetOffsetClassifier (event_type='offset')

    Returns
    -------
    dict:
        hours          : list[dict] — one entry per forecast hour
        onset_events   : list[dict] — predicted rain start times
        offset_events  : list[dict] — predicted rain stop times
        daily_summary  : dict       — max/min temp, total rain, comfort
    """
    # ── Stage 1: Ensemble rain probability + amount ───────────────────────────
    ens_out    = ensemble.predict(forecast_window)
    rain_probs = ens_out["rain_probability"]   # shape (n_hours,)
    rain_flags = ens_out["rain_flag"]           # shape (n_hours,)
    rain_mm    = ens_out["rainfall_mm"]         # shape (n_hours,)

    # Stage 2: Onset / offset events (None when phase-3 models not loaded)
    onset_events  = onset_clf.predict_events(forecast_window)  if onset_clf  else []
    offset_events = offset_clf.predict_events(forecast_window) if offset_clf else []

    # ── Stage 3: Build per-hour output ───────────────────────────────────────
    hours = []
    for i, (ts, row) in enumerate(forecast_window.iterrows()):
        temp_c       = float(row.get("temp_c", 0))
        wind_kmh     = float(row.get("wind_speed_kmh", 0))
        humidity_pct = float(row.get("humidity_pct", 70))
        pressure_hpa = float(row.get("pressure_hpa", 1013))
        wind_dir_deg = int(row.get("wind_dir_deg", 0)) if not pd.isna(row.get("wind_dir_deg", 0)) else 0

        p_rain   = float(rain_probs[i])
        pred_mm  = float(rain_mm[i])
        flag     = int(rain_flags[i])

        derived = derive_all(temp_c, wind_kmh, p_rain, humidity_pct)

        hours.append({
            "datetime":        ts.isoformat(),
            "temp_c":          round(temp_c, 1),
            "feels_like_c":    derived["feels_like_c"],
            "rain_probability": round(p_rain, 3),
            "rain_flag":       flag,
            "rainfall_mm":     round(max(pred_mm, 0.0), 2),
            "wind_speed_kmh":  round(wind_kmh, 1),
            "wind_dir_deg":    wind_dir_deg,
            "humidity_pct":    round(humidity_pct, 1),
            "pressure_hpa":    round(pressure_hpa, 1),
            "comfort_score":   derived["comfort_score"],
            "umbrella_risk":   derived["umbrella_risk"],
            "clothing":        derived["clothing"],
        })

    # ── Stage 4: Daily summary ────────────────────────────────────────────────
    temps        = [h["temp_c"] for h in hours]
    comfort_vals = [h["comfort_score"] for h in hours]
    total_rain   = sum(h["rainfall_mm"] for h in hours)
    peak_rain_p  = max(h["rain_probability"] for h in hours)
    avg_comfort  = round(sum(comfort_vals) / len(comfort_vals), 1) if comfort_vals else 0.0

    daily_summary = {
        "max_temp_c":          round(max(temps), 1),
        "min_temp_c":          round(min(temps), 1),
        "total_rainfall_mm":   round(total_rain, 2),
        "peak_rain_probability": round(peak_rain_p, 3),
        "avg_comfort_score":   avg_comfort,
        "rain_hours":          int(sum(h["rain_flag"] for h in hours)),
        "forecast_hours":      len(hours),
    }

    return {
        "hours":         hours,
        "onset_events":  onset_events,
        "offset_events": offset_events,
        "daily_summary": daily_summary,
    }
