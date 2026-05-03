"""
api/router.py
─────────────
All forecast endpoints for the Cork City Weather API.

Endpoints:
    GET /forecast/today      — Full 24h hourly forecast
    GET /forecast/now        — Current conditions + next 3 hours
    GET /forecast/outfit     — Clothing recommendations for today
    GET /forecast/commute    — Rain/comfort at specific departure times
    GET /forecast/events     — Predicted onset/offset events for today

The router imports models and forecast data from `app.state`, which is
populated by the lifespan handler in main.py. No model files are loaded
here — this keeps the router stateless and testable in isolation.
"""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from api import schemas
from src.config import FEATURE_COLUMNS, STATION_NAME
from src.predict import generate_forecast, get_forecast_window

router = APIRouter(prefix="/forecast", tags=["Forecast"])


def _get_state(request: Request) -> dict:
    """Pull model store and feature data from app.state."""
    return {
        "ensemble":    request.app.state.ensemble,
        "onset_clf":   request.app.state.onset_clf,
        "offset_clf":  request.app.state.offset_clf,
        "features_df": request.app.state.features_df,
    }


def _build_onset_event(e: dict) -> schemas.OnsetEvent:
    """Convert a raw onset/offset dict to an OnsetEvent schema."""
    dt_str = e["datetime"].isoformat() if hasattr(e["datetime"], "isoformat") else str(e["datetime"])
    hour   = e["datetime"].hour if hasattr(e["datetime"], "hour") else "?"
    verb   = "start" if e["event"] == "onset" else "stop"
    return schemas.OnsetEvent(
        event=e["event"],
        datetime=dt_str,
        confidence=round(e["confidence"], 3),
        message=f"Rain expected to {verb} around {hour:02d}:00 (confidence {int(e['confidence']*100)}%)",
    )


def _run_forecast(state: dict, n_hours: int = 24) -> dict:
    """Run the full model stack on the last n_hours of the feature DataFrame."""
    df = state["features_df"]
    window = get_forecast_window(df, n_hours=n_hours)
    return generate_forecast(
        window,
        state["ensemble"],
        state["onset_clf"],
        state["offset_clf"],
    )


# ── /forecast/today ───────────────────────────────────────────────────────────

@router.get(
    "/today",
    response_model=schemas.DailyForecast,
    summary="Full 24-hour hourly forecast",
    description=(
        "Returns 24 hourly forecast records for the next 24 hours, including "
        "rain probability, apparent temperature, comfort score, and predicted "
        "rain onset/offset events."
    ),
)
async def forecast_today(request: Request) -> schemas.DailyForecast:
    state    = _get_state(request)
    forecast = _run_forecast(state, n_hours=24)

    hourly = [schemas.HourlyForecast(**h) for h in forecast["hours"]]
    onset  = [_build_onset_event(e) for e in forecast["onset_events"]]
    offset = [_build_onset_event(e) for e in forecast["offset_events"]]

    return schemas.DailyForecast(
        generated_at=datetime.now(timezone.utc).isoformat(),
        station=STATION_NAME,
        hours=hourly,
        onset_events=onset,
        offset_events=offset,
        daily_summary=schemas.DailySummary(**forecast["daily_summary"]),
    )


# ── /forecast/now ─────────────────────────────────────────────────────────────

@router.get(
    "/now",
    response_model=list[schemas.HourlyForecast],
    summary="Current conditions + next 3 hours",
    description="Returns 4 hourly records: the current hour and the next 3.",
)
async def forecast_now(request: Request) -> list[schemas.HourlyForecast]:
    state    = _get_state(request)
    forecast = _run_forecast(state, n_hours=4)
    return [schemas.HourlyForecast(**h) for h in forecast["hours"]]


# ── /forecast/outfit ──────────────────────────────────────────────────────────

@router.get(
    "/outfit",
    response_model=schemas.OutfitResponse,
    summary="Clothing recommendations for today",
    description=(
        "Returns a clothing recommendation based on the worst conditions "
        "expected in the next 24 hours. Distinguishes between umbrella (calm rain) "
        "and waterproof jacket (windy rain)."
    ),
)
async def forecast_outfit(request: Request) -> schemas.OutfitResponse:
    state    = _get_state(request)
    forecast = _run_forecast(state, n_hours=24)
    hours    = forecast["hours"]

    # Use worst-case conditions: peak rain hour for rain decisions,
    # average temperature for clothing layers.
    peak_hour = max(hours, key=lambda h: h["rain_probability"])
    avg_temp  = sum(h["temp_c"] for h in hours) / len(hours)

    # Re-run clothing logic on worst-case values
    clothing = peak_hour["clothing"]

    return schemas.OutfitResponse(
        items=clothing["items"],
        confidence=clothing["confidence"],
        comfort_score=forecast["daily_summary"]["avg_comfort_score"],
        umbrella_risk=peak_hour["umbrella_risk"],
        waterproof=clothing["waterproof"],
    )


# ── /forecast/commute ─────────────────────────────────────────────────────────

@router.get(
    "/commute",
    response_model=schemas.CommuteResponse,
    summary="Rain and comfort at commute departure times",
    description=(
        "Returns forecast conditions at specific morning/evening departure times. "
        "At least one of morning_depart or evening_depart must be provided. "
        "Format: HH:MM (24-hour, e.g. '08:30')."
    ),
)
async def forecast_commute(
    request: Request,
    morning_depart: Optional[str] = Query(
        default=None, description="Morning departure time HH:MM"
    ),
    evening_depart: Optional[str] = Query(
        default=None, description="Evening departure time HH:MM"
    ),
) -> schemas.CommuteResponse:
    if not morning_depart and not evening_depart:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of morning_depart or evening_depart (HH:MM).",
        )

    state    = _get_state(request)
    forecast = _run_forecast(state, n_hours=24)
    hours    = forecast["hours"]

    def _find_hour(target_hhmm: str) -> Optional[schemas.CommuteSlot]:
        """Find the forecast hour matching a HH:MM target, or nearest available."""
        try:
            target_h, target_m = map(int, target_hhmm.split(":"))
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid time format '{target_hhmm}'. Use HH:MM (e.g. '08:30').",
            )

        target_minutes = target_h * 60 + target_m
        best = None
        best_diff = float("inf")

        for h in hours:
            try:
                dt = datetime.fromisoformat(h["datetime"])
            except Exception:
                continue
            h_minutes = dt.hour * 60 + dt.minute
            diff = abs(h_minutes - target_minutes)
            if diff < best_diff:
                best_diff = diff
                best = h

        if best is None:
            return None

        p_rain = best["rain_probability"]
        if p_rain > 0.70:
            advice = "High rain risk — bring waterproof gear."
        elif p_rain > 0.40:
            advice = "Possible rain — consider an umbrella."
        else:
            advice = "Low rain risk — should be fine."

        return schemas.CommuteSlot(
            departure_time=target_hhmm,
            temp_c=best["temp_c"],
            feels_like_c=best["feels_like_c"],
            rain_probability=best["rain_probability"],
            rainfall_mm=best["rainfall_mm"],
            wind_speed_kmh=best["wind_speed_kmh"],
            comfort_score=best["comfort_score"],
            umbrella_risk=best["umbrella_risk"],
            recommendation=advice,
        )

    return schemas.CommuteResponse(
        morning=_find_hour(morning_depart) if morning_depart else None,
        evening=_find_hour(evening_depart) if evening_depart else None,
    )


# ── /forecast/events ──────────────────────────────────────────────────────────

@router.get(
    "/events",
    response_model=list[schemas.OnsetEvent],
    summary="Predicted rain onset and offset events",
    description=(
        "Returns a list of predicted rain transition events (onset = rain starts, "
        "offset = rain stops) for the next 24 hours, ordered by datetime."
    ),
)
async def forecast_events(request: Request) -> list[schemas.OnsetEvent]:
    state    = _get_state(request)
    forecast = _run_forecast(state, n_hours=24)

    events = [
        _build_onset_event(e)
        for e in forecast["onset_events"] + forecast["offset_events"]
    ]
    events.sort(key=lambda e: e.datetime)
    return events
