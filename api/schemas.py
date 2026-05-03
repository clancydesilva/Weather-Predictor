"""
api/schemas.py
──────────────
Pydantic v2 request/response models for the Cork City Weather API.

Every endpoint response is typed through one of these schemas.
FastAPI validates outbound data automatically — if the forecast logic
returns a field that doesn't match the schema, a 500 is raised at
development time (not silently passed to the client).

Schema hierarchy:
    HourlyForecast      — single forecast hour (temperature, rain, comfort...)
    OnsetEvent          — a predicted rain start/stop transition
    DailyForecast       — 24h of HourlyForecast + daily summary
    OutfitResponse      — clothing recommendation
    CommuteResponse     — rain/comfort snapshot for a specific departure time
    HealthResponse      — /health liveness check
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Per-hour ──────────────────────────────────────────────────────────────────

class ClothingDetail(BaseModel):
    items:      list[str] = Field(description="Ordered clothing items")
    confidence: str       = Field(description="e.g. '72% rain chance'")
    umbrella:   bool      = Field(description="True if umbrella recommended")
    waterproof: bool      = Field(description="True if waterproof jacket recommended instead")


class HourlyForecast(BaseModel):
    datetime:         str   = Field(description="ISO-8601 datetime string (UTC)")
    temp_c:           float = Field(description="Air temperature in Celsius")
    feels_like_c:     float = Field(description="Apparent temperature in Celsius")
    rain_probability: float = Field(ge=0, le=1, description="P(rain) 0–1")
    rain_flag:        int   = Field(ge=0, le=1, description="1 = rain predicted, 0 = dry")
    rainfall_mm:      float = Field(ge=0, description="Expected rainfall in mm")
    wind_speed_kmh:   float = Field(ge=0, description="Wind speed in km/h")
    wind_dir_deg:     int   = Field(ge=0, le=360, description="Wind direction in degrees")
    humidity_pct:     float = Field(ge=0, le=100)
    pressure_hpa:     float
    comfort_score:    float = Field(ge=0, le=10, description="Outdoor comfort 0–10")
    umbrella_risk:    bool  = Field(description="True if wind too strong for umbrella")
    clothing:         ClothingDetail


# ── Events ────────────────────────────────────────────────────────────────────

class OnsetEvent(BaseModel):
    event:      str   = Field(description="'onset' or 'offset'")
    datetime:   str   = Field(description="ISO-8601 datetime of predicted transition")
    confidence: float = Field(ge=0, le=1)
    message:    str   = Field(description="Human-readable summary, e.g. 'Rain expected to start around 14:00'")


# ── Daily summary ─────────────────────────────────────────────────────────────

class DailySummary(BaseModel):
    max_temp_c:            float
    min_temp_c:            float
    total_rainfall_mm:     float = Field(ge=0)
    peak_rain_probability: float = Field(ge=0, le=1)
    avg_comfort_score:     float = Field(ge=0, le=10)
    rain_hours:            int   = Field(ge=0)
    forecast_hours:        int   = Field(ge=0)


class DailyForecast(BaseModel):
    generated_at:  str              = Field(description="ISO-8601 timestamp of forecast generation")
    station:       str              = Field(description="Met station name")
    hours:         list[HourlyForecast]
    onset_events:  list[OnsetEvent]
    offset_events: list[OnsetEvent]
    daily_summary: DailySummary


# ── Outfit ────────────────────────────────────────────────────────────────────

class OutfitResponse(BaseModel):
    items:         list[str] = Field(description="Ordered clothing recommendations")
    confidence:    str       = Field(description="Human-readable rain probability summary")
    comfort_score: float     = Field(ge=0, le=10, description="Overall outdoor comfort score")
    umbrella_risk: bool      = Field(description="True if wind makes umbrella inadvisable")
    waterproof:    bool      = Field(description="True if waterproof jacket recommended over umbrella")


# ── Commute ───────────────────────────────────────────────────────────────────

class CommuteSlot(BaseModel):
    departure_time:   str
    temp_c:           float
    feels_like_c:     float
    rain_probability: float = Field(ge=0, le=1)
    rainfall_mm:      float = Field(ge=0)
    wind_speed_kmh:   float = Field(ge=0)
    comfort_score:    float = Field(ge=0, le=10)
    umbrella_risk:    bool
    recommendation:   str   = Field(description="One-line commute advice")


class CommuteResponse(BaseModel):
    morning: Optional[CommuteSlot] = None
    evening: Optional[CommuteSlot] = None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_version: str
    last_data_ts:  str   = Field(description="ISO-8601 timestamp of most recent observation in parquet")
    inference_ms:  float = Field(description="Dummy inference latency in milliseconds")
    feature_count: int
