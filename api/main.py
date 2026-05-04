"""
api/main.py
───────────
FastAPI application entry point for the Cork City Weather API.

Startup (lifespan handler):
  1. Load ensemble, onset, and offset models from models/ into app.state
  2. Load the feature parquet tail (last 200 rows) into app.state for inference
     — LOOKBACK_HOURS(72) + FORECAST_HOURS(24) = 96 rows max needed;
       200 gives a safe buffer and stays constant regardless of data growth.
  3. Record last data timestamp and feature count for /health

Shutdown:
  1. Clear model store from app.state

Usage
-----
    # Development:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

    # Via environment variables:
    API_HOST=0.0.0.0 API_PORT=8000 python -m uvicorn api.main:app

Endpoints:
    GET /            — redirect to /docs
    GET /health      — liveness check with model version and inference latency
    GET /forecast/*  — see api/router.py

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api import router as forecast_router
from api.schemas import HealthResponse
from src.config import (
    API_HOST,
    API_PORT,
    FEATURE_COLUMNS,
    FEATURES_PARQUET,
    FORECAST_HOURS,
    LOOKBACK_HOURS,
    MODELS_DIR,
    STATION_NAME,
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models and feature data into app.state on startup.
    Clears everything on shutdown.

    Models are loaded once and held in memory — inference latency stays
    under 10ms even for 24h forecasts.
    """
    print("Loading models...")
    t0 = time.perf_counter()

    ensemble_path = MODELS_DIR / "ensemble_latest.joblib"
    onset_path    = MODELS_DIR / "onset_classifier_latest.joblib"
    offset_path   = MODELS_DIR / "offset_classifier_latest.joblib"

    # Ensemble is required — crash early if missing
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"Ensemble model not found: {ensemble_path}\n"
            f"Run 'python -m src.train' first."
        )
    app.state.ensemble = joblib.load(ensemble_path)
    print(f"  ensemble   : {ensemble_path.name}")

    # Onset/offset are optional — disable /events gracefully if absent
    app.state.onset_clf  = None
    app.state.offset_clf = None
    app.state.events_available = False
    try:
        app.state.onset_clf  = joblib.load(onset_path)
        app.state.offset_clf = joblib.load(offset_path)
        app.state.events_available = True
        print(f"  onset      : {onset_path.name}")
        print(f"  offset     : {offset_path.name}")
    except FileNotFoundError:
        print("  onset/offset: NOT FOUND — /forecast/events disabled.")
        print("  Run 'python scripts/train_phase3.py' to enable it.")

    print("Loading feature parquet...")
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"Feature parquet not found: {FEATURES_PARQUET}\n"
            f"Run 'python -m src.data.features' first."
        )
    df = pd.read_parquet(FEATURES_PARQUET)

    # Cap to the last (LOOKBACK_HOURS + FORECAST_HOURS + buffer) rows only.
    # Inference never needs more than 96 rows (72h lookback + 24h forecast).
    # Using a fixed tail means memory is O(1) regardless of how much new data
    # fetch_latest.py appends over time — prevents the OOM growth bug.
    _inference_buffer = LOOKBACK_HOURS + FORECAST_HOURS + 104  # = 200 rows
    app.state.features_df    = df.tail(_inference_buffer)
    app.state.last_data_ts   = df.index.max()
    app.state.feature_count  = len(FEATURE_COLUMNS)
    app.state.model_version  = ensemble_path.stem  # e.g. "ensemble_latest"

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"API ready in {elapsed:.0f}ms — {len(app.state.features_df):,} inference rows loaded (capped).")

    yield

    # Shutdown cleanup
    del app.state.ensemble
    del app.state.onset_clf
    del app.state.offset_clf
    del app.state.features_df
    print("Models unloaded.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cork City Weather API",
    description=(
        "Hyperlocal hourly weather forecasts for Cork City, Ireland.\n\n"
        "Powered by 60+ years of Met Éireann observations from Cork Airport "
        "(station 3904). Uses a two-stage gradient boosting ensemble (XGBoost + "
        "LightGBM) for rain probability and amount, plus dedicated onset/offset "
        "classifiers for rain start/stop prediction.\n\n"
        "**Honest horizon**: ~3–6 hours. This is a statistical pattern-matching "
        "model, not NWP. It cannot know about storms forming over the Atlantic "
        "right now."
    ),
    version="1.0.0",
    contact={"name": "Cork Weather Predictor"},
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="API liveness check",
    description=(
        "Returns model version, timestamp of the most recent observation in the "
        "feature parquet, and a dummy inference latency measurement."
    ),
)
async def health() -> HealthResponse:
    """Liveness check — also measures a dummy inference round-trip."""
    from src.predict import get_forecast_window, generate_forecast

    t0 = time.perf_counter()
    try:
        window = get_forecast_window(app.state.features_df, n_hours=1)
        app.state.ensemble.predict(window)
    except Exception:
        pass
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    last_ts = app.state.last_data_ts
    last_ts_str = last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts)

    return HealthResponse(
        status="ok",
        model_version=app.state.model_version,
        last_data_ts=last_ts_str,
        inference_ms=inference_ms,
        feature_count=app.state.feature_count,
    )


# ── Admin ────────────────────────────────────────────────────────────────────

@app.post(
    "/admin/reload",
    tags=["System"],
    summary="Hot-reload feature parquet",
    description=(
        "Reloads the feature parquet from disk into memory without restarting "
        "the server. Call this after fetch_latest.py appends new observations "
        "so the API serves fresh predictions immediately."
    ),
)
async def reload_features() -> dict:
    """
    Hot-reload the features parquet from disk.

    Called automatically by the fetch scheduler after each successful fetch.
    Safe to call at any time — reads are atomic at the DataFrame level.
    """
    if not FEATURES_PARQUET.exists():
        return {"status": "error", "detail": "Feature parquet not found on disk."}

    t0 = time.perf_counter()
    df = pd.read_parquet(FEATURES_PARQUET)
    _inference_buffer = LOOKBACK_HOURS + FORECAST_HOURS + 104
    app.state.features_df  = df.tail(_inference_buffer)
    app.state.last_data_ts = df.index.max()
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    last_ts = app.state.last_data_ts
    last_ts_str = last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts)
    print(f"[reload] Features refreshed in {elapsed_ms}ms — last obs: {last_ts_str}")

    return {
        "status":   "ok",
        "rows":     len(app.state.features_df),
        "last_ts":  last_ts_str,
        "elapsed_ms": elapsed_ms,
    }


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(forecast_router.router)


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
