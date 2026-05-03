"""
api/main.py
───────────
FastAPI application entry point for the Cork City Weather API.

Startup (lifespan handler):
  1. Load ensemble, onset, and offset models from models/ into app.state
  2. Load the feature parquet into app.state (in-memory for fast inference)
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
    MODELS_DIR,
    STATION_NAME,
    TEST_START_DATE,
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

    for path in (ensemble_path, onset_path, offset_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run 'python -m src.train' and 'python scripts/train_phase3.py' first."
            )

    app.state.ensemble   = joblib.load(ensemble_path)
    app.state.onset_clf  = joblib.load(onset_path)
    app.state.offset_clf = joblib.load(offset_path)

    print(f"  ensemble   : {ensemble_path.name}")
    print(f"  onset      : {onset_path.name}")
    print(f"  offset     : {offset_path.name}")

    print("Loading feature parquet...")
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"Feature parquet not found: {FEATURES_PARQUET}\n"
            f"Run 'python -m src.data.features' first."
        )
    df = pd.read_parquet(FEATURES_PARQUET)

    # Keep only the test-set window for inference (last ~4 years, fast to slice)
    # In production (Phase 5) this will be the live-updated parquet.
    app.state.features_df    = df.loc[TEST_START_DATE:]
    app.state.last_data_ts   = df.index.max()
    app.state.feature_count  = len(FEATURE_COLUMNS)
    app.state.model_version  = ensemble_path.stem  # e.g. "ensemble_latest"

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"API ready in {elapsed:.0f}ms — {len(app.state.features_df):,} inference rows loaded.")

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
