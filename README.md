# Cork City Weather Forecasting System

Hyperlocal hourly weather forecasts for Cork City, powered by 60+ years of Met Éireann historical data (station 3904, Cork Airport). Built as a production-ready ML pipeline — not a notebook experiment.

## Goal

Produce Google Weather-style hourly forecasts:
- **Temperature** — actual and "feels like" (apparent temperature)
- **Rainfall** — probability, expected amount (mm), and onset/offset times
- **Wind** — speed and direction
- **Comfort score** — composite outdoor usability rating (0–10)
- **Clothing recommendations** — context-aware suggestions

All served via a local FastAPI JSON API.

## Data Source

[Met Éireann Open Data](https://www.met.ie/climate/available-data/historical-data) — station 3904 (Cork Airport).  
File: `data/raw/hly3904.csv` — 558,096 hourly rows from January 1962 to present. **Do not modify this file.**

## Architecture

Two-stage rainfall pipeline (mandatory for zero-inflated data):

```
Stage A: Binary classifier      → P(rain_occurred)
Stage B: Regressor (wet only)   → log1p(rainfall_mm)
Output:  P(rain) × expm1(Stage B)
```

Models: XGBoost + LightGBM ensemble (soft-vote classifier, inverse-MAE-weighted regressor).  
PyTorch (LSTM/TFT) is deferred unless ensemble F1 < 0.70 on rain occurrence.

## Quick Start

```bash
pip install -r requirements.txt
```

Copy the environment template:
```bash
cp .env.example .env
```

Run the data audit (produces plots + console report):
```bash
python scripts/audit.py
```

Run the full pipeline in order:
```bash
python scripts/audit.py          # Phase 1 — verify data
python -m src.data.loader        # Phase 1 — load raw CSV
python -m src.data.cleaner       # Phase 1 — clean → hourly_clean.parquet
python -m src.data.features      # Phase 1 — engineer features → hourly_features.parquet
python src/train.py              # Phase 2 — train XGBoost + LightGBM ensemble
python src/predict.py            # Phase 3 — generate 24h forecast
uvicorn api.main:app --reload    # Phase 5 — start API server
```

## Pipeline Steps

| Step | Script | Output |
|---|---|---|
| Data audit | `scripts/audit.py` | `results/plots/audit_*.png` |
| Clean data | `src/data/cleaner.py` | `data/processed/hourly_clean.parquet` |
| Feature engineering | `src/data/features.py` | `data/processed/hourly_features.parquet` |
| Train models | `src/train.py` | `models/*_latest.joblib`, `results/metrics.json` |
| Generate forecast | `src/predict.py` | Console / JSON output |
| Serve API | `api/main.py` | HTTP JSON API on `localhost:8000` |
| Fetch latest data | `scripts/fetch_latest.py` | Appends new rows to raw CSV (Phase 5 only) |
| Nightly retrain | `scripts/retrain.py` | Updated model artifacts |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/forecast/today` | Full 24-hour hourly forecast |
| GET | `/forecast/now` | Current conditions + next 3 hours |
| GET | `/forecast/windows` | Dry/wet windows with onset/offset times |
| GET | `/forecast/outfit` | Clothing recommendations |
| GET | `/forecast/commute` | Rain forecast for specific departure times |
| GET | `/health` | Model version, latency, feature count |

API documentation available at `http://localhost:8000/docs` (Swagger UI) when the server is running.

## Configuration

All constants live in `src/config.py`. Configure via `.env`:

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | `8000` | Uvicorn bind port |

## Project Structure

```
Weather-Predictor/
├── archive/              # Preserved old scripts — not used in production
├── data/
│   ├── raw/              # hly3904.csv — READ ONLY source of truth
│   └── processed/        # Parquet outputs from the data pipeline
├── src/
│   ├── config.py         # All constants, paths, hyperparameters
│   ├── data/             # loader, cleaner, features
│   ├── models/           # pipeline, baselines, xgb, lgbm, ensemble, onset_offset
│   ├── evaluate.py       # All metric functions
│   ├── train.py          # Training orchestrator
│   └── predict.py        # Inference: 72h input → 24h forecast
├── api/                  # FastAPI application
├── scripts/              # audit, fetch_latest, retrain
├── models/               # Serialised .joblib model artifacts
├── results/              # metrics.json + plots/
├── notebooks/            # Exploratory only — not production
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Model Performance Gates

| Metric | Minimum | Target |
|---|---|---|
| Rain occurrence F1 | 0.55 | **0.70** |
| Rainfall MAE (non-zero) | — | **< 1.20 mm** |

If F1 ≥ 0.70 and MAE < 1.20mm: gradient boosting v1 ships. PyTorch is optional.

## Key Data Facts (Audit, 2026-04-13)

- **Date range**: 1962-01-01 → 2025-09-01 (558,096 hourly rows, zero gaps)
- **Rainfall**: 82.81% exact zeros, 17.19% ≥ 0.1mm (wet). No values between 0 and 0.1mm.
- **Threshold**: 0.1mm inclusive (WMO standard for measurable precipitation)
- **Known issues fixed**: `rhum` clipped to [0, 100]; `wdsp` converted from knots to km/h