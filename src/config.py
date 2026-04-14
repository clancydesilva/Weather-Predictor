"""
src/config.py
─────────────
Single source of truth for every constant, path, threshold, and hyperparameter
in the Cork City Weather Forecasting System.

ALL other modules import from here. Never define paths, split dates, thresholds,
or model hyperparameters as literals in any other file.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR      = ROOT / "models"
RESULTS_DIR     = ROOT / "results"
PLOTS_DIR       = RESULTS_DIR / "plots"

RAW_HOURLY_PATH     = DATA_RAW / "hly3904.csv"
CLEAN_PARQUET       = DATA_PROCESSED / "hourly_clean.parquet"
FEATURES_PARQUET    = DATA_PROCESSED / "hourly_features.parquet"
METRICS_JSON        = RESULTS_DIR / "metrics.json"

# ── Station metadata ──────────────────────────────────────────────────────────
STATION_ID      = 3904
STATION_NAME    = "Cork Airport"
STATION_LAT     = 51.8408
STATION_LON     = -8.4894

# ── Time-based split dates ────────────────────────────────────────────────────
# Train:      all data up to and including this datetime
TRAIN_END_DATE  = "2015-12-31 23:00"
# Validation: 2016-01-01 00:00 → 2020-12-31 23:00
VAL_END_DATE    = "2020-12-31 23:00"
# Test:       2021-01-01 00:00 → end of dataset
TEST_START_DATE = "2021-01-01 00:00"

# ── Rainfall thresholds ───────────────────────────────────────────────────────
# Audit ground truth (re-verified 2026-04-13):
#   - 460,388 / 555,937 non-null rows = 82.81% exact zeros
#   - 25,534 rows exactly == 0.1mm (4.59%) — these ARE rain events (WMO standard)
#   - 95,549 rows >= 0.1mm (17.19%) — total wet hours
#   - Zero rows with 0 < rain < 0.1mm — threshold is clean with no ambiguous values
RAIN_OCCURRENCE_THRESHOLD = 0.1    # mm — inclusive (>= 0.1 = rain occurred)
RAIN_CAP_MM               = 6.4    # 99.9th percentile from audit — cap before scaling

# ── Feature engineering ───────────────────────────────────────────────────────
LAG_HOURS       = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 24]           # hours
LOOKBACK_HOURS  = 72                   # sequence model context window (future use)
FORECAST_HOURS  = 24                   # prediction horizon

# ── Feature columns (explicit list — the exact set fed to every model) ────────
FEATURE_COLUMNS = [
    # Raw meteorological
    "temp_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh",
    "dewpoint_c", "wetbulb_c", "cloud_cover_oktas", "vapour_pressure_hpa",
    # Cyclic time encodings
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    # Wind components (meteorological u/v decomposition)
    "u_wind", "v_wind",
    # Rainfall lags (most predictive for rain continuation)
    "rainfall_lag_1h", "rainfall_lag_2h", "rainfall_lag_3h",
    "rainfall_lag_6h", "rainfall_lag_12h", "rainfall_lag_24h",
    "rain_occurred_lag_1h", "rain_occurred_lag_3h", "rain_occurred_lag_6h",
    # Pressure lags (pressure drops precede rain)
    "pressure_lag_1h", "pressure_lag_3h", "pressure_lag_6h",
    # Rolling rainfall statistics
    "rainfall_roll_mean_3h", "rainfall_roll_std_3h",
    "rainfall_roll_mean_6h", "rainfall_roll_std_6h",
    "rainfall_roll_mean_24h", "rainfall_roll_std_24h",
    # Rolling pressure statistics
    "pressure_roll_mean_3h", "pressure_roll_std_3h",
    "pressure_roll_mean_6h", "pressure_roll_std_6h",
    # Rolling humidity
    "humidity_roll_mean_3h", "humidity_roll_mean_6h",
    # Pressure tendency (Atlantic front predictor)
    "pressure_tendency_1h", "pressure_tendency_3h",
    "pressure_tendency_6h", "pressure_tendency_12h",
    # Derived meteorological
    "dewpoint_depression",
]

# Target columns — separated explicitly so they are never accidentally used as features
TARGET_BINARY     = "rain_occurred"     # classifier target (int8: 0 or 1)
TARGET_REGRESSION = "rainfall_log1p"   # regressor target (log1p-transformed mm)
TARGET_TEMP       = "temp_c"           # temperature regression target

# ── Model hyperparameters (defaults — overrideable by train.py sweep) ─────────
#
# scale_pos_weight formula (XGBoost): n_negative_class / n_positive_class
#   Using >= 0.1mm threshold: negatives = 460,388 (dry), positives = 95,549 (wet)
#   460,388 / 95,549 = 4.82
#   This is NOT 82.8 / 17.2 (fraction arithmetic) — it is raw count arithmetic.
XGB_SCALE_POS_WEIGHT = 4.82
LGB_IS_UNBALANCE     = True   # LightGBM equivalent of scale_pos_weight

# ── Metric thresholds (product gates) ──────────────────────────────────────────
MIN_F1_RAIN         = 0.55   # below this → investigate class imbalance before continuing
TARGET_F1_RAIN      = 0.70   # at this → gradient boosting v1 ships; PyTorch becomes optional
TARGET_MAE_RAIN_MM  = 1.20   # MAE on non-zero inverse-transformed rainfall (mm)

# ── API configuration ─────────────────────────────────────────────────────────
# Live URL intentionally NOT used until Phase 5 (fetch_latest.py).
# For Phases 1–4, work exclusively from local RAW_HOURLY_PATH.
MET_EIREANN_LIVE_URL = "https://cli.fusio.net/cli/climate_data/webdata/hly3904.csv"
API_HOST             = os.getenv("API_HOST", "0.0.0.0")
API_PORT             = int(os.getenv("API_PORT", "8000"))
MODEL_VERSION_FILE   = MODELS_DIR / "current_version.txt"
MAX_INFERENCE_MS     = 200   # API must respond under this threshold (milliseconds)
