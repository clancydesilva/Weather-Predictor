"""
src/data/features.py
────────────────────
Build the complete feature matrix that all models will train on.

Input:  hourly_clean.parquet (output of cleaner.clean_pipeline)
Output: hourly_features.parquet  (~550k rows × 50+ columns, no NaN)

Feature groups:
  A. Cyclic time encodings      — hour, month, day-of-week, day-of-year
  B. Wind component decomposition — u/v vectors (meteorological convention)
  C. Rainfall lag features      — 1,2,3,6,12,24h lags
  D. Pressure lag features      — 1,3,6h lags (pressure drops precede rain)
  E. Rolling statistics         — mean/std over 3,6,24h windows (shifted to avoid leakage)
  F. Pressure tendency          — 1,3,6,12h differences (Atlantic front predictor)
  G. Dew point depression       — temp - dewpoint (saturation proximity)
  H. Onset / offset labels      — for Phase 3 onset/offset classifier
  I. Drop NaN rows              — from lag/rolling lookback windows

Public API:
    build_features(df) -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    FEATURES_PARQUET,
    CLEAN_PARQUET,
    LAG_HOURS,
    ROLLING_WINDOWS,
    FEATURE_COLUMNS,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete feature matrix from a clean DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame produced by cleaner.clean_pipeline().
        Must be indexed by datetime in ascending order.

    Returns
    -------
    pd.DataFrame
        Feature matrix containing all FEATURE_COLUMNS plus target columns
        (rain_occurred, rainfall_log1p, rain_onset, rain_offset).
        NaN rows from lag/rolling windows are dropped.
        Also saved to FEATURES_PARQUET.
    """
    df = df.copy()

    # ── Group A: Cyclic time encodings ───────────────────────────────────────
    # Cyclic encoding preserves the circular relationship between time values
    # (e.g. hour 23 is adjacent to hour 0, December is adjacent to January).
    # Raw integer encoding (hour=0..23) falsely implies 23 is far from 0.

    # Hour of day (0–23) — daily cycle
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    # Month (1–12) — annual seasonality
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Day of week (0=Monday … 6=Sunday)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Day of year (1–365) — captures summer/winter temperature pattern
    df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    # ── Group B: Wind component decomposition ────────────────────────────────
    # Wind direction + speed → u (east-west) and v (north-south) components.
    # Meteorological convention: u > 0 = wind blowing eastward,
    #                            v > 0 = wind blowing northward.
    # A westerly (270°) with 50 km/h is a completely different weather signal
    # than an easterly (90°) — decomposition makes this learnable by gradient
    # boosting trees which cannot otherwise understand circular direction.
    wind_dir_rad = np.deg2rad(df["wind_dir_deg"].fillna(0))
    df["u_wind"] = -df["wind_speed_kmh"] * np.sin(wind_dir_rad)
    df["v_wind"] = -df["wind_speed_kmh"] * np.cos(wind_dir_rad)

    # ── Group C: Rainfall lag features ───────────────────────────────────────
    # Lag features give the model memory of recent conditions.
    # These are the strongest predictors for rain continuation (persistence).
    for lag in LAG_HOURS:  # [1, 2, 3, 6, 12, 24] from config
        df[f"rainfall_lag_{lag}h"] = df["rainfall_mm"].shift(lag)
        df[f"rain_occurred_lag_{lag}h"] = df["rain_occurred"].shift(lag)

    # ── Group D: Pressure lag features ───────────────────────────────────────
    # Pressure drops in the hours before rain from approaching Atlantic fronts.
    for lag in [1, 3, 6]:
        df[f"pressure_lag_{lag}h"] = df["pressure_hpa"].shift(lag)

    # ── Group E: Rolling statistics ───────────────────────────────────────────
    # Rolling windows capture trends rather than point values.
    # CRITICAL: all rolling windows use .shift(1) before computing to prevent
    # data leakage. Without shift(1), the window at time T includes the value
    # at T itself — the model would be seeing the current hour's data when
    # "predicting" it, which inflates training accuracy but fails at inference.
    for window in ROLLING_WINDOWS:  # [3, 6, 24] from config
        # Rainfall rolling mean and std
        shifted_rain = df["rainfall_mm"].shift(1)
        df[f"rainfall_roll_mean_{window}h"] = shifted_rain.rolling(window).mean()
        df[f"rainfall_roll_std_{window}h"] = shifted_rain.rolling(window).std()

        # Pressure rolling mean and std
        shifted_pressure = df["pressure_hpa"].shift(1)
        df[f"pressure_roll_mean_{window}h"] = shifted_pressure.rolling(window).mean()
        df[f"pressure_roll_std_{window}h"] = shifted_pressure.rolling(window).std()

        # Humidity rolling mean
        shifted_humidity = df["humidity_pct"].shift(1)
        df[f"humidity_roll_mean_{window}h"] = shifted_humidity.rolling(window).mean()

    # ── Group F: Pressure tendency ───────────────────────────────────────────
    # Pressure tendency is the change in pressure over a given time window.
    # A drop of > 1 hPa/3h signals an approaching low-pressure system.
    # A rise (positive tendency) signals clearing conditions.
    # These are the strongest physical predictors of incoming Atlantic weather.
    df["pressure_tendency_1h"] = df["pressure_hpa"] - df["pressure_hpa"].shift(1)
    df["pressure_tendency_3h"] = df["pressure_hpa"] - df["pressure_hpa"].shift(3)
    df["pressure_tendency_6h"] = df["pressure_hpa"] - df["pressure_hpa"].shift(6)
    df["pressure_tendency_12h"] = df["pressure_hpa"] - df["pressure_hpa"].shift(12)

    # ── Group G: Dew point depression ────────────────────────────────────────
    # Dew point depression = temp - dewpoint. When it approaches 0°C, the air
    # is near saturation and rain is imminent. Strong physical signal.
    df["dewpoint_depression"] = df["temp_c"] - df["dewpoint_c"]

    # ── Group H: Onset / offset event labels (Phase 3 targets) ───────────────
    # rain_onset:  current hour is wet, previous hour was dry  → start of rain
    # rain_offset: current hour is dry, previous hour was wet  → end of rain
    # These are used to train the onset/offset classifiers in Phase 3.
    df["rain_onset"] = (
        (df["rain_occurred"] == 1) & (df["rain_occurred"].shift(1) == 0)
    ).astype(np.int8)

    df["rain_offset"] = (
        (df["rain_occurred"] == 0) & (df["rain_occurred"].shift(1) == 1)
    ).astype(np.int8)

    # ── Group I: Drop NaN rows from lag/rolling lookback windows ─────────────
    # The maximum lag is 24h, so the first 24 rows will have NaN lag values.
    # Rolling windows of up to 24h add a further window-1 NaN rows.
    # Total rows lost is at most ~48 out of 558,096 — completely negligible.
    rows_before = len(df)
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    rows_dropped = rows_before - len(df)
    print(f"Dropped {rows_dropped} rows with NaN from lag/rolling lookback windows.")

    # ── Validate all required feature columns are present ────────────────────
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected feature columns after build_features: {missing}"
        )

    # ── Save to parquet ───────────────────────────────────────────────────────
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET, engine="pyarrow", compression="snappy")
    print(f"Saved features parquet: {FEATURES_PARQUET}  ({len(df):,} rows, {df.shape[1]} columns)")

    return df


if __name__ == "__main__":
    print("Loading clean parquet...")
    df_clean = pd.read_parquet(CLEAN_PARQUET)

    print("Building features...")
    df_feat = build_features(df_clean)

    print(f"\nFeature matrix shape: {df_feat.shape}")
    print(f"Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"\nOnset events:  {df_feat['rain_onset'].sum():,}  "
          f"({100 * df_feat['rain_onset'].mean():.3f}%)")
    print(f"Offset events: {df_feat['rain_offset'].sum():,}  "
          f"({100 * df_feat['rain_offset'].mean():.3f}%)")
    print(f"\nNull counts (must all be 0 for feature columns):")
    null_counts = df_feat[FEATURE_COLUMNS].isnull().sum()
    print(null_counts[null_counts > 0] if null_counts.any() else "  None — all clean.")
