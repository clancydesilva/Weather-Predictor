"""
src/data/cleaner.py
───────────────────
Apply all data quality fixes in a reproducible, ordered pipeline.

Input:  raw DataFrame from loader.load_raw()
Output: clean DataFrame saved to data/processed/hourly_clean.parquet

Cleaning steps (applied in this exact order — do not reorder):
  1. Clip humidity to [0, 100]       — audit found values up to 760 (sensor error)
  2. Convert wind speed knots→km/h   — raw column is in knots
  3. Forward-fill missing rainfall   — 2,159 isolated null rows, ffill is safe
  4. Cap rainfall at 99.9th pct      — 6.4mm from config, prevents regression skew
  5. Create binary rain_occurred     — >= 0.1mm threshold (WMO standard)
  6. Create log1p rainfall target    — regressor trains on this, not raw mm
  7. Assert data integrity           — fails loudly if any invariant is violated
  8. Save to parquet                 — snappy-compressed, dtype-preserving

Public API:
    clean_pipeline(df) -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_PARQUET,
    DATA_PROCESSED,
    RAIN_CAP_MM,
    RAIN_OCCURRENCE_THRESHOLD,
)
from src.data.loader import validate_schema


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps in order. Returns a clean DataFrame.
    Also writes the result to CLEAN_PARQUET.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame produced by loader.load_raw(). Must be indexed by datetime.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with wind_speed_kmh replacing wind_speed_knots,
        rain_occurred and rainfall_log1p target columns added.
    """
    df = df.copy()

    # ── Step 1: Fix humidity sensor errors ───────────────────────────────────
    # Audit found rhum values up to 760% — physical maximum is 100%.
    df["humidity_pct"] = df["humidity_pct"].clip(lower=0, upper=100)

    # ── Step 2: Convert wind speed from knots to km/h ────────────────────────
    # All downstream code uses wind_speed_kmh. The knots column is dropped.
    df["wind_speed_kmh"] = df["wind_speed_knots"] * 1.852
    df.drop(columns=["wind_speed_knots"], inplace=True)

    # ── Step 3: Forward-fill missing rainfall ────────────────────────────────
    # 2,159 rows (0.39%) have null rainfall. Isolated hour-gaps throughout the
    # record. ffill copies the previous valid reading forward. bfill handles
    # any leading nulls at the very start of the 1962 series where ffill has
    # no prior value to copy from.
    df["rainfall_mm"] = df["rainfall_mm"].ffill().bfill()

    # ── Step 4: Cap extreme rainfall at 99.9th percentile ────────────────────
    # Max raw value is 22.6mm/hr. Capping at 6.4mm (RAIN_CAP_MM from config)
    # prevents a handful of extreme-event hours from dominating regression loss.
    df["rainfall_mm"] = df["rainfall_mm"].clip(upper=RAIN_CAP_MM)

    # ── Step 5: Create binary rain occurrence label ───────────────────────────
    # >= 0.1mm is the WMO lower bound for measurable precipitation.
    # Audit confirmed: no values exist between 0 and 0.1mm — threshold is clean.
    df["rain_occurred"] = (
        df["rainfall_mm"] >= RAIN_OCCURRENCE_THRESHOLD
    ).astype(np.int8)

    # ── Step 6: Apply log1p transform to rainfall ─────────────────────────────
    # This is the regressor target. Right-skewed rainfall becomes approximately
    # normal. Always apply np.expm1() when converting predictions back to mm.
    df["rainfall_log1p"] = np.log1p(df["rainfall_mm"])

    # ── Step 7: Assert data integrity ────────────────────────────────────────
    _assert_clean(df)

    # ── Step 8: Save to parquet ───────────────────────────────────────────────
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, engine="pyarrow", compression="snappy")
    print(f"Saved clean parquet: {CLEAN_PARQUET}  ({len(df):,} rows)")

    return df


def _assert_clean(df: pd.DataFrame) -> None:
    """
    Internal guard. Raises ValueError with a descriptive message if any
    post-cleaning invariant is violated. Do not call from outside this module.
    """
    if df["humidity_pct"].max() > 100:
        raise ValueError(
            f"humidity_pct max is {df['humidity_pct'].max()} after clipping — "
            "Step 1 did not apply correctly."
        )

    if df["rainfall_mm"].isnull().sum() > 0:
        raise ValueError(
            f"{df['rainfall_mm'].isnull().sum()} null rainfall_mm rows remain "
            "after forward-fill. Check for leading nulls at the start of the series."
        )

    if df["rainfall_mm"].max() > RAIN_CAP_MM + 0.001:
        raise ValueError(
            f"rainfall_mm max is {df['rainfall_mm'].max():.3f} — exceeds cap of "
            f"{RAIN_CAP_MM}mm. Step 4 did not apply correctly."
        )

    if "wind_speed_kmh" not in df.columns:
        raise ValueError(
            "wind_speed_kmh column missing — knots conversion (Step 2) failed."
        )

    if "wind_speed_knots" in df.columns:
        raise ValueError(
            "wind_speed_knots still present — it should have been dropped in Step 2."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError("datetime index is not sorted ascending.")

    if df.index.duplicated().sum() > 0:
        n = df.index.duplicated().sum()
        raise ValueError(f"{n} duplicate datetime index entries found.")


if __name__ == "__main__":
    from src.data.loader import load_raw

    raw = load_raw()
    clean = clean_pipeline(raw)

    print(f"\nRows:    {len(clean):,}")
    print(f"Columns: {list(clean.columns)}")
    print(f"\nRain occurred (1=wet): {clean['rain_occurred'].sum():,} "
          f"({100 * clean['rain_occurred'].mean():.2f}%)")
    print(f"Dry hours (0):         {(clean['rain_occurred'] == 0).sum():,} "
          f"({100 * (clean['rain_occurred'] == 0).mean():.2f}%)")
    print(f"\nHumidity max: {clean['humidity_pct'].max():.1f}%  (must be <= 100)")
    print(f"Rainfall max: {clean['rainfall_mm'].max():.2f}mm  (must be <= {RAIN_CAP_MM})")
    print(f"Wind max:     {clean['wind_speed_kmh'].max():.1f} km/h")
    print(f"Null counts:\n{clean.isnull().sum()}")
