"""
src/data/loader.py
──────────────────
Load hly3904.csv with correct dtype handling, rename columns to readable names,
and validate schema.

Public API:
    load_raw(path)       → pd.DataFrame (indexed by datetime)
    validate_schema(df)  → None  (raises ValueError on missing columns)
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_HOURLY_PATH

# ── Column mapping: raw Met Éireann name → readable name ─────────────────────
# Columns not listed here are dropped (indicator flags, sparse fields).
COLUMN_MAP: dict[str, str] = {
    "date":  "datetime",
    "rain":  "rainfall_mm",
    "temp":  "temp_c",
    "rhum":  "humidity_pct",
    "msl":   "pressure_hpa",
    "wdsp":  "wind_speed_knots",   # converted to km/h in cleaner.py
    "wddir": "wind_dir_deg",
    "dewpt": "dewpoint_c",
    "wetb":  "wetbulb_c",
    "clamt": "cloud_cover_oktas",
    "vappr": "vapour_pressure_hpa",
}

# Columns to force through pd.to_numeric (mixed-type warning sources from audit)
NUMERIC_COERCE: list[str] = [
    "rain", "temp", "rhum", "msl", "wdsp", "wddir",
    "dewpt", "wetb", "clamt", "vappr",
]

# Schema that must be present after loading (validated by validate_schema)
REQUIRED_COLUMNS: list[str] = [
    "rainfall_mm",
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_knots",
    "wind_dir_deg",
    "dewpoint_c",
]


def load_raw(path: Path = RAW_HOURLY_PATH) -> pd.DataFrame:
    """
    Load hly3904.csv, coerce dtypes, rename columns, sort by datetime.

    Returns a DataFrame indexed by datetime (UTC-naive, hourly) with
    readable column names. Indicator columns and sparse fields are dropped.

    Raises
    ------
    FileNotFoundError  if path does not exist.
    ValueError         if required columns are missing after loading.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Download hly3904.csv from Met Éireann and place it in data/raw/."
        )

    # low_memory=False suppresses DtypeWarning on mixed-type columns.
    # dtype={'rain': str} then numeric coerce handles rows with empty strings.
    df = pd.read_csv(path, low_memory=False, dtype={"rain": str})

    # Coerce all numeric columns — empty strings / stray text → NaN
    for col in NUMERIC_COERCE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse datetime column
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M", errors="coerce")

    # Keep only columns we have a mapping for; drop everything else
    cols_present = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(cols_present.keys())].rename(columns=cols_present)

    # Sort ascending and set datetime index
    df = df.sort_values("datetime").set_index("datetime")
    df.index.name = "datetime"

    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Raise ValueError if any required column is missing from df.

    Call this after load_raw() and again after clean_pipeline() to catch
    any accidental column drops early.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing columns: {missing}\n"
            f"Present columns: {list(df.columns)}"
        )


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} rows  |  {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Null counts:\n{df.isnull().sum()}")
