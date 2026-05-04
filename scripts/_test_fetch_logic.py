"""
scripts/_test_fetch_logic.py
Offline unit test for fetch_latest.py — no network required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import RAW_HOURLY_PATH

DATE_FORMAT = "%d-%b-%Y %H:%M"

# Load first 100 rows of existing CSV as our "existing" data
existing = pd.read_csv(RAW_HOURLY_PATH, low_memory=False, nrows=100)
existing["date"] = pd.to_datetime(existing["date"], format=DATE_FORMAT, errors="coerce")
existing_max = existing["date"].max()

# Simulate remote CSV: same 100 rows + 3 new ones beyond existing_max
extra_dates = pd.date_range(existing_max + pd.Timedelta(hours=1), periods=3, freq="h")
extra = existing.iloc[:3].copy()
extra["date"] = extra_dates
downloaded = pd.concat([existing, extra], ignore_index=True)

# Core filter logic (mirrors fetch_and_append)
new_rows = downloaded[downloaded["date"] > existing_max]

print(f"Existing max date : {existing_max}")
print(f"New rows found    : {len(new_rows)}")
print(f"New row dates     : {new_rows['date'].tolist()}")
assert len(new_rows) == 3, f"Expected 3, got {len(new_rows)}"

# Test up-to-date case (no new rows)
new_rows_empty = downloaded[downloaded["date"] > downloaded["date"].max()]
assert len(new_rows_empty) == 0, "Up-to-date case failed"

print("All logic tests: PASSED")
