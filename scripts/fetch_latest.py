"""
scripts/fetch_latest.py
───────────────────────
Download the latest hourly observations from Met Éireann and append only
genuinely new rows to the local raw CSV (data/raw/hly3904.csv).

CRITICAL IMPLEMENTATION NOTE
─────────────────────────────
Met Éireann's data endpoint returns the COMPLETE historical CSV on every
request (~50 MB, 558,000+ rows going back to 1962).  It is NOT a delta feed.

Naive append would duplicate years of data.  The correct approach is:
  1. Read existing local CSV → record existing_max_date
  2. Download full remote CSV into memory (never write directly to disk)
  3. Parse dates in the downloaded data
  4. Filter: keep only rows where downloaded_date > existing_max_date
  5. If no new rows → return 0 (safe no-op)
  6. Append only those new rows (header=False) → return count

Usage
-----
    python scripts/fetch_latest.py               # fetch and append
    python scripts/fetch_latest.py --dry-run     # check without writing

Exit codes
----------
    0 — success (even if 0 new rows)
    1 — network or file error
"""

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MET_EIREANN_LIVE_URL, RAW_HOURLY_PATH

DATE_FORMAT = "%d-%b-%Y %H:%M"


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'date' column in-place, coercing unparseable values to NaT."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")
    return df


def fetch_and_append(dry_run: bool = False) -> int:
    """
    Download the full Met Éireann CSV, append only rows newer than the local
    maximum date, and return the number of new rows written.

    Parameters
    ----------
    dry_run : bool
        If True, calculate how many new rows exist but do NOT write to disk.

    Returns
    -------
    int : number of new rows appended (0 if up to date or dry_run)

    Raises
    ------
    FileNotFoundError  : if RAW_HOURLY_PATH does not exist
    requests.HTTPError : if the download fails
    """
    # ── Step 1: find the latest timestamp we already have ────────────────────
    if not RAW_HOURLY_PATH.exists():
        raise FileNotFoundError(
            f"Local CSV not found: {RAW_HOURLY_PATH}\n"
            "Run the full data pipeline first (src/data/ingest.py or equivalent)."
        )

    print(f"Reading local CSV: {RAW_HOURLY_PATH}")
    t0 = time.perf_counter()
    existing = pd.read_csv(RAW_HOURLY_PATH, low_memory=False)
    existing = _parse_dates(existing)

    existing_max_date = existing["date"].max()
    local_rows = len(existing)
    print(f"  Local rows     : {local_rows:,}")
    print(f"  Local max date : {existing_max_date}")
    del existing  # free memory — the file is large

    # ── Step 2: download full remote CSV into memory ──────────────────────────
    print(f"\nDownloading from Met Éireann...")
    print(f"  URL: {MET_EIREANN_LIVE_URL}")
    try:
        response = requests.get(MET_EIREANN_LIVE_URL, timeout=120)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("ERROR: Download timed out after 120s.")
        return -1
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP {e.response.status_code} from Met Éireann.")
        return -1
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Met Éireann. Check network.")
        return -1

    elapsed_dl = time.perf_counter() - t0
    size_mb = len(response.content) / (1024 * 1024)
    print(f"  Downloaded     : {size_mb:.1f} MB in {elapsed_dl:.1f}s")

    # ── Step 3: parse the downloaded data ─────────────────────────────────────
    downloaded = pd.read_csv(StringIO(response.text), low_memory=False)
    downloaded = _parse_dates(downloaded)

    remote_max_date = downloaded["date"].max()
    print(f"  Remote rows    : {len(downloaded):,}")
    print(f"  Remote max date: {remote_max_date}")

    # ── Step 4: filter to genuinely new rows ──────────────────────────────────
    new_rows = downloaded[downloaded["date"] > existing_max_date].copy()
    n_new = len(new_rows)

    if n_new == 0:
        print(f"\n✓ Already up to date — no new rows.")
        return 0

    print(f"\n  New rows found : {n_new}")
    print(f"  Date range     : {new_rows['date'].min()} → {new_rows['date'].max()}")

    if dry_run:
        print(f"\n[dry-run] Would append {n_new} rows — no files written.")
        return 0

    # ── Step 5: append new rows (header=False avoids duplicate header) ────────
    # Re-format date back to original string format for CSV consistency
    new_rows["date"] = new_rows["date"].dt.strftime(DATE_FORMAT).str.lower()
    new_rows.to_csv(RAW_HOURLY_PATH, mode="a", header=False, index=False)

    elapsed_total = time.perf_counter() - t0
    print(f"✓ Appended {n_new} new rows to {RAW_HOURLY_PATH.name}")
    print(f"  Total time: {elapsed_total:.1f}s")
    return n_new


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch latest Met Éireann observations and append to local CSV."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check for new rows but do not write to disk.",
    )
    args = parser.parse_args()

    try:
        n = fetch_and_append(dry_run=args.dry_run)
        if n < 0:
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
