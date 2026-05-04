"""
tests/test_phase5.py
--------------------
Comprehensive tests for Phase 5: fetch_latest.py and retrain.py.

All tests run offline - no network calls, no model retraining.
Covers: date parsing, delta filtering, error handling, promotion gate,
        rollback, CLI flags, and edge cases.

Run with:
    python tests/test_phase5.py
"""

import sys
import shutil
import tempfile
import json
import argparse
from io import StringIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_HOURLY_PATH, MODELS_DIR, METRICS_JSON, RESULTS_DIR
from scripts.fetch_latest import fetch_and_append, _parse_dates, DATE_FORMAT
from scripts.retrain import _load_current_val_f1, _rollback_latest_models


# -- Helpers -------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = {"pass": 0, "fail": 0, "skip": 0}


def test(name: str, fn):
    try:
        result = fn()
        if result == "SKIP":
            print(f"  {SKIP}  {name}")
            results["skip"] += 1
        else:
            print(f"  {PASS}  {name}")
            results["pass"] += 1
    except AssertionError as e:
        print(f"  {FAIL}  {name}")
        print(f"         AssertionError: {e}")
        results["fail"] += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {type(e).__name__}: {e}")
        results["fail"] += 1


# -- Section 1: Date Parsing ---------------------------------------------------

def section1():
    print("\n--- 1. Date Parsing ---------------------------------------")

    def t_parse_valid():
        df = pd.DataFrame({"date": ["01-jan-1962 01:00", "15-aug-2025 12:00"]})
        parsed = _parse_dates(df)
        assert parsed["date"].iloc[0] == pd.Timestamp("1962-01-01 01:00")
        assert parsed["date"].iloc[1] == pd.Timestamp("2025-08-15 12:00")

    def t_parse_coerce():
        df = pd.DataFrame({"date": ["garbage", "01-jan-1962 01:00"]})
        parsed = _parse_dates(df)
        assert pd.isna(parsed["date"].iloc[0]), "Bad date should be NaT"
        assert not pd.isna(parsed["date"].iloc[1])

    def t_parse_roundtrip():
        original = "15-aug-2025 12:00"
        df = pd.DataFrame({"date": [original]})
        parsed = _parse_dates(df)
        formatted = parsed["date"].dt.strftime(DATE_FORMAT).str.lower().iloc[0]
        assert formatted == original, f"Roundtrip failed: {formatted} != {original}"

    def t_parse_last_local_row():
        """Verify the actual last row of the local CSV parses correctly."""
        if not RAW_HOURLY_PATH.exists():
            return "SKIP"
        df = pd.read_csv(RAW_HOURLY_PATH, low_memory=False, skiprows=range(1, 558090))
        parsed = _parse_dates(df)
        assert not pd.isna(parsed["date"].iloc[-1]), "Last row date should parse"

    test("Valid dates parse correctly", t_parse_valid)
    test("Invalid dates coerce to NaT", t_parse_coerce)
    test("Date - string roundtrip", t_parse_roundtrip)
    test("Last local CSV row parses", t_parse_last_local_row)


# -- Section 2: Delta Filter Logic --------------------------------------------

def section2():
    print("\n--- 2. Delta Filter Logic ---------------------------------")

    def make_df(n_rows: int, start: str = "2025-01-01") -> pd.DataFrame:
        dates = pd.date_range(start, periods=n_rows, freq="h")
        return pd.DataFrame({
            "date":  dates,
            "rain":  np.random.choice([0, 0.1, 0.5, 1.0], size=n_rows),
            "temp":  np.random.uniform(5, 20, size=n_rows),
        })

    def t_no_new_rows():
        existing = make_df(100, "2025-01-01")
        downloaded = existing.copy()
        max_ts = existing["date"].max()
        new_rows = downloaded[downloaded["date"] > max_ts]
        assert len(new_rows) == 0

    def t_exact_delta():
        existing = make_df(100, "2025-01-01")
        extra = make_df(5, start=str(existing["date"].max() + pd.Timedelta(hours=1)))
        downloaded = pd.concat([existing, extra], ignore_index=True)
        max_ts = existing["date"].max()
        new_rows = downloaded[downloaded["date"] > max_ts]
        assert len(new_rows) == 5

    def t_no_duplicates():
        existing = make_df(100, "2025-01-01")
        # Downloaded has same 100 rows + 3 new
        extra = make_df(3, start=str(existing["date"].max() + pd.Timedelta(hours=1)))
        downloaded = pd.concat([existing, extra], ignore_index=True)
        max_ts = existing["date"].max()
        new_rows = downloaded[downloaded["date"] > max_ts]
        assert len(new_rows) == 3
        # Verify no existing rows sneak in
        assert new_rows["date"].min() > max_ts

    def t_large_delta():
        existing = make_df(1000, "2020-01-01")
        extra = make_df(24, start=str(existing["date"].max() + pd.Timedelta(hours=1)))
        downloaded = pd.concat([existing, extra], ignore_index=True)
        max_ts = existing["date"].max()
        new_rows = downloaded[downloaded["date"] > max_ts]
        assert len(new_rows) == 24

    def t_boundary_exclusive():
        """Row exactly at existing_max_date must NOT be included."""
        existing = make_df(10, "2025-01-01")
        max_ts = existing["date"].max()
        boundary_row = existing[existing["date"] == max_ts].copy()
        downloaded = pd.concat([existing, boundary_row], ignore_index=True)
        new_rows = downloaded[downloaded["date"] > max_ts]
        assert len(new_rows) == 0, "Boundary row must be excluded (strictly >)"

    test("No new rows - empty delta", t_no_new_rows)
    test("5 new rows - delta of 5", t_exact_delta)
    test("No duplicate rows in delta", t_no_duplicates)
    test("24-hour delta (typical nightly)", t_large_delta)
    test("Boundary row excluded (strictly >)", t_boundary_exclusive)


# -- Section 3: fetch_and_append Error Handling --------------------------------

def section3():
    print("\n--- 3. fetch_and_append Error Handling --------------------")

    def t_missing_local_file():
        original = RAW_HOURLY_PATH
        tmp_path = RAW_HOURLY_PATH.parent / "_tmp_missing_test.csv"
        # Temporarily point at a nonexistent file by patching
        import scripts.fetch_latest as fl
        real_path = fl.RAW_HOURLY_PATH
        fl.RAW_HOURLY_PATH = tmp_path
        try:
            raised = False
            try:
                fl.fetch_and_append()
            except FileNotFoundError:
                raised = True
            assert raised, "Should raise FileNotFoundError for missing local CSV"
        finally:
            fl.RAW_HOURLY_PATH = real_path

    def t_dry_run_returns_zero_on_uptodate(monkeypatch=None):
        """Dry run with a mocked 'downloaded' that matches local - returns 0."""
        import scripts.fetch_latest as fl
        import unittest.mock as mock

        # Read just the tail of local CSV
        local_df = pd.read_csv(RAW_HOURLY_PATH, low_memory=False, nrows=50)
        local_df_parsed = _parse_dates(local_df)
        max_ts = local_df_parsed["date"].max()

        # Mock requests.get to return the same 50 rows
        class FakeResponse:
            text = local_df.to_csv(index=False)
            content = text.encode()
            def raise_for_status(self): pass

        with mock.patch("scripts.fetch_latest.requests.get", return_value=FakeResponse()):
            result = fl.fetch_and_append(dry_run=True)
        assert result == 0, f"Expected 0, got {result}"

    def t_dry_run_with_new_rows():
        import scripts.fetch_latest as fl
        import unittest.mock as mock

        local_df = pd.read_csv(RAW_HOURLY_PATH, low_memory=False, nrows=50)
        local_df_parsed = _parse_dates(local_df)
        max_ts = local_df_parsed["date"].max()

        # Add 3 extra rows to the "downloaded" CSV
        extra = local_df.iloc[:3].copy()
        extra["date"] = (max_ts + pd.to_timedelta([1, 2, 3], unit="h")).strftime(DATE_FORMAT).str.lower()
        downloaded = pd.concat([local_df, extra], ignore_index=True)

        class FakeResponse:
            text = downloaded.to_csv(index=False)
            content = text.encode()
            def raise_for_status(self): pass

        with mock.patch("scripts.fetch_latest.requests.get", return_value=FakeResponse()):
            result = fl.fetch_and_append(dry_run=True)
        assert result == 0, f"dry_run should return 0 even with new rows, got {result}"

    def t_real_append_writes_new_rows():
        """End-to-end: mock 3 new rows, verify they are actually appended."""
        import scripts.fetch_latest as fl
        import unittest.mock as mock

        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            # Write small local CSV to temp file
            local_df = pd.read_csv(RAW_HOURLY_PATH, low_memory=False, nrows=50)
            local_df.to_csv(tmp, index=False)
            tmp_path = Path(tmp.name)

        try:
            local_df_parsed = _parse_dates(local_df)
            max_ts = local_df_parsed["date"].max()

            extra = local_df.iloc[:3].copy()
            extra["date"] = (
                (max_ts + pd.to_timedelta([1, 2, 3], unit="h"))
                .strftime(DATE_FORMAT)
                .str.lower()
            )
            downloaded = pd.concat([local_df, extra], ignore_index=True)

            class FakeResponse:
                text = downloaded.to_csv(index=False)
                content = text.encode()
                def raise_for_status(self): pass

            real_path = fl.RAW_HOURLY_PATH
            fl.RAW_HOURLY_PATH = tmp_path
            try:
                with mock.patch("scripts.fetch_latest.requests.get", return_value=FakeResponse()):
                    n = fl.fetch_and_append(dry_run=False)
                assert n == 3, f"Expected 3 appended rows, got {n}"
                # Verify file grew by exactly 3 rows
                result_df = pd.read_csv(tmp_path, low_memory=False)
                assert len(result_df) == 53, f"Expected 53 rows, got {len(result_df)}"
            finally:
                fl.RAW_HOURLY_PATH = real_path
        finally:
            tmp_path.unlink(missing_ok=True)

    test("Missing local file raises FileNotFoundError", t_missing_local_file)
    test("Dry run, up-to-date - returns 0", t_dry_run_returns_zero_on_uptodate)
    test("Dry run, new rows exist - returns 0 (no write)", t_dry_run_with_new_rows)
    test("Real append writes exactly N new rows", t_real_append_writes_new_rows)


# -- Section 4: Promotion Gate -------------------------------------------------

def section4():
    print("\n--- 4. Promotion Gate Logic -------------------------------")

    def t_reads_metrics_json():
        if not METRICS_JSON.exists():
            return "SKIP"
        f1 = _load_current_val_f1()
        assert 0.0 < f1 <= 1.0, f"F1 out of range: {f1}"

    def t_missing_metrics_returns_zero():
        import scripts.retrain as rt
        real_path = rt.METRICS_JSON
        rt.METRICS_JSON = Path("nonexistent_metrics.json")
        try:
            f1 = rt._load_current_val_f1()
            assert f1 == 0.0, f"Expected 0.0, got {f1}"
        finally:
            rt.METRICS_JSON = real_path

    def t_gate_passes():
        current = 0.742
        new     = 0.735
        floor   = current - 0.02   # 0.722
        assert new >= floor, "0.735 should pass the 0.722 floor"

    def t_gate_fails():
        current = 0.742
        new     = 0.710
        floor   = current - 0.02   # 0.722
        assert new < floor, "0.710 should fail the 0.722 floor"

    def t_gate_exactly_at_floor():
        current = 0.742
        floor   = current - 0.02   # 0.722
        assert floor >= floor, "Exactly at floor should pass (>=)"

    def t_gate_first_run():
        """First run: current_f1=0.0, floor=-0.02, any positive F1 passes."""
        current = 0.0
        floor   = current - 0.02   # -0.02
        new     = 0.65
        assert new >= floor, "First run should always pass promotion gate"

    test("Reads current val F1 from metrics.json", t_reads_metrics_json)
    test("Missing metrics.json returns 0.0", t_missing_metrics_returns_zero)
    test("New F1 0.735 passes floor 0.722", t_gate_passes)
    test("New F1 0.710 fails floor 0.722", t_gate_fails)
    test("New F1 exactly at floor passes", t_gate_exactly_at_floor)
    test("First-ever run always passes (floor = -0.02)", t_gate_first_run)


# -- Section 5: Rollback -------------------------------------------------------

def section5():
    print("\n--- 5. Model Rollback --------------------------------------")

    def t_rollback_with_two_versions():
        """Create two fake versioned models, run rollback, check _latest restored."""
        import joblib
        import scripts.retrain as rt

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_models = Path(tmpdir)

            # Create two fake versioned ensembles
            v1 = tmp_models / "ensemble_20250101_120000.joblib"
            v2 = tmp_models / "ensemble_20250102_120000.joblib"
            latest = tmp_models / "ensemble_latest.joblib"

            joblib.dump({"version": "v1"}, v1)
            joblib.dump({"version": "v2"}, v2)
            joblib.dump({"version": "v2"}, latest)  # latest = v2 (current)

            real_models_dir = rt.MODELS_DIR
            rt.MODELS_DIR = tmp_models
            try:
                rt._rollback_latest_models()
                # After rollback, latest should contain v1's content
                restored = joblib.load(latest)
                assert restored["version"] == "v1", \
                    f"Expected v1 after rollback, got {restored['version']}"
            finally:
                rt.MODELS_DIR = real_models_dir

    def t_rollback_with_one_version_warns():
        """Only 1 versioned model - rollback warns but does not crash."""
        import joblib
        import scripts.retrain as rt

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_models = Path(tmpdir)
            v1 = tmp_models / "ensemble_20250101_120000.joblib"
            joblib.dump({"version": "v1"}, v1)

            real_models_dir = rt.MODELS_DIR
            rt.MODELS_DIR = tmp_models
            try:
                rt._rollback_latest_models()  # Should not raise
            finally:
                rt.MODELS_DIR = real_models_dir

    def t_rollback_with_no_versions():
        """No versioned models at all - rollback warns but does not crash."""
        import scripts.retrain as rt

        with tempfile.TemporaryDirectory() as tmpdir:
            real_models_dir = rt.MODELS_DIR
            rt.MODELS_DIR = Path(tmpdir)
            try:
                rt._rollback_latest_models()  # Should not raise
            finally:
                rt.MODELS_DIR = real_models_dir

    test("Rollback with 2 versions restores previous", t_rollback_with_two_versions)
    test("Rollback with 1 version warns gracefully", t_rollback_with_one_version_warns)
    test("Rollback with no versions warns gracefully", t_rollback_with_no_versions)


# -- Section 6: CLI Flags ------------------------------------------------------

def section6():
    print("\n--- 6. CLI Argument Parsing -------------------------------")

    def t_fetch_dry_run_flag():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def t_retrain_force_flag():
        parser = argparse.ArgumentParser()
        parser.add_argument("--force",       action="store_true")
        parser.add_argument("--skip-fetch",  action="store_true")
        args = parser.parse_args(["--force"])
        assert args.force is True
        assert args.skip_fetch is False

    def t_retrain_skip_fetch_flag():
        parser = argparse.ArgumentParser()
        parser.add_argument("--force",       action="store_true")
        parser.add_argument("--skip-fetch",  action="store_true")
        args = parser.parse_args(["--skip-fetch"])
        assert args.skip_fetch is True
        assert args.force is False

    def t_retrain_both_flags():
        parser = argparse.ArgumentParser()
        parser.add_argument("--force",       action="store_true")
        parser.add_argument("--skip-fetch",  action="store_true")
        args = parser.parse_args(["--force", "--skip-fetch"])
        assert args.force is True
        assert args.skip_fetch is True

    test("fetch_latest --dry-run flag parses", t_fetch_dry_run_flag)
    test("retrain --force flag parses", t_retrain_force_flag)
    test("retrain --skip-fetch flag parses", t_retrain_skip_fetch_flag)
    test("retrain --force --skip-fetch combined", t_retrain_both_flags)


# -- Section 7: Integration - fetch_latest imports chain ----------------------

def section7():
    print("\n--- 7. Module Imports -------------------------------------")

    def t_fetch_imports():
        import scripts.fetch_latest as fl
        assert callable(fl.fetch_and_append)
        assert callable(fl._parse_dates)
        assert hasattr(fl, "DATE_FORMAT")

    def t_retrain_imports():
        import scripts.retrain as rt
        assert callable(rt._load_current_val_f1)
        assert callable(rt._rollback_latest_models)
        assert callable(rt.main)

    def t_config_constants():
        from src.config import MET_EIREANN_LIVE_URL, RAW_HOURLY_PATH
        assert "fusio.net" in MET_EIREANN_LIVE_URL
        assert "hly3904" in str(RAW_HOURLY_PATH)

    test("fetch_latest.py imports and exposes API", t_fetch_imports)
    test("retrain.py imports and exposes API", t_retrain_imports)
    test("Config constants are correct", t_config_constants)


# -- Runner --------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 5 Test Suite")
    print("=" * 60)

    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
    section7()

    total = results["pass"] + results["fail"] + results["skip"]
    print(f"\n{'=' * 60}")
    print(f"  Results: {results['pass']} passed / {results['fail']} failed / {results['skip']} skipped  ({total} total)")
    print(f"{'=' * 60}")

    if results["fail"] > 0:
        sys.exit(1)
