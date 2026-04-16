"""
scripts/audit.py
────────────────
Data profiling report for hly3904.csv. Run this once to verify the full
pipeline is working and produce a paper-trail of data quality decisions.

Outputs
-------
Console report:
  - Row/column counts, date range, gap check
  - Zero fraction and null counts per column
  - Rainfall percentile table
  - Train/val/test split sizes

Plots saved to results/plots/:
  - audit_rainfall_dist.png    — log-scale histogram of rainfall_mm
  - audit_temp_dist.png        — temperature distribution
  - audit_pressure_dist.png    — pressure distribution
  - audit_humidity_dist.png    — humidity post-clipping
  - train_val_test_split.png   — time-series with coloured split bands
                                  (visual proof of no temporal leakage)

Usage
-----
    python scripts/audit.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless/server runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLEAN_PARQUET,
    PLOTS_DIR,
    RAW_HOURLY_PATH,
    RESULTS_DIR,
    RAIN_CAP_MM,
    RAIN_OCCURRENCE_THRESHOLD,
    TRAIN_END_DATE,
    VAL_END_DATE,
    TEST_START_DATE,
)
from src.data.loader import load_raw
from src.data.cleaner import clean_pipeline

# ── Style ─────────────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4a",
    "axes.labelcolor":  "#c8ccd8",
    "xtick.color":      "#8b8fa8",
    "ytick.color":      "#8b8fa8",
    "text.color":       "#c8ccd8",
    "grid.color":       "#2a2d3a",
    "grid.alpha":       0.5,
}
ACCENT   = "#5b8dee"
ACCENT2  = "#e8607a"
ACCENT3  = "#50c878"

SPLIT_COLORS = {
    "train": ("#5b8dee", "Train  (≤ 2015)"),
    "val":   ("#f5a623", "Val    (2016–2020)"),
    "test":  ("#e8607a", "Test   (2021–present)"),
}


def _apply_style(ax: plt.Axes) -> None:
    ax.set_facecolor(PLOT_STYLE["axes.facecolor"])
    ax.tick_params(colors=PLOT_STYLE["xtick.color"])
    ax.xaxis.label.set_color(PLOT_STYLE["axes.labelcolor"])
    ax.yaxis.label.set_color(PLOT_STYLE["axes.labelcolor"])
    ax.title.set_color(PLOT_STYLE["text.color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_STYLE["axes.edgecolor"])
    ax.grid(True, color=PLOT_STYLE["grid.color"], alpha=PLOT_STYLE["grid.alpha"])


def section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ── 1. Load data ──────────────────────────────────────────────────────────────
def run_audit() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    section("1. LOADING RAW DATA")
    print(f"Source: {RAW_HOURLY_PATH}")
    raw = load_raw()
    print(f"Rows:    {len(raw):,}")
    print(f"Columns: {list(raw.columns)}")
    print(f"Date range: {raw.index.min()} to {raw.index.max()}")

    # ── 2. Gap check ─────────────────────────────────────────────────────────
    section("2. GAP CHECK")
    diffs = raw.index.to_series().diff().dropna()
    max_gap = diffs.max()
    gaps_gt_1h = (diffs > pd.Timedelta("1h")).sum()
    print(f"Max gap between consecutive rows: {max_gap}")
    print(f"Gaps > 1 hour: {gaps_gt_1h}")
    if gaps_gt_1h == 0:
        print("  PASS - no temporal gaps in the record.")
    else:
        print(f"  WARNING - {gaps_gt_1h} gaps found. Investigate before training.")

    # ── 3. Null counts ────────────────────────────────────────────────────────
    section("3. NULL COUNTS PER COLUMN")
    null_pct = (raw.isnull().sum() / len(raw) * 100).round(3)
    for col, pct in null_pct.items():
        flag = "  <-- ACTION NEEDED" if pct > 1.0 else ""
        print(f"  {col:<25} {raw[col].isnull().sum():>6,}  ({pct:.3f}%){flag}")

    # ── 4. Rainfall profile ───────────────────────────────────────────────────
    section("4. RAINFALL PROFILE")
    rain = raw["rainfall_mm"].dropna()
    n_total   = len(rain)
    n_zero    = (rain == 0).sum()
    n_point1  = (rain == 0.1).sum()
    n_gt_pt1  = (rain > 0.1).sum()
    n_wet     = (rain >= RAIN_OCCURRENCE_THRESHOLD).sum()

    print(f"  Non-null rows:              {n_total:>8,}")
    print(f"  Exact zeros:                {n_zero:>8,}  ({100*n_zero/n_total:.2f}%)")
    print(f"  Exactly 0.1mm:              {n_point1:>8,}  ({100*n_point1/n_total:.2f}%)")
    print(f"  > 0.1mm (strictly):         {n_gt_pt1:>8,}  ({100*n_gt_pt1/n_total:.2f}%)")
    print(f"  >= 0.1mm (threshold=wet):   {n_wet:>8,}  ({100*n_wet/n_total:.2f}%)")
    print(f"  Max:                        {rain.max():.2f} mm")
    print(f"\n  Percentiles:")
    for p in [50, 75, 90, 95, 99, 99.9, 99.99]:
        print(f"    {p:>6.2f}th: {np.percentile(rain, p):.4f} mm")
    print(f"\n  Rainfall cap (RAIN_CAP_MM): {RAIN_CAP_MM} mm  (99.9th pct from audit)")

    # ── 5. Clean data stats ───────────────────────────────────────────────────
    section("5. CLEAN DATA STATS")
    if CLEAN_PARQUET.exists():
        clean = pd.read_parquet(CLEAN_PARQUET)
        print(f"  Loaded from: {CLEAN_PARQUET}")
    else:
        print("  hourly_clean.parquet not found — running clean_pipeline()...")
        clean = clean_pipeline(raw)

    print(f"  Humidity max (must be <= 100): {clean['humidity_pct'].max():.1f}%")
    print(f"  Rainfall max (must be <= {RAIN_CAP_MM}):  {clean['rainfall_mm'].max():.2f} mm")
    print(f"  Wind speed max:                {clean['wind_speed_kmh'].max():.1f} km/h")
    print(f"  Rain occurred (wet hours):     {clean['rain_occurred'].sum():,}  "
          f"({100*clean['rain_occurred'].mean():.2f}%)")

    # ── 6. Train / val / test split sizes ────────────────────────────────────
    section("6. TRAIN / VAL / TEST SPLIT")
    train = clean.loc[:TRAIN_END_DATE]
    val   = clean.loc[TRAIN_END_DATE:VAL_END_DATE].iloc[1:]
    test  = clean.loc[TEST_START_DATE:]
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        wet_pct = 100 * split["rain_occurred"].mean()
        print(f"  {name:<6}: {len(split):>8,} rows  "
              f"({split.index.min().date()} to {split.index.max().date()})  "
              f"wet={wet_pct:.2f}%")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    section("7. GENERATING PLOTS")
    _plot_rainfall_dist(clean)
    _plot_distribution(clean, "temp_c",       "Temperature (C)",      "audit_temp_dist.png",     ACCENT)
    _plot_distribution(clean, "pressure_hpa", "Pressure (hPa)",       "audit_pressure_dist.png", ACCENT2)
    _plot_distribution(clean, "humidity_pct", "Humidity (%) post-clip","audit_humidity_dist.png", ACCENT3)
    _plot_split(clean, train, val, test)

    section("AUDIT COMPLETE")
    print(f"  Plots saved to: {PLOTS_DIR}")
    print(f"  Run the full pipeline: loader -> cleaner -> features -> train\n")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_rainfall_dist(clean: pd.DataFrame) -> None:
    """Log-scale histogram of non-zero rainfall_mm."""
    wet = clean.loc[clean["rainfall_mm"] > 0, "rainfall_mm"]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PLOT_STYLE["figure.facecolor"])
    ax.hist(wet, bins=80, color=ACCENT, edgecolor="none", alpha=0.85, log=True)
    ax.axvline(RAIN_CAP_MM, color=ACCENT2, lw=1.5, linestyle="--",
               label=f"Cap ({RAIN_CAP_MM}mm, 99.9th pct)")
    ax.set_xlabel("Rainfall (mm/hr)  [non-zero hours only]")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Rainfall Distribution — Non-Zero Hours (log y-axis)")
    ax.legend(facecolor="#1a1d27", edgecolor="#3a3d4a", labelcolor="#c8ccd8")
    _apply_style(ax)

    path = PLOTS_DIR / "audit_rainfall_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_distribution(
    clean: pd.DataFrame, col: str, xlabel: str, filename: str, color: str
) -> None:
    """Generic distribution histogram."""
    data = clean[col].dropna()

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PLOT_STYLE["figure.facecolor"])
    ax.hist(data, bins=80, color=color, edgecolor="none", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(f"{xlabel} Distribution")
    _apply_style(ax)

    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_split(
    clean: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Time-series of rolling 30-day mean rainfall with coloured split bands.
    Visual proof that train/val/test are temporally ordered with no leakage.
    """
    rolling = clean["rainfall_mm"].rolling(24 * 30).mean()   # 30-day rolling mean

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=PLOT_STYLE["figure.facecolor"])

    # Shade split regions
    ax.axvspan(train.index.min(), train.index.max(),
               color=SPLIT_COLORS["train"][0], alpha=0.12)
    ax.axvspan(val.index.min(), val.index.max(),
               color=SPLIT_COLORS["val"][0], alpha=0.15)
    ax.axvspan(test.index.min(), test.index.max(),
               color=SPLIT_COLORS["test"][0], alpha=0.15)

    # Boundary lines
    ax.axvline(pd.Timestamp(TRAIN_END_DATE), color=SPLIT_COLORS["val"][0],
               lw=1, linestyle="--", alpha=0.8)
    ax.axvline(pd.Timestamp(TEST_START_DATE), color=SPLIT_COLORS["test"][0],
               lw=1, linestyle="--", alpha=0.8)

    ax.plot(rolling.index, rolling.values, color=ACCENT, lw=0.8, alpha=0.9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rainfall mm (30-day rolling mean)")
    ax.set_title("Train / Validation / Test Split — No Temporal Leakage")

    patches = [
        mpatches.Patch(color=SPLIT_COLORS[k][0], alpha=0.4, label=SPLIT_COLORS[k][1])
        for k in ("train", "val", "test")
    ]
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#3a3d4a",
              labelcolor="#c8ccd8", loc="upper right")

    _apply_style(ax)
    path = PLOTS_DIR / "train_val_test_split.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


if __name__ == "__main__":
    run_audit()
