"""
scripts/backtest.py
───────────────────
Human-readable sanity check: compare ensemble predictions against actual
Cork Airport observations from the test set (2021-2025).

This is NOT a metric computation script — that is done in train.py.
This is a *qualitative* check: can you look at the output and say
"yes, the model is doing something sensible"?

Ground truth is the actual hly3904.csv observations — no external API
needed. The model was never trained on 2021-2025 data.

Usage
-----
    # Last 7 days of the test set (default):
    python scripts/backtest.py

    # Specific date (shows 24 hours):
    python scripts/backtest.py --date 2023-10-15

    # 7-day window starting from a specific date:
    python scripts/backtest.py --week 2023-10-15
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    FEATURES_PARQUET,
    MODELS_DIR,
    TARGET_BINARY,
    TARGET_REGRESSION,
    TEST_START_DATE,
)
from src.models.baselines import PersistenceBaseline


# ── Formatting helpers ────────────────────────────────────────────────────────

def _verdict(rain_flag: int, p_rain: float) -> str:
    if rain_flag == 1:
        return f"RAIN ({int(p_rain * 100)}%)"
    return f"DRY  ({int(p_rain * 100)}%)"


def _match(predicted_flag: int, actual_flag: int, actual_mm: float) -> str:
    if predicted_flag == actual_flag:
        return "YES"
    if predicted_flag == 0 and actual_flag == 1:
        return f"MISS  ({actual_mm:.1f}mm missed)"
    return "FALSE ALARM"


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    ensemble,
    label: str,
    n_hours: int | None = None,
) -> dict:
    """
    Run the ensemble on a slice of df and print a comparison table.

    Parameters
    ----------
    df       : feature DataFrame with actual rain_occurred and rainfall_mm
    ensemble : fitted SoftVoteEnsemble (or any model with .predict())
    label    : title string for the output block
    n_hours  : if set, only show first n_hours rows in the table

    Returns
    -------
    dict : {'accuracy', 'f1', 'false_alarm_rate', 'miss_rate', 'persistence_f1'}
    """
    out = ensemble.predict(df)
    rain_prob  = out["rain_probability"]
    rain_flag  = out["rain_flag"]
    pred_mm    = out["rainfall_mm"]

    actual_flag = df[TARGET_BINARY].values
    actual_mm   = np.expm1(df[TARGET_REGRESSION].values)   # un-log the stored target

    # ── Table ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  Cork Weather Backtest  --  {label}")
    print(f"{'=' * 72}")
    print(f"  {'Datetime (IST)':22}  {'Verdict':14}  {'Pred mm':8}  {'Actual mm':10}  Match")
    print(f"  {'-' * 68}")

    display_rows = df.iloc[:n_hours] if n_hours else df
    for i, (ts, _) in enumerate(display_rows.iterrows()):
        verdict = _verdict(rain_flag[i], rain_prob[i])
        match   = _match(rain_flag[i], actual_flag[i], actual_mm[i])
        print(
            f"  {str(ts):22}  {verdict:14}  {pred_mm[i]:7.2f}mm"
            f"  {actual_mm[i]:8.2f}mm  {match}"
        )

    if n_hours and len(df) > n_hours:
        print(f"  ... ({len(df) - n_hours} more rows not shown)")

    # ── Summary stats ─────────────────────────────────────────────────────────
    correct       = (rain_flag == actual_flag).sum()
    accuracy      = correct / len(df)
    tp            = ((rain_flag == 1) & (actual_flag == 1)).sum()
    fp            = ((rain_flag == 1) & (actual_flag == 0)).sum()
    fn            = ((rain_flag == 0) & (actual_flag == 1)).sum()
    precision     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall        = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1            = 2 * precision * recall / (precision + recall + 1e-9)
    false_alarm   = fp / (fp + (rain_flag == 0).sum() + 1e-9)
    miss_rate     = fn / (actual_flag == 1).sum() if actual_flag.sum() > 0 else 0.0

    # Persistence comparison (lag-1)
    pb = PersistenceBaseline()
    pb_flag   = (pb.predict_proba(df) >= 0.5).astype(int)
    pb_tp     = ((pb_flag == 1) & (actual_flag == 1)).sum()
    pb_fp     = ((pb_flag == 1) & (actual_flag == 0)).sum()
    pb_fn     = ((pb_flag == 0) & (actual_flag == 1)).sum()
    pb_prec   = pb_tp / (pb_tp + pb_fp) if (pb_tp + pb_fp) > 0 else 0.0
    pb_rec    = pb_tp / (pb_tp + pb_fn) if (pb_tp + pb_fn) > 0 else 0.0
    pb_f1     = 2 * pb_prec * pb_rec / (pb_prec + pb_rec + 1e-9)

    print(f"\n  {'=' * 68}")
    print(f"  Summary  ({len(df):,} hours)")
    print(f"  {'=' * 68}")
    print(f"  Accuracy          : {accuracy:.1%}")
    print(f"  F1                : {f1:.4f}")
    print(f"  Precision         : {precision:.4f}  (of predicted rain hours, this many were wet)")
    print(f"  Recall            : {recall:.4f}  (of actual rain hours, this many were caught)")
    print(f"  False alarm rate  : {false_alarm:.1%}  (model said rain, was dry)")
    print(f"  Miss rate         : {miss_rate:.1%}  (model said dry, was wet)")
    print(f"")
    print(f"  Persistence F1    : {pb_f1:.4f}  (lag-1 only — no model, just 'it rained last hour')")
    print(f"  Model advantage   : +{(f1 - pb_f1):.4f} F1 over persistence")

    if f1 > pb_f1 + 0.02:
        print(f"  VERDICT: Model genuinely outperforms persistence on this window.")
    elif f1 > pb_f1:
        print(f"  VERDICT: Marginal improvement over persistence. Check miss patterns.")
    else:
        print(f"  VERDICT: Model does NOT beat persistence on this window. Investigate.")

    print(f"  {'=' * 68}\n")

    return {
        "accuracy": accuracy, "f1": f1,
        "false_alarm_rate": false_alarm, "miss_rate": miss_rate,
        "persistence_f1": pb_f1,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ensemble against real Cork observations.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--date", metavar="YYYY-MM-DD",
                       help="Show predictions for a single day (24 hours).")
    group.add_argument("--week", metavar="YYYY-MM-DD",
                       help="Show predictions for 7 days starting from this date.")
    args = parser.parse_args()

    print("Loading features parquet...")
    df_full = pd.read_parquet(FEATURES_PARQUET)
    test    = df_full.loc[TEST_START_DATE:]

    if args.date:
        start = pd.Timestamp(args.date)
        end   = start + pd.Timedelta(hours=23)
        window = test.loc[start:end]
        label  = f"{args.date} (24h)"
        n_show = None
    elif args.week:
        start  = pd.Timestamp(args.week)
        end    = start + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        window = test.loc[start:end]
        label  = f"{args.week} to {end.date()} (7 days)"
        n_show = 48   # show first 48 rows in table, summarise rest
    else:
        # Default: last 7 days of the test set
        end    = test.index.max()
        start  = end - pd.Timedelta(days=7)
        window = test.loc[start:end]
        label  = f"{start.date()} to {end.date()} (last 7 days of test set)"
        n_show = 48

    if len(window) == 0:
        print(f"ERROR: No data found for the requested window. "
              f"Test set runs from {test.index.min().date()} to {test.index.max().date()}.")
        sys.exit(1)

    print(f"Loading ensemble model...")
    model_path = MODELS_DIR / "ensemble_latest.joblib"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run 'python -m src.train' first.")
        sys.exit(1)

    ensemble = joblib.load(model_path)
    print(f"Loaded: {model_path.name}")

    run_backtest(window, ensemble, label, n_show)


if __name__ == "__main__":
    main()
