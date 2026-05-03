"""
src/models/onset_offset.py
──────────────────────────
Binary classifiers for rain onset and offset transition events.

Why separate classifiers?
─────────────────────────
Onset (dry → wet) and Offset (wet → dry) have different physical drivers.
Onset is driven by approaching fronts — pressure drops, humidity rises,
dew point depression narrows. Offset is driven by clearing — pressure
rises, wind shifts, cloud cover decreases. Combining them into a single
model forces it to learn contradictory patterns simultaneously.

Class imbalance (19:1)
──────────────────────
Onset/offset events occur in ~5.7% of hours (from audit: 31,863 onset,
31,860 offset events in 557,987 rows). This is a 16.5:1 imbalance — much
more extreme than the 5.8:1 of the main rain classifier.

Strategy: SMOTE + LightGBM `is_unbalance=True` ("belt and braces").
- SMOTE synthetically upsamples onset/offset rows in feature space by
  interpolating between existing minority samples. sampling_strategy=0.3
  means upsample to 30% of majority (not 50/50 — we don't want to
  over-represent onset, just give the model enough examples to learn).
- SMOTE applied to TRAINING SET ONLY. Val and test are never resampled.
- `is_unbalance=True` adds an additional LightGBM-level reweighting on top.

Evaluation
──────────
Standard F1 is the primary metric for model selection. Precision@3 is
computed as a secondary metric to answer: "If we predict the top 3 most
likely onset hours in a day, how many are actually onset hours?"

Public API
──────────
    OnsetOffsetClassifier(event_type='onset')
        .fit(X_train, y_train, X_val, y_val)
        .predict_events(X)    -> list[dict]
        .predict_proba(X)     -> np.ndarray

    build_onset_classifier()  -> OnsetOffsetClassifier
    build_offset_classifier() -> OnsetOffsetClassifier
    train_onset_offset(train_df, val_df) -> tuple[OnsetOffsetClassifier, OnsetOffsetClassifier, dict]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

from src.config import FEATURE_COLUMNS
from src.evaluate import find_optimal_threshold, precision_at_k

# Onset/offset event target columns (created in features.py)
TARGET_ONSET  = "rain_onset"
TARGET_OFFSET = "rain_offset"


class OnsetOffsetClassifier:
    """
    Binary classifier for a single transition event type (onset OR offset).

    Parameters
    ----------
    event_type : 'onset' | 'offset'
    """

    def __init__(self, event_type: str):
        if event_type not in ("onset", "offset"):
            raise ValueError(f"event_type must be 'onset' or 'offset', got '{event_type}'")

        self.event_type = event_type
        self.threshold  = 0.5
        self._is_fitted = False

        self._model = lgb.LGBMClassifier(
            n_estimators=600,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_by_tree=0.8,
            is_unbalance=True,          # auto-reweight on top of SMOTE
            objective="binary",
            metric="average_precision", # more informative than AUC for rare events
            verbose=-1,
            random_state=42,
        )

        # SMOTE: upsample minority class to 30% of majority size.
        # 0.3 chosen deliberately — not 0.5. We want enough minority samples
        # to learn patterns without the model becoming over-confident on onset.
        self._smote = SMOTE(sampling_strategy=0.3, random_state=42)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "OnsetOffsetClassifier":
        """
        Fit the classifier. SMOTE is applied to training data only.

        Parameters
        ----------
        X_train, X_val : feature DataFrames (FEATURE_COLUMNS only are used)
        y_train, y_val : binary Series (0=no event, 1=onset/offset event)
        """
        X_tr = X_train[FEATURE_COLUMNS]
        X_v  = X_val[FEATURE_COLUMNS]

        n_pos = y_train.sum()
        n_neg = (y_train == 0).sum()
        print(f"  [{self.event_type}] Training: {n_pos:,} events / {n_neg:,} non-events "
              f"({100*n_pos/(n_pos+n_neg):.2f}%)")

        # SMOTE — on training features only
        X_resampled, y_resampled = self._smote.fit_resample(X_tr, y_train)
        print(f"  [{self.event_type}] After SMOTE: {y_resampled.sum():,} events / "
              f"{(y_resampled == 0).sum():,} non-events")

        # Train with early stopping on val set
        self._model.fit(
            X_resampled, y_resampled,
            eval_set=[(X_v, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),  # silence per-round output
            ],
        )

        # Calibrate threshold on the real (un-resampled) validation set
        val_prob = self._model.predict_proba(X_v)[:, 1]
        self.threshold = find_optimal_threshold(y_val.values, val_prob)
        print(f"  [{self.event_type}] Optimal threshold = {self.threshold:.4f}")

        self._is_fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of event for each row."""
        assert self._is_fitted, f"Call fit() before predict_proba()"
        return self._model.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def predict_events(self, X: pd.DataFrame) -> list[dict]:
        """
        Return a list of predicted event dicts for rows above threshold.

        Each dict: {'datetime': Timestamp, 'confidence': float, 'event': str}

        Typical usage: call on a 24h or 72h forecast window to get a list
        like [{'datetime': '2025-08-25 14:00', 'confidence': 0.83, 'event': 'onset'}]
        which maps directly to the API response "Rain expected to start around 14:00".
        """
        assert self._is_fitted, f"Call fit() before predict_events()"
        probs = self.predict_proba(X)
        events = []
        for ts, prob in zip(X.index, probs):
            if prob >= self.threshold:
                events.append({
                    "datetime":   ts,
                    "confidence": round(float(prob), 3),
                    "event":      self.event_type,
                })
        return events

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"OnsetOffsetClassifier(event_type={self.event_type!r}, "
            f"status={status}, threshold={self.threshold:.4f})"
        )


# ── Convenience builders ──────────────────────────────────────────────────────

def build_onset_classifier()  -> OnsetOffsetClassifier:
    return OnsetOffsetClassifier("onset")

def build_offset_classifier() -> OnsetOffsetClassifier:
    return OnsetOffsetClassifier("offset")


# ── Combined training function ────────────────────────────────────────────────

def train_onset_offset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[OnsetOffsetClassifier, OnsetOffsetClassifier, dict]:
    """
    Train and evaluate both onset and offset classifiers.

    Returns
    -------
    onset_clf   : fitted OnsetOffsetClassifier
    offset_clf  : fitted OnsetOffsetClassifier
    metrics     : dict with val and test F1, avg_precision, precision_at_3
    """
    metrics = {}

    for event_type, target_col in [("onset", TARGET_ONSET), ("offset", TARGET_OFFSET)]:
        print(f"\n  --- {event_type.upper()} CLASSIFIER ---")

        clf = OnsetOffsetClassifier(event_type)
        clf.fit(
            train_df, train_df[target_col],
            val_df,   val_df[target_col],
        )

        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            y_true = split_df[target_col].values
            y_prob = clf.predict_proba(split_df)

            y_pred = (y_prob >= clf.threshold).astype(int)
            tp     = ((y_pred == 1) & (y_true == 1)).sum()
            fp     = ((y_pred == 1) & (y_true == 0)).sum()
            fn     = ((y_pred == 0) & (y_true == 1)).sum()
            prec   = tp / (tp + fp + 1e-9)
            rec    = tp / (tp + fn + 1e-9)
            f1     = 2 * prec * rec / (prec + rec + 1e-9)
            p_at_3 = precision_at_k(y_true, y_prob, k=3)

            metrics[f"{event_type}_{split_name}_f1"]            = round(float(f1), 4)
            metrics[f"{event_type}_{split_name}_precision"]      = round(float(prec), 4)
            metrics[f"{event_type}_{split_name}_recall"]         = round(float(rec), 4)
            metrics[f"{event_type}_{split_name}_precision_at_3"] = round(float(p_at_3), 4)

            print(f"  [{event_type}] {split_name:4}  F1={f1:.4f}  "
                  f"Prec={prec:.4f}  Rec={rec:.4f}  P@3={p_at_3:.4f}")

        if event_type == "onset":
            onset_clf = clf
        else:
            offset_clf = clf

    return onset_clf, offset_clf, metrics


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import FEATURES_PARQUET, TRAIN_END_DATE, VAL_END_DATE, TEST_START_DATE

    print("Loading features...")
    df = pd.read_parquet(FEATURES_PARQUET)

    train = df.loc[:TRAIN_END_DATE]
    val   = df.loc[TRAIN_END_DATE:VAL_END_DATE].iloc[1:]
    test  = df.loc[TEST_START_DATE:]

    # Sanity-check label counts before training
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        for col in (TARGET_ONSET, TARGET_OFFSET):
            n = split[col].sum()
            pct = 100 * split[col].mean()
            print(f"  {split_name:5} {col:15}: {n:,} events  ({pct:.2f}%)")

    print("\nTraining onset and offset classifiers...")
    onset_clf, offset_clf, metrics = train_onset_offset(train, val, test)

    print("\nOnset/offset metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Sample: show predicted onset events in the first week of test set
    sample = test.iloc[:168]
    onset_events  = onset_clf.predict_events(sample)
    offset_events = offset_clf.predict_events(sample)

    print(f"\nPredicted onset events (first 168h of test): {len(onset_events)}")
    for e in onset_events[:10]:
        print(f"  {e['datetime']}  confidence={e['confidence']:.3f}")

    print(f"\nPredicted offset events (first 168h of test): {len(offset_events)}")
    for e in offset_events[:10]:
        print(f"  {e['datetime']}  confidence={e['confidence']:.3f}")
