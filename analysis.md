# Codebase Analysis: Cork City Weather Predictor

This document provides an intricate, in-depth review of the Weather-Predictor machine learning pipeline and API. The analysis scrutinizes the data engineering, model architectures, codebase structure, and potential bugs, highlighting both exemplary practices and critical issues that need addressing.

## 1. Architectural Overview & Structure

The codebase is highly modular, clearly separating concerns into distinct logical components:
- **Data Pipeline (`src/data/`)**: Cleaning raw data and engineering cyclic/lag/rolling features.
- **Models (`src/models/`)**: Abstracting complex training logic into a unified `TwoStagePipeline` and custom soft-voting ensemble.
- **Orchestration (`src/train.py`, `scripts/`)**: Scripts designed around specific sequential phases of the project.
- **API (`api/`)**: A stateless, high-performance inference server using FastAPI.
- **Configuration (`src/config.py`)**: A commendable single source of truth for all paths, constants, thresholds, and hyperparameters.

## 2. Exemplary Practices (Things Done Well)

### A. The Two-Stage Pipeline Design
The most impressive design decision is the `TwoStagePipeline` (`src/models/pipeline.py`). Rainfall forecasting suffers heavily from "zero-inflation" (82.8% of hours in this dataset are dry). A standard regressor would learn to constantly predict near-zero values to minimize aggregate loss. 
By cleanly decoupling the problem into:
1. **Classification (Will it rain?)** trained on all data.
2. **Regression (How much?)** trained **only on wet hours** (`rain_occurred == 1`).

You have successfully bypassed the zero-inflation trap. The final output `P(rain) × amount` acts as a continuous probability gate.

### B. Feature Engineering and Leakage Prevention
Feature generation in `src/data/features.py` is textbook perfect for meteorological data:
- **Leakage Prevention:** You correctly apply `.shift(1)` to all rolling statistics (e.g., `rainfall_roll_mean_3h`). Neglecting this is the #1 cause of data leakage in time-series forecasting.
- **Cyclic Encodings:** Using `sin` and `cos` transforms for hour, month, day-of-week, and day-of-year prevents the model from assuming that hour 23 and hour 0 are far apart.
- **Wind Decomposition:** Converting wind direction and speed into `u_wind` and `v_wind` vectors allows tree-based models to organically understand circular wind directions.

### C. Imbalance Handling
The dataset exhibits extreme class imbalances. Your strategies are well-chosen:
- **XGBoost:** Using `scale_pos_weight = 4.82` (n_dry / n_wet) is mathematically correct and leverages XGBoost's native handling.
- **Onset/Offset Modeler:** A 16.5:1 imbalance is aggressively handled using `SMOTE` with a conservative `sampling_strategy=0.3`. Crucially, SMOTE is applied **only to the training set** while the validation/test sets remain unadulterated, ensuring metrics remain honest.

### D. Threshold Optimization
Rather than defaulting to `0.5` for binary classification, `evaluate.py` calculates the optimal threshold that maximizes the F1-score dynamically on the **Validation Set**. This threshold is seamlessly persisted within the pipeline and correctly applied to unseen data.

---

## 3. Critical Issues & Potential Bugs (Scrutiny)

### A. LightGBM Early Stopping is Silently Broken
In `src/models/lgbm_model.py`, the `build_lgbm_pipeline` function accepts `early_stopping_rounds=50` as a parameter. However, this parameter is **never used or passed** to either `lgb.LGBMClassifier` or `lgb.LGBMRegressor` during initialization. 

Furthermore, in `src/models/pipeline.py`'s `TwoStagePipeline.fit()` method, `_early_stop_kwargs` only passes the `eval_set` to the `.fit()` function. 
Unlike older versions of XGBoost, modern LightGBM's Scikit-Learn API strictly requires early stopping to be defined as a callback during `fit()` (e.g., `callbacks=[lgb.early_stopping(stopping_rounds=50)]`).
**Impact:** Because early stopping is never correctly invoked for LightGBM, the LightGBM models will train blindly for the full `n_estimators` (up to 1,000 rounds). This significantly wastes compute time and risks severe overfitting. 

### B. API Memory Growth (OOM Risk)
In `api/main.py`, the lifespan handler executes:
```python
app.state.features_df = df.loc[TEST_START_DATE:]
```
Currently, this loads all data from `2021-01-01` into memory (about 4+ years of hourly data). Since `get_forecast_window()` in `predict.py` only slices the last `n_hours` (e.g., 24 for a forecast, plus up to 72 for lookback context), loading the *entire* test set into the API memory is highly inefficient. 
**Impact:** As time progresses into Phase 5 (live data fetching), this DataFrame will grow infinitely. Over time, the server will experience memory bloat and potentially crash (OOM). 
**Fix:** You only need to retain the last `LOOKBACK_HOURS + FORECAST_HOURS` rows in `app.state.features_df` during inference.

### C. Evaluation Metric Missing in Regressor Early Stopping
In `pipeline.py`, when calling `_early_stop_kwargs` for the regressor, you pass `TARGET_REGRESSION` (log1p rainfall) as the target. The evaluation metric used by XGBoost and LightGBM for early stopping defaults to standard squared error, but the `build_xgb_pipeline` sets `eval_metric="mae"`. While XGBoost accepts string `eval_metric` in the constructor, standardizing the evaluation functions ensures both frameworks behave identically when deciding when to stop.

### D. Duplicate Variable Extraction Bug 
In `predict.py`, the dictionary generation loop contains a minor oversight:
```python
wind_dir_deg = int(row.get("wind_dir_deg", 0)) if not pd.isna(row.get("wind_dir_deg", 0)) else 0
```
However, the input to `generate_forecast` is `hourly_features.parquet`, which **does not contain NaNs** (they were dropped in `features.py`). Furthermore, `wind_dir_deg` was kept in the feature dataset alongside `u_wind` and `v_wind`, which is safe but slightly redundant if tree models rely exclusively on the decomposed vectors.

---

## 4. Recommendations for Next Steps

1. **Fix LightGBM Early Stopping:** Update `src/models/pipeline.py`'s `_early_stop_kwargs` to return proper callback lists for LightGBM using the `inspect.signature` or explicitly checking model types.
2. **Cap API DataFrame Size:** Update `api/main.py` lifespan to only keep the tail of the DataFrame: `app.state.features_df = df.loc[TEST_START_DATE:].tail(200)`.
3. **Advanced Huber Loss:** In the `TwoStagePipeline` regressor, consider utilizing `Huber` loss (or `reg:pseudohubererror` in XGBoost) instead of standard squared error. Although you capped extreme rain at `6.4mm`, Huber loss offers superior robustness against remaining weather outliers.
4. **Resilience in Production:** If `train_phase3.py` is skipped, `api/main.py` will completely fail to boot due to `FileNotFoundError`. Consider wrapping the model loading in `try/except` blocks and gracefully disabling the `/events` endpoint if the onset/offset classifiers are absent.
