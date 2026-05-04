# Makefile — Cork City Weather Predictor
# ─────────────────────────────────────────────────────────────────────────────
# Works in:
#   Windows PowerShell  : make <target>   (requires 'make' from winget/choco)
#   WSL / Linux / macOS : make <target>   (native)
#   Docker              : make <target>   (inside container)
#
# Install make on Windows:
#   winget install GnuWin32.Make
#   — or —
#   choco install make
# ─────────────────────────────────────────────────────────────────────────────

PYTHON   := python
PIP      := pip
PYTEST   := $(PYTHON) -m pytest
UVICORN  := $(PYTHON) -m uvicorn

# ── Dev setup ─────────────────────────────────────────────────────────────────

.PHONY: install
install:          ## Install all dependencies
	$(PIP) install -r requirements.txt

.PHONY: setup
setup: install    ## Full first-time setup (install + data pipeline)
	$(PYTHON) -m src.data.loader
	$(PYTHON) -m src.data.cleaner
	$(PYTHON) -m src.data.features

# ── Testing ───────────────────────────────────────────────────────────────────

.PHONY: test
test:             ## Run all tests
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-fast
test-fast:        ## Run fast tests only (skip phase5 heavyweight tests)
	$(PYTEST) tests/ --ignore=tests/test_phase5.py -q

.PHONY: test-phase6
test-phase6:      ## Run Phase 6 tests only
	$(PYTEST) tests/test_phase6.py -v

# ── Data pipeline ─────────────────────────────────────────────────────────────

.PHONY: fetch
fetch:            ## Fetch latest Met Éireann data and reload API
	$(PYTHON) scripts/fetch_latest.py

.PHONY: fetch-dry
fetch-dry:        ## Check for new data without writing to disk
	$(PYTHON) scripts/fetch_latest.py --dry-run

.PHONY: features
features:         ## Rebuild feature parquet from cleaned data
	$(PYTHON) -m src.data.features

# ── Training ──────────────────────────────────────────────────────────────────

.PHONY: train
train:            ## Train ensemble (XGBoost + LightGBM)
	$(PYTHON) -m src.train

.PHONY: train-phase3
train-phase3:     ## Train onset/offset classifiers (Phase 3)
	$(PYTHON) scripts/train_phase3.py

.PHONY: retrain
retrain:          ## Full nightly retrain (fetch + clean + features + train)
	$(PYTHON) scripts/retrain.py --skip-fetch

.PHONY: retrain-full
retrain-full:     ## Full retrain including Met Éireann fetch
	$(PYTHON) scripts/retrain.py

# ── API ───────────────────────────────────────────────────────────────────────

.PHONY: serve
serve:            ## Start the FastAPI server (dev mode, auto-reload)
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: serve-prod
serve-prod:       ## Start the FastAPI server (production, no auto-reload)
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --workers 1

.PHONY: reload
reload:           ## Hot-reload API parquet without restart
	$(PYTHON) -c "import requests; r=requests.post('http://localhost:8000/admin/reload'); print(r.json())"

.PHONY: forecast
forecast:         ## Print today's forecast and clothing advice to terminal
	$(PYTHON) scripts/print_forecast.py

# ── Scheduling (dev machine only) ─────────────────────────────────────────────

.PHONY: schedule-windows
schedule-windows: ## Register Task Scheduler jobs (run as Administrator)
	powershell -ExecutionPolicy Bypass -File scripts/setup_scheduler.ps1

.PHONY: schedule-linux
schedule-linux:   ## Register cron jobs (Linux / WSL / cloud server)
	bash scripts/setup_cron.sh

# ── Utilities ─────────────────────────────────────────────────────────────────

.PHONY: clean
clean:            ## Remove pycache and pytest artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

.PHONY: logs
logs:             ## Tail the pipeline log
	tail -f logs/pipeline.log

.PHONY: help
help:             ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
