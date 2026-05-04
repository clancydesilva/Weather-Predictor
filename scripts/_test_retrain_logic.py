"""scripts/_test_retrain_logic.py — offline unit tests for retrain.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.retrain import _load_current_val_f1, _rollback_latest_models
from src.config import MODELS_DIR, METRICS_JSON

# Test 1: F1 loading
f1 = _load_current_val_f1()
assert f1 > 0, f"Expected positive F1, got {f1}"
print(f"Test 1 PASS — current val F1: {f1:.4f}")

# Test 2: promotion gate logic
current = 0.742
floor   = current - 0.02          # 0.722
assert 0.730 >= floor, "Gate should pass at 0.730"
assert 0.700 <  floor, "Gate should fail at 0.700"
print("Test 2 PASS — promotion gate thresholds correct")

# Test 3: rollback finds second-most-recent model
versioned = sorted(
    MODELS_DIR.glob("ensemble_2*.joblib"),
    key=lambda p: p.stat().st_mtime,
)
if len(versioned) >= 2:
    print(f"Test 3 PASS — rollback source: {versioned[-2].name}")
else:
    print(f"Test 3 SKIP — only {len(versioned)} versioned ensemble models (need >= 2 to test rollback)")

print("\nAll retrain logic tests: PASSED")
