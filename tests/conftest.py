"""
tests/conftest.py
Shared pytest configuration and fixtures for the test suite.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so all imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
