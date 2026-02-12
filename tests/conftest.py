"""
Test configuration for the project.

Ensures the project root is on sys.path so tests can import modules
like `config`, `router`, `main`, etc. regardless of how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

