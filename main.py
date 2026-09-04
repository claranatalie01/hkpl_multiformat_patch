"""Backward-compatible ASGI entry point for local and container execution.

The FastAPI application lives in :mod:`hkpl_agent.app`. Keeping this wrapper
preserves the established ``uvicorn main:app`` command.
"""

import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

app = import_module("hkpl_agent.app").app

__all__ = ["app"]
