"""Default configuration values, overridable via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Inference
SENSQML_RUN_DIR: str = os.getenv("SENSQML_RUN_DIR", str(BASE_DIR / "runs" / "latest"))
SENSQML_THRESHOLD: float = float(os.getenv("SENSQML_THRESHOLD", "0.5"))

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Server
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
