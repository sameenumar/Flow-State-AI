"""Backend configuration settings."""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
AI_MODULES_DIR = BASE_DIR / "ai_modules"

# Application settings
APP_NAME = "Flow-State-AI"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "True") == "True"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# CORS settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# Model settings
MODEL_CONFIDENCE_THRESHOLD = 0.5
MAX_FRAME_SIZE = (640, 480)
