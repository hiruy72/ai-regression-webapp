"""
Configuration settings for the House Price Predictor application.
"""
import os
from pathlib import Path
from typing import Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
API_DIR = PROJECT_ROOT / "api"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Model configuration
DEFAULT_MODEL_ALGORITHM = os.getenv("MODEL_ALGORITHM", "random_forest")
MODEL_FILE_EXTENSION = ".joblib"
METADATA_FILE_EXTENSION = ".json"

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Feature configuration
REQUIRED_FEATURES = [
    "bedrooms",
    "bathrooms", 
    "sqft_living",
    "sqft_lot",
    "floors",
    "grade"
]

FEATURE_RANGES = {
    "bedrooms": (1, 10),
    "bathrooms": (1, 10),
    "sqft_living": (500, 10000),
    "sqft_lot": (1000, 100000),
    "floors": (1, 4),
    "grade": (1, 13)
}

def get_model_path(algorithm: str) -> Path:
    """Get the path to a model file."""
    return MODELS_DIR / f"{algorithm}_model{MODEL_FILE_EXTENSION}"

def get_metadata_path(algorithm: str) -> Path:
    """Get the path to a model metadata file."""
    return MODELS_DIR / f"{algorithm}_metadata{METADATA_FILE_EXTENSION}"

def get_scaler_path(algorithm: str) -> Path:
    """Get the path to a scaler file."""
    return MODELS_DIR / f"{algorithm}_scaler{MODEL_FILE_EXTENSION}"

def ensure_directories():
    """Ensure all required directories exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    API_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)