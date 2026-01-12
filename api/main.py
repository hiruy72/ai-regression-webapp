"""
FastAPI application for house price prediction.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_loader import get_model_instance, reload_model
from config import (
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    LOG_LEVEL,
    DEFAULT_MODEL_ALGORITHM,
    REQUIRED_FEATURES
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app = FastAPI(
    title="House Price Prediction API",
    description="A FastAPI service for house price prediction using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for tracking
app_start_time = datetime.now()
last_prediction_time: Optional[datetime] = None


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, Union[float, int]]
    explain: Optional[bool] = False
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v: Dict[str, Union[float, int]]) -> Dict[str, float]:
        """Validate and convert feature values."""
        if not isinstance(v, dict):
            raise ValueError("Features must be a dictionary")
        
        # Check all required features are present
        missing = set(REQUIRED_FEATURES) - set(v.keys())
        if missing:
            raise ValueError(f"Missing required features: {list(missing)}")
        
        # Convert all values to float and validate
        validated_features = {}
        for key, value in v.items():
            try:
                validated_features[key] = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{key}' must be a number, got: {value}")
        
        return validated_features


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: float
    confidence_interval: Optional[Tuple[float, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_version: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    model_loaded: bool
    model_algorithm: str
    uptime_seconds: int
    version: str
    last_prediction: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: str
    timestamp: datetime


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    """Handle runtime errors."""
    logger.error(f"Runtime error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Runtime Error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


# API Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/regression",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.post("/predict/regression", response_model=PredictionResponse, tags=["Prediction"])
async def predict_regression(request: PredictionRequest):
    """
    Make a house price prediction based on input features.
    
    - **features**: Dictionary of feature values (bedrooms, bathrooms, sqft_living, sqft_lot, floors, grade)
    - **explain**: Optional flag to include feature importance in response
    """
    global last_prediction_time
    
    try:
        # Get model instance
        model = get_model_instance(DEFAULT_MODEL_ALGORITHM)
        
        if not model.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check server logs."
            )
        
        # Make prediction
        prediction = model.predict(request.features)
        
        # Get feature importance if requested
        feature_importance = None
        if request.explain:
            feature_importance = model.get_feature_importance()
        
        # Calculate confidence interval (simplified approach)
        confidence_interval = None
        model_info = model.get_model_info()
        if model_info.get('performance_metrics'):
            rmse = model_info['performance_metrics'].get('rmse', 0)
            if rmse > 0:
                # Simple confidence interval: prediction ± 1.96 * RMSE (95% CI approximation)
                margin = 1.96 * rmse
                confidence_interval = (prediction - margin, prediction + margin)
        
        last_prediction_time = datetime.now()
        
        logger.info(f"Prediction made: ${prediction:,.2f}")
        
        return PredictionResponse(
            prediction=prediction,
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
            model_version=f"{model.algorithm}_v1.0",
            timestamp=last_prediction_time
        )
        
    except ValueError as e:
        logger.warning(f"Validation error in prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the API service."""
    try:
        # Try to get model instance to check if it's loaded
        model = get_model_instance(DEFAULT_MODEL_ALGORITHM)
        model_loaded = model.is_loaded()
        status_value = "healthy" if model_loaded else "degraded"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        model_loaded = False
        status_value = "unhealthy"
    
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status=status_value,
        model_loaded=model_loaded,
        model_algorithm=DEFAULT_MODEL_ALGORITHM,
        uptime_seconds=int(uptime),
        version="1.0.0",
        last_prediction=last_prediction_time
    )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    try:
        model = get_model_instance(DEFAULT_MODEL_ALGORITHM)
        return model.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )


@app.post("/model/reload", tags=["Model"])
async def reload_model_endpoint():
    """Reload the model (useful for updates)."""
    try:
        model = reload_model(DEFAULT_MODEL_ALGORITHM)
        logger.info("Model reloaded successfully")
        return {
            "message": "Model reloaded successfully",
            "algorithm": model.algorithm,
            "loaded": model.is_loaded(),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading model: {str(e)}"
        )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting House Price Prediction API...")
    logger.info(f"API Host: {API_HOST}")
    logger.info(f"API Port: {API_PORT}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Default Model: {DEFAULT_MODEL_ALGORITHM}")
    
    try:
        # Pre-load the model
        model = get_model_instance(DEFAULT_MODEL_ALGORITHM)
        if model.is_loaded():
            logger.info("✓ Model loaded successfully on startup")
        else:
            logger.warning("⚠ Model failed to load on startup")
    except Exception as e:
        logger.error(f"✗ Failed to load model on startup: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown."""
    logger.info("Shutting down House Price Prediction API...")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level=LOG_LEVEL.lower()
    )