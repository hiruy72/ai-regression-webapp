from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Dict, Optional, Tuple, Union
from datetime import datetime
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_loader import get_model_instance

app = FastAPI(
    title="Regression Prediction API",
    description="A FastAPI service for house price prediction using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: Dict[str, Union[float, int]]
    explain: Optional[bool] = False
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        required_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade']
        
        # Check all required features are present
        missing = set(required_features) - set(v.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Convert all values to float
        for key, value in v.items():
            try:
                v[key] = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{key}' must be a number")
        
        return v

class PredictionResponse(BaseModel):
    prediction: float
    confidence_interval: Optional[Tuple[float, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_version: str
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: int
    version: str
    last_prediction: Optional[datetime] = None

# Global variables
app_start_time = datetime.now()
last_prediction_time = None

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Regression Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/predict/regression", response_model=PredictionResponse, tags=["Prediction"])
async def predict_regression(request: PredictionRequest):
    """
    Make a regression prediction based on input features
    
    - **features**: Dictionary of feature values (bedrooms, bathrooms, sqft_living, sqft_lot, floors, grade)
    - **explain**: Optional flag to include feature importance in response
    """
    global last_prediction_time
    
    try:
        # Get model instance
        model = get_model_instance('random_forest')
        
        # Validate features
        is_valid, error_msg = model.validate_features(request.features)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Make prediction
        prediction = model.predict(request.features)
        
        # Get feature importance if requested
        feature_importance = None
        if request.explain:
            feature_importance = model.get_feature_importance()
        
        # Calculate confidence interval (simplified approach)
        confidence_interval = None
        if model.metadata and 'performance_metrics' in model.metadata:
            rmse = model.metadata['performance_metrics']['rmse']
            # Simple confidence interval: prediction ± 1.96 * RMSE (95% CI approximation)
            margin = 1.96 * rmse
            confidence_interval = (prediction - margin, prediction + margin)
        
        last_prediction_time = datetime.now()
        
        return PredictionResponse(
            prediction=prediction,
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
            model_version=f"{model.algorithm}_v1.0",
            timestamp=last_prediction_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check the health status of the API service
    """
    try:
        # Try to get model instance to check if it's loaded
        model = get_model_instance('random_forest')
        model_loaded = model.model is not None
        status = "healthy" if model_loaded else "degraded"
        
    except Exception:
        model_loaded = False
        status = "unhealthy"
    
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        uptime_seconds=int(uptime),
        version="1.0.0",
        last_prediction=last_prediction_time
    )

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model
    """
    try:
        model = get_model_instance('random_forest')
        return model.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)