"""
Model loading and management utilities.
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import joblib

from config import (
    get_model_path,
    get_metadata_path, 
    get_scaler_path,
    REQUIRED_FEATURES,
    FEATURE_RANGES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and using trained ML models."""
    
    def __init__(self, algorithm: str = "random_forest"):
        self.algorithm = algorithm
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        self._is_loaded = False
        
    def load_model(self) -> bool:
        """Load the trained model and associated components."""
        try:
            # Get file paths
            model_path = get_model_path(self.algorithm)
            metadata_path = get_metadata_path(self.algorithm)
            scaler_path = get_scaler_path(self.algorithm)
            
            # Check if model files exist
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
                
            if not metadata_path.exists():
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
            
            # Load metadata first
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Load model
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            
            # Load scaler if needed
            if self.metadata.get('use_scaler', False) and scaler_path.exists():
                logger.info(f"Loading scaler from: {scaler_path}")
                self.scaler = joblib.load(scaler_path)
            
            self.feature_names = self.metadata.get('features', REQUIRED_FEATURES)
            self._is_loaded = True
            
            logger.info(f"Model {self.algorithm} loaded successfully")
            logger.info(f"Model features: {self.feature_names}")
            logger.info(f"Model performance: {self.metadata.get('performance_metrics', {})}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._is_loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded and self.model is not None
    
    def predict(self, features: Dict[str, Union[float, int]]) -> float:
        """Make prediction from feature dictionary."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate features
        is_valid, error_msg = self.validate_features(features)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Convert features dict to array in correct order
        try:
            feature_array = np.array([
                float(features[name]) for name in self.feature_names
            ]).reshape(1, -1)
        except KeyError as e:
            raise ValueError(f"Missing required feature: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid feature value: {e}")
        
        # Apply scaling if needed
        if self.scaler is not None:
            feature_array = self.scaler.transform(feature_array)
        
        # Make prediction
        try:
            prediction = self.model.predict(feature_array)[0]
            return float(prediction)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_loaded():
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models (Random Forest, etc.)
                importance = self.model.feature_importances_
                return dict(zip(self.feature_names, importance))
            elif hasattr(self.model, 'coef_'):
                # Linear models
                importance = np.abs(self.model.coef_)
                # Normalize to sum to 1
                importance = importance / np.sum(importance)
                return dict(zip(self.feature_names, importance))
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
        
        return None
    
    def get_model_info(self) -> Dict:
        """Get model metadata and performance info."""
        if not self.is_loaded() or self.metadata is None:
            return {
                "algorithm": self.algorithm,
                "loaded": False,
                "error": "Model not loaded"
            }
        
        return {
            "algorithm": self.metadata.get("algorithm", self.algorithm),
            "training_date": self.metadata.get("training_date"),
            "performance_metrics": self.metadata.get("performance_metrics", {}),
            "features": self.metadata.get("features", []),
            "loaded": True,
            "use_scaler": self.metadata.get("use_scaler", False)
        }
    
    def validate_features(self, features: Dict[str, Union[float, int]]) -> Tuple[bool, str]:
        """Validate input features."""
        if not self.feature_names:
            return False, "Model not loaded"
        
        # Check all required features are present
        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            return False, f"Missing required features: {list(missing_features)}"
        
        # Check for extra features
        extra_features = set(features.keys()) - set(self.feature_names)
        if extra_features:
            return False, f"Unknown features: {list(extra_features)}"
        
        # Validate feature ranges
        for feature, value in features.items():
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False, f"Feature '{feature}' must be a number, got: {value}"
            
            if feature in FEATURE_RANGES:
                min_val, max_val = FEATURE_RANGES[feature]
                if not (min_val <= value <= max_val):
                    return False, f"Feature '{feature}' must be between {min_val} and {max_val}, got: {value}"
        
        return True, "Valid"


# Global model instance
_model_instance: Optional[ModelLoader] = None


def get_model_instance(algorithm: str = "random_forest") -> ModelLoader:
    """Get or create model instance (singleton pattern)."""
    global _model_instance
    
    if _model_instance is None or _model_instance.algorithm != algorithm:
        logger.info(f"Creating new model instance for algorithm: {algorithm}")
        _model_instance = ModelLoader(algorithm)
        
        if not _model_instance.load_model():
            raise RuntimeError(f"Failed to load {algorithm} model. Please ensure model files exist.")
    
    return _model_instance


def reload_model(algorithm: str = "random_forest") -> ModelLoader:
    """Force reload of model instance."""
    global _model_instance
    _model_instance = None
    return get_model_instance(algorithm)