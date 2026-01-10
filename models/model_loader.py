import joblib
import json
import numpy as np
from typing import Dict, Optional, Tuple
import os

class ModelLoader:
    def __init__(self, algorithm='random_forest'):
        self.algorithm = algorithm
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
    def load_model(self):
        """Load the trained model and associated components"""
        try:
            # Load metadata
            metadata_path = f'models/saved/{self.algorithm}_metadata.json'
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Load model
            model_path = self.metadata['model_path']
            self.model = joblib.load(model_path)
            
            # Load scaler if needed
            if self.metadata.get('use_scaler', False):
                scaler_path = self.metadata['scaler_path']
                self.scaler = joblib.load(scaler_path)
            
            self.feature_names = self.metadata['features']
            print(f"Model {self.algorithm} loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features: Dict[str, float]) -> float:
        """Make prediction from feature dictionary"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert features dict to array in correct order
        feature_array = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        
        # Apply scaling if needed
        if self.scaler is not None:
            feature_array = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.model.predict(feature_array)[0]
        return float(prediction)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients as importance
            importance = np.abs(self.model.coef_)
            # Normalize to sum to 1
            importance = importance / np.sum(importance)
            return dict(zip(self.feature_names, importance))
        
        return None
    
    def get_model_info(self) -> Dict:
        """Get model metadata and performance info"""
        if self.metadata is None:
            return {}
        
        return {
            'algorithm': self.metadata['algorithm'],
            'training_date': self.metadata['training_date'],
            'performance_metrics': self.metadata['performance_metrics'],
            'features': self.metadata['features']
        }
    
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, str]:
        """Validate input features"""
        if self.feature_names is None:
            return False, "Model not loaded"
        
        # Check all required features are present
        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            return False, f"Missing features: {missing_features}"
        
        # Check for extra features
        extra_features = set(features.keys()) - set(self.feature_names)
        if extra_features:
            return False, f"Unknown features: {extra_features}"
        
        # Validate feature ranges (basic validation)
        validations = {
            'bedrooms': (1, 10),
            'bathrooms': (1, 10),
            'sqft_living': (500, 10000),
            'sqft_lot': (1000, 100000),
            'floors': (1, 4),
            'grade': (1, 13)
        }
        
        for feature, value in features.items():
            if feature in validations:
                min_val, max_val = validations[feature]
                if not (min_val <= value <= max_val):
                    return False, f"{feature} must be between {min_val} and {max_val}"
        
        return True, "Valid"

# Global model instance
model_instance = None

def get_model_instance(algorithm='random_forest'):
    """Get or create model instance"""
    global model_instance
    if model_instance is None or model_instance.algorithm != algorithm:
        model_instance = ModelLoader(algorithm)
        if not model_instance.load_model():
            raise RuntimeError(f"Failed to load {algorithm} model")
    return model_instance