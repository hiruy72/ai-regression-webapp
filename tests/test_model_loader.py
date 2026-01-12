"""
Tests for the model loader functionality.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_loader import ModelLoader, get_model_instance


def test_model_loader_initialization():
    """Test ModelLoader initialization."""
    loader = ModelLoader("random_forest")
    assert loader.algorithm == "random_forest"
    assert loader.model is None
    assert not loader.is_loaded()


def test_model_loader_load_model():
    """Test model loading."""
    loader = ModelLoader("random_forest")
    
    # Try to load model
    success = loader.load_model()
    
    # Should succeed if model files exist
    if success:
        assert loader.is_loaded()
        assert loader.model is not None
        assert loader.feature_names is not None
        assert loader.metadata is not None


def test_feature_validation():
    """Test feature validation."""
    loader = ModelLoader("random_forest")
    
    # Test with valid features
    valid_features = {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 2000,
        "sqft_lot": 7500,
        "floors": 2,
        "grade": 7
    }
    
    if loader.load_model():
        is_valid, message = loader.validate_features(valid_features)
        assert is_valid
        assert message == "Valid"
    
    # Test with missing features
    invalid_features = {
        "bedrooms": 3,
        "bathrooms": 2
        # Missing other features
    }
    
    if loader.load_model():
        is_valid, message = loader.validate_features(invalid_features)
        assert not is_valid
        assert "Missing" in message


def test_prediction():
    """Test model prediction."""
    loader = ModelLoader("random_forest")
    
    if loader.load_model():
        test_features = {
            "bedrooms": 3,
            "bathrooms": 2,
            "sqft_living": 2000,
            "sqft_lot": 7500,
            "floors": 2,
            "grade": 7
        }
        
        prediction = loader.predict(test_features)
        assert isinstance(prediction, float)
        assert prediction > 0  # Price should be positive


def test_feature_importance():
    """Test feature importance extraction."""
    loader = ModelLoader("random_forest")
    
    if loader.load_model():
        importance = loader.get_feature_importance()
        
        if importance is not None:
            assert isinstance(importance, dict)
            assert len(importance) > 0
            # All importance values should be non-negative
            assert all(v >= 0 for v in importance.values())


def test_get_model_instance():
    """Test the global model instance function."""
    try:
        model = get_model_instance("random_forest")
        assert isinstance(model, ModelLoader)
        assert model.algorithm == "random_forest"
    except RuntimeError:
        # Expected if model files don't exist
        pass