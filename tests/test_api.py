"""
Tests for the FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "uptime_seconds" in data


def test_model_info_endpoint():
    """Test the model info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm" in data or "error" in data


def test_prediction_endpoint_valid_input():
    """Test prediction with valid input."""
    test_features = {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 2000,
        "sqft_lot": 7500,
        "floors": 2,
        "grade": 7
    }
    
    response = client.post(
        "/predict/regression",
        json={"features": test_features, "explain": True}
    )
    
    # Should succeed if model is loaded, otherwise return 503
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "timestamp" in data
        assert "model_version" in data
        assert isinstance(data["prediction"], (int, float))


def test_prediction_endpoint_missing_features():
    """Test prediction with missing features."""
    test_features = {
        "bedrooms": 3,
        "bathrooms": 2
        # Missing other required features
    }
    
    response = client.post(
        "/predict/regression",
        json={"features": test_features}
    )
    
    assert response.status_code == 422  # Validation error


def test_prediction_endpoint_invalid_values():
    """Test prediction with invalid feature values."""
    test_features = {
        "bedrooms": -1,  # Invalid value
        "bathrooms": 2,
        "sqft_living": 2000,
        "sqft_lot": 7500,
        "floors": 2,
        "grade": 7
    }
    
    response = client.post(
        "/predict/regression",
        json={"features": test_features}
    )
    
    assert response.status_code in [400, 422]  # Validation error


def test_prediction_endpoint_non_numeric_values():
    """Test prediction with non-numeric values."""
    test_features = {
        "bedrooms": "three",  # String instead of number
        "bathrooms": 2,
        "sqft_living": 2000,
        "sqft_lot": 7500,
        "floors": 2,
        "grade": 7
    }
    
    response = client.post(
        "/predict/regression",
        json={"features": test_features}
    )
    
    assert response.status_code == 422  # Validation error