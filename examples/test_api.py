#!/usr/bin/env python3
"""
Test script for the Regression Prediction API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        health_data = response.json()
        print(f"✓ Health check passed: {health_data['status']}")
        print(f"  Model loaded: {health_data['model_loaded']}")
        print(f"  Uptime: {health_data['uptime_seconds']} seconds")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\nTesting prediction endpoint...")
    
    test_cases = [
        {
            "name": "Small house",
            "features": {
                "bedrooms": 2,
                "bathrooms": 1,
                "sqft_living": 1200,
                "sqft_lot": 5000,
                "floors": 1,
                "grade": 6
            }
        },
        {
            "name": "Medium house",
            "features": {
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft_living": 2000,
                "sqft_lot": 7500,
                "floors": 2,
                "grade": 7
            }
        },
        {
            "name": "Large house",
            "features": {
                "bedrooms": 5,
                "bathrooms": 4,
                "sqft_living": 3500,
                "sqft_lot": 12000,
                "floors": 2,
                "grade": 10
            }
        }
    ]
    
    for test_case in test_cases:
        try:
            prediction_data = {
                "features": test_case["features"],
                "explain": True
            }
            
            response = requests.post(
                f"{BASE_URL}/predict/regression",
                json=prediction_data
            )
            response.raise_for_status()
            
            result = response.json()
            price = result['prediction']
            
            print(f"✓ {test_case['name']}: ${price:,.2f}")
            
            # Show top 3 most important features
            if result.get('feature_importance'):
                importance = result['feature_importance']
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top features: {', '.join([f'{k}({v:.1%})' for k, v in top_features])}")
            
        except Exception as e:
            print(f"✗ {test_case['name']} failed: {e}")

def test_invalid_input():
    """Test error handling with invalid input"""
    print("\nTesting error handling...")
    
    invalid_cases = [
        {
            "name": "Missing features",
            "data": {"features": {"bedrooms": 3}}
        },
        {
            "name": "Invalid feature values",
            "data": {"features": {
                "bedrooms": -1,
                "bathrooms": 2,
                "sqft_living": 2000,
                "sqft_lot": 7500,
                "floors": 2,
                "grade": 7
            }}
        },
        {
            "name": "Non-numeric values",
            "data": {"features": {
                "bedrooms": "three",
                "bathrooms": 2,
                "sqft_living": 2000,
                "sqft_lot": 7500,
                "floors": 2,
                "grade": 7
            }}
        }
    ]
    
    for test_case in invalid_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/predict/regression",
                json=test_case["data"]
            )
            
            if response.status_code == 400 or response.status_code == 422:
                print(f"✓ {test_case['name']}: Correctly rejected (HTTP {response.status_code})")
            else:
                print(f"✗ {test_case['name']}: Should have been rejected but got HTTP {response.status_code}")
                
        except Exception as e:
            print(f"✓ {test_case['name']}: Correctly rejected ({e})")

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        response.raise_for_status()
        info = response.json()
        print(f"✓ Model info retrieved:")
        print(f"  Algorithm: {info.get('algorithm', 'N/A')}")
        print(f"  Training date: {info.get('training_date', 'N/A')}")
        if 'performance_metrics' in info:
            metrics = info['performance_metrics']
            print(f"  RMSE: ${metrics.get('rmse', 0):,.2f}")
            print(f"  MAE: ${metrics.get('mae', 0):,.2f}")
            print(f"  R²: {metrics.get('r2', 0):.4f}")
    except Exception as e:
        print(f"✗ Model info failed: {e}")

def main():
    """Run all tests"""
    print("=" * 50)
    print("Regression Prediction API Test Suite")
    print("=" * 50)
    
    # Test if API is available
    if not test_health():
        print("\n❌ API is not available. Please ensure the server is running on port 8000.")
        return
    
    # Run all tests
    test_prediction()
    test_invalid_input()
    test_model_info()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()