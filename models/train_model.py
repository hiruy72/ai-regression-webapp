"""
Model training script with improved path handling and model management.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    get_model_path,
    get_metadata_path,
    get_scaler_path,
    ensure_directories,
    REQUIRED_FEATURES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic house price data for demonstration."""
    logger.info(f"Generating {n_samples} synthetic data samples")
    np.random.seed(42)
    
    # Generate features
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples)
    sqft_living = np.random.normal(2000, 800, n_samples)
    sqft_living = np.clip(sqft_living, 500, 8000)
    sqft_lot = np.random.normal(7500, 3000, n_samples)
    sqft_lot = np.clip(sqft_lot, 1000, 50000)
    floors = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples)
    grade = np.random.randint(3, 12, n_samples)
    
    # Create realistic price based on features with some noise
    price = (
        bedrooms * 15000 +
        bathrooms * 20000 +
        sqft_living * 150 +
        sqft_lot * 2 +
        floors * 10000 +
        grade * 25000 +
        np.random.normal(0, 50000, n_samples)
    )
    
    # Ensure positive prices
    price = np.clip(price, 100000, 2000000)
    
    data = pd.DataFrame({
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'grade': grade,
        'price': price
    })
    
    logger.info(f"Generated data shape: {data.shape}")
    logger.info(f"Price range: ${data['price'].min():,.0f} - ${data['price'].max():,.0f}")
    
    return data


def train_model(algorithm: str = 'random_forest', n_samples: int = 2000) -> tuple:
    """Train regression model and save it with proper path handling."""
    logger.info(f"Training {algorithm} model with {n_samples} samples...")
    
    # Ensure directories exist
    ensure_directories()
    
    # Generate or load data
    data = generate_synthetic_data(n_samples)
    
    # Prepare features and target
    X = data[REQUIRED_FEATURES]
    y = data['price']
    
    logger.info(f"Features: {REQUIRED_FEATURES}")
    logger.info(f"Target variable: price")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # Initialize scaler
    scaler = StandardScaler()
    use_scaler = False
    
    # Train model based on algorithm
    if algorithm == 'linear_regression':
        logger.info("Training Linear Regression model...")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        use_scaler = True
        
    elif algorithm == 'random_forest':
        logger.info("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        use_scaler = False
        
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model Performance:")
    logger.info(f"  RMSE: ${rmse:,.2f}")
    logger.info(f"  MAE: ${mae:,.2f}")
    logger.info(f"  R²: {r2:.4f}")
    
    # Get file paths
    model_path = get_model_path(algorithm)
    metadata_path = get_metadata_path(algorithm)
    scaler_path = get_scaler_path(algorithm)
    
    # Save model
    logger.info(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)
    
    # Save scaler if used
    if use_scaler:
        logger.info(f"Saving scaler to: {scaler_path}")
        joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'algorithm': algorithm,
        'features': REQUIRED_FEATURES,
        'target_variable': 'price',
        'training_date': datetime.now().isoformat(),
        'training_samples': n_samples,
        'test_samples': len(X_test),
        'performance_metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'use_scaler': use_scaler,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path) if use_scaler else None,
        'metadata_path': str(metadata_path)
    }
    
    logger.info(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model training completed successfully!")
    
    return model, metadata


def main():
    """Train both models for comparison."""
    algorithms = ['random_forest', 'linear_regression']
    
    logger.info("Starting model training pipeline...")
    logger.info("=" * 60)
    
    results = {}
    
    for algorithm in algorithms:
        try:
            logger.info(f"Training {algorithm}...")
            model, metadata = train_model(algorithm, n_samples=2000)
            results[algorithm] = metadata['performance_metrics']
            logger.info(f"✓ {algorithm} training completed")
            
        except Exception as e:
            logger.error(f"✗ {algorithm} training failed: {e}")
            results[algorithm] = {"error": str(e)}
        
        logger.info("-" * 60)
    
    # Summary
    logger.info("Training Summary:")
    for algorithm, metrics in results.items():
        if "error" in metrics:
            logger.error(f"  {algorithm}: FAILED - {metrics['error']}")
        else:
            logger.info(f"  {algorithm}:")
            logger.info(f"    RMSE: ${metrics['rmse']:,.2f}")
            logger.info(f"    MAE: ${metrics['mae']:,.2f}")
            logger.info(f"    R²: {metrics['r2']:.4f}")
    
    logger.info("=" * 60)
    logger.info("Training pipeline completed!")


if __name__ == "__main__":
    main()