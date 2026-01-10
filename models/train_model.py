import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
import os

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic house price data for demonstration"""
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
    
    return data

def train_model(algorithm='random_forest'):
    """Train regression model and save it"""
    print(f"Training {algorithm} model...")
    
    # Generate or load data
    data = generate_synthetic_data(2000)
    
    # Prepare features and target
    feature_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade']
    X = data[feature_columns]
    y = data['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for linear regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if algorithm == 'linear_regression':
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        use_scaler = True
    elif algorithm == 'random_forest':
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
    
    print(f"Model Performance:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models/saved', exist_ok=True)
    
    # Save model
    model_path = f'models/saved/{algorithm}_model.joblib'
    joblib.dump(model, model_path)
    
    # Save scaler if used
    if use_scaler:
        scaler_path = f'models/saved/{algorithm}_scaler.joblib'
        joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'algorithm': algorithm,
        'features': feature_columns,
        'target_variable': 'price',
        'training_date': datetime.now().isoformat(),
        'performance_metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'use_scaler': use_scaler,
        'model_path': model_path,
        'scaler_path': f'models/saved/{algorithm}_scaler.joblib' if use_scaler else None
    }
    
    metadata_path = f'models/saved/{algorithm}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return model, metadata

if __name__ == "__main__":
    # Train both models for comparison
    algorithms = ['random_forest', 'linear_regression']
    
    for algorithm in algorithms:
        print(f"\n{'='*50}")
        train_model(algorithm)
        print(f"{'='*50}")