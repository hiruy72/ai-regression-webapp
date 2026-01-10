# Regression Prediction Web App

A full-stack machine learning application for house price prediction using FastAPI backend and responsive frontend.

## Features

- **Machine Learning Models**: Random Forest and Linear Regression algorithms
- **REST API**: FastAPI backend with automatic documentation
- **Responsive Frontend**: Clean, intuitive web interface
- **Real-time Predictions**: Instant price predictions with feature importance
- **Model Explanations**: Optional feature importance visualization
- **Health Monitoring**: API health checks and status monitoring

## Project Structure

```
├── models/                 # ML model training and loading
│   ├── train_model.py     # Model training script
│   ├── model_loader.py    # Model loading utilities
│   └── saved/             # Trained model files (generated)
├── api/                   # FastAPI backend
│   └── main.py           # API endpoints and logic
├── frontend/              # Web frontend
│   ├── index.html        # Main HTML page
│   ├── style.css         # Styling
│   └── script.js         # JavaScript functionality
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd models
python train_model.py
```

This will:
- Generate synthetic house price data
- Train Random Forest and Linear Regression models
- Save models and metadata to `models/saved/`
- Display performance metrics (RMSE, MAE, R²)

### 3. Start the API Server

```bash
cd api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it using a simple HTTP server:

```bash
cd frontend
python -m http.server 3000
```

Then visit: http://localhost:3000

## API Endpoints

### POST /predict/regression
Make a house price prediction.

**Request Body:**
```json
{
  "features": {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 2000,
    "sqft_lot": 7500,
    "floors": 2,
    "grade": 7
  },
  "explain": true
}
```

**Response:**
```json
{
  "prediction": 425000.50,
  "confidence_interval": [375000.25, 474000.75],
  "feature_importance": {
    "sqft_living": 0.35,
    "grade": 0.25,
    "bathrooms": 0.15,
    "bedrooms": 0.12,
    "floors": 0.08,
    "sqft_lot": 0.05
  },
  "model_version": "random_forest_v1.0",
  "timestamp": "2024-01-10T15:30:45.123456"
}
```

### GET /health
Check API service health.

### GET /docs
Interactive API documentation (Swagger UI).

## Example API Requests

### Using curl

```bash
# Make a prediction
curl -X POST "http://localhost:8000/predict/regression" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "bedrooms": 3,
         "bathrooms": 2.5,
         "sqft_living": 2200,
         "sqft_lot": 8000,
         "floors": 2,
         "grade": 8
       },
       "explain": true
     }'

# Check health
curl -X GET "http://localhost:8000/health"
```

### Using Python requests

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict/regression",
    json={
        "features": {
            "bedrooms": 4,
            "bathrooms": 3,
            "sqft_living": 2800,
            "sqft_lot": 10000,
            "floors": 2,
            "grade": 9
        },
        "explain": True
    }
)

result = response.json()
print(f"Predicted price: ${result['prediction']:,.2f}")
```

## Model Performance

### Random Forest Model
- **RMSE**: ~$85,000
- **MAE**: ~$65,000  
- **R²**: ~0.85

### Linear Regression Model
- **RMSE**: ~$95,000
- **MAE**: ~$72,000
- **R²**: ~0.80

*Note: These are approximate values from synthetic data. Actual performance may vary.*

## Features Explained

### Input Features
- **Bedrooms**: Number of bedrooms (1-10)
- **Bathrooms**: Number of bathrooms (1-10, including half baths)
- **Living Area**: Square footage of living space (500-10,000 sqft)
- **Lot Size**: Square footage of the lot (1,000-100,000 sqft)
- **Floors**: Number of floors (1-4, including half floors)
- **Grade**: Construction quality grade (1-13, where 7 is average)

### Model Features
- **Feature Importance**: Shows which features most influence the prediction
- **Confidence Intervals**: Provides uncertainty estimates for predictions
- **Real-time Validation**: Input validation with helpful error messages
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Development

### Adding New Models

1. Implement training in `models/train_model.py`
2. Update `models/model_loader.py` to handle the new algorithm
3. Modify `api/main.py` to use the new model

### Customizing Features

1. Update the feature list in `models/train_model.py`
2. Modify validation rules in `models/model_loader.py`
3. Update the frontend form in `frontend/index.html`
4. Adjust the API request schema in `api/main.py`

## Testing

The application includes comprehensive error handling and validation:

- **Input Validation**: Ensures all required features are present and within valid ranges
- **Model Loading**: Graceful handling of missing or corrupted model files
- **API Error Handling**: Proper HTTP status codes and error messages
- **Frontend Validation**: Real-time form validation and user feedback

## Deployment Considerations

- **Environment Variables**: Configure API URLs for different environments
- **CORS**: Update CORS settings for production domains
- **Model Versioning**: Implement model versioning for production deployments
- **Monitoring**: Add logging and monitoring for production use
- **Security**: Implement authentication and rate limiting as needed

## License

This project is for educational and demonstration purposes.