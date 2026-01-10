# API Examples

## Using curl (Windows PowerShell)

### Health Check
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET
```

### Make a Prediction
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/predict/regression" -Method POST -ContentType "application/json" -Body '{"features":{"bedrooms":3,"bathrooms":2,"sqft_living":2000,"sqft_lot":7500,"floors":2,"grade":7},"explain":true}'
```

### Get Model Info
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/model/info" -Method GET
```

## Using curl (Linux/Mac)

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Make a Prediction
```bash
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
```

## Using Python requests

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Make a prediction
prediction_data = {
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

response = requests.post(
    f"{BASE_URL}/predict/regression",
    json=prediction_data
)

result = response.json()
print(f"Predicted price: ${result['prediction']:,.2f}")
print(f"Feature importance: {result['feature_importance']}")
```

## Postman Collection

### 1. Health Check
- **Method**: GET
- **URL**: `http://localhost:8000/health`

### 2. Predict House Price
- **Method**: POST
- **URL**: `http://localhost:8000/predict/regression`
- **Headers**: 
  - `Content-Type: application/json`
- **Body** (raw JSON):
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

### 3. Get Model Information
- **Method**: GET
- **URL**: `http://localhost:8000/model/info`

## Sample Responses

### Health Check Response
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 150,
  "version": "1.0.0",
  "last_prediction": "2024-01-10T15:30:45.123456"
}
```

### Prediction Response
```json
{
  "prediction": 622090.10,
  "confidence_interval": [512392.45, 731787.76],
  "feature_importance": {
    "sqft_living": 0.694,
    "grade": 0.158,
    "bathrooms": 0.032,
    "bedrooms": 0.030,
    "floors": 0.047,
    "sqft_lot": 0.039
  },
  "model_version": "random_forest_v1.0",
  "timestamp": "2024-01-10T15:30:45.123456"
}
```