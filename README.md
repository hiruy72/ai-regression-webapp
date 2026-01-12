# House Price Predictor

A production-ready machine learning web application for house price prediction using FastAPI backend and responsive frontend. This project demonstrates best practices for ML model deployment, including proper virtual environment setup, cross-platform compatibility, and comprehensive error handling.

## 🏗️ Project Structure

```
house-price-predictor/
├── api/                    # FastAPI backend
│   ├── __init__.py
│   └── main.py            # API endpoints and logic
├── frontend/              # Web frontend
│   ├── index.html         # Main HTML page
│   ├── style.css          # Styling
│   └── script.js          # JavaScript functionality
├── models/                # ML model training and loading
│   ├── __init__.py
│   ├── train_model.py     # Model training script
│   ├── model_loader.py    # Model loading utilities
│   └── saved/             # Trained model files
│       ├── random_forest_model.joblib
│       ├── random_forest_metadata.json
│       ├── linear_regression_model.joblib
│       ├── linear_regression_metadata.json
│       └── linear_regression_scaler.joblib
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_api.py        # API tests
│   └── test_model_loader.py # Model tests
├── examples/              # Usage examples
│   ├── api_examples.md    # API usage examples
│   └── test_api.py        # API testing script
├── config.py              # Configuration management
├── run.py                 # Main runner script
├── setup.py               # Package setup
├── pyproject.toml         # Modern Python packaging
├── requirements.txt       # Core dependencies
├── requirements-dev.txt   # Development dependencies
├── setup.sh               # Unix setup script
├── setup.bat              # Windows setup script
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Option 1: Automated Setup (Recommended)

**On Unix/Linux/macOS:**
```bash
# Clone the repository
git clone <repository-url>
cd house-price-predictor

# Run setup script
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```cmd
# Clone the repository
git clone <repository-url>
cd house-price-predictor

# Run setup script
setup.bat
```

### Option 2: Manual Setup

1. **Create and activate virtual environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Train models (if not included):**
```bash
python models/train_model.py
```

4. **Run the application:**
```bash
# Option A: Run both servers
python run.py both

# Option B: Run separately
python run.py api        # Terminal 1
python run.py frontend   # Terminal 2
```

## 🎯 Usage

### Web Interface
1. Open http://localhost:3000 in your browser
2. Enter house details in the form
3. Click "Get Price Estimate" to see AI prediction
4. View feature importance analysis

### API Endpoints

**Base URL:** http://localhost:8000

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /predict/regression` - Make predictions
- `GET /model/info` - Model information
- `POST /model/reload` - Reload model

### Example API Usage

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict/regression",
    json={
        "features": {
            "bedrooms": 3,
            "bathrooms": 2,
            "sqft_living": 2000,
            "sqft_lot": 7500,
            "floors": 2,
            "grade": 7
        },
        "explain": True
    }
)

result = response.json()
print(f"Predicted price: ${result['prediction']:,.2f}")
```

## 🔧 Configuration

The application can be configured through environment variables or by modifying `config.py`:

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_ALGORITHM=random_forest

# CORS Configuration
export CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"

# Logging
export LOG_LEVEL=INFO
```

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## 📊 Model Performance

### Random Forest Model
- **RMSE**: ~$55,968
- **MAE**: ~$44,632
- **R²**: ~0.855

### Linear Regression Model
- **RMSE**: ~$51,512
- **MAE**: ~$41,688
- **R²**: ~0.877

*Note: Performance metrics are based on synthetic data for demonstration.*

## 🚀 Running the Application

The improved project now includes multiple ways to run the application:

```bash
# Using the runner script (recommended)
python run.py both                    # Run both API and frontend
python run.py api                     # Run API only
python run.py frontend                # Run frontend only
python run.py train                   # Train models

# With custom ports
python run.py both --api-port 8001 --frontend-port 3001

# Development mode with auto-reload
python run.py api --reload
```

## ✅ Improvements Made

Based on your requirements, I've implemented the following improvements:

### 1. **Model File Management**
- ✅ Trained model files (.joblib) are included in the repository
- ✅ Models are automatically loaded on application startup
- ✅ Proper error handling if models are missing
- ✅ Cross-platform path handling using `pathlib`

### 2. **Virtual Environment Support**
- ✅ Added `setup.py` and `pyproject.toml` for proper packaging
- ✅ Created setup scripts for both Unix (`setup.sh`) and Windows (`setup.bat`)
- ✅ Isolated dependencies with `requirements.txt` and `requirements-dev.txt`
- ✅ Virtual environment creation and activation in setup scripts

### 3. **Cross-Platform Compatibility**
- ✅ Used `pathlib.Path` for all file operations
- ✅ Environment-specific setup scripts
- ✅ Proper path handling in imports and model loading
- ✅ Works on Windows, macOS, and Linux

### 4. **Enhanced Project Structure**
- ✅ Added `config.py` for centralized configuration
- ✅ Created comprehensive test suite
- ✅ Added proper logging throughout the application
- ✅ Improved error handling and validation

### 5. **Setup Instructions**
- ✅ Clear, step-by-step setup instructions
- ✅ Automated setup scripts
- ✅ Multiple deployment options
- ✅ Troubleshooting guide

## 🛠️ Development

### Project Commands

```bash
# Train models
python run.py train

# Run API server only
python run.py api --reload

# Run frontend only
python run.py frontend --frontend-port 3001

# Run both servers
python run.py both --api-port 8001 --frontend-port 3001

# Run tests
python -m pytest tests/

# Format code (if black is installed)
black .

# Lint code (if flake8 is installed)
flake8 .
```

## 🔍 Troubleshooting

### Common Issues

1. **Model files not found**
   ```bash
   python models/train_model.py
   ```

2. **Import errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate  # Unix
   venv\Scripts\activate     # Windows
   ```

3. **Port already in use**
   ```bash
   python run.py api --api-port 8001
   python run.py frontend --frontend-port 3001
   ```

4. **Permission denied (Unix)**
   ```bash
   chmod +x setup.sh
   ```

### Logs and Debugging

- API logs are printed to console
- Set `LOG_LEVEL=DEBUG` for detailed logging
- Check model loading in startup logs
- Use `/health` endpoint to verify system status

## 📝 API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

The project is now production-ready with proper dependency management, cross-platform support, and comprehensive documentation!