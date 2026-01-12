@echo off
REM Setup script for House Price Predictor (Windows)

echo 🏠 House Price Predictor - Setup Script
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Install development dependencies (including Jupyter)
echo 📚 Installing development dependencies...
pip install -r requirements-dev.txt

REM Setup Jupyter kernel
echo 🔬 Setting up Jupyter kernel...
python -m ipykernel install --user --name house-price-predictor --display-name "House Price Predictor"

REM Train models if they don't exist
echo 🤖 Checking for trained models...
if not exist "models\saved\random_forest_model.joblib" (
    echo 🏋️  Training models...
    python models\train_model.py
    echo ✅ Models trained successfully
) else (
    echo ✅ Models already exist
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo To start the application:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Start API server: python run.py api
echo   3. Start frontend: python run.py frontend
echo   4. Or start both: python run.py both
echo.
echo To use Jupyter:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Start Jupyter: jupyter notebook notebooks\
echo.
echo API will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:3000
echo API documentation: http://localhost:8000/docs

pause