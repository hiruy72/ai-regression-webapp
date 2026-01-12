#!/bin/bash
# Setup script for House Price Predictor

set -e  # Exit on any error

echo "🏠 House Price Predictor - Setup Script"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies (including Jupyter)
echo "📚 Installing development dependencies..."
pip install -r requirements-dev.txt

# Setup Jupyter kernel
echo "🔬 Setting up Jupyter kernel..."
python -m ipykernel install --user --name house-price-predictor --display-name "House Price Predictor"

# Train models if they don't exist
echo "🤖 Checking for trained models..."
if [ ! -f "models/saved/random_forest_model.joblib" ] || [ ! -f "models/saved/linear_regression_model.joblib" ]; then
    echo "🏋️  Training models..."
    python models/train_model.py
    echo "✅ Models trained successfully"
else
    echo "✅ Models already exist"
fi

# Run tests if available
if [ -d "tests" ]; then
    echo "🧪 Running tests..."
    python -m pytest tests/ -v
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Start API server: python run.py api"
echo "  3. Start frontend: python run.py frontend"
echo "  4. Or start both: python run.py both"
echo ""
echo "To use Jupyter:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Start Jupyter: jupyter notebook notebooks/"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:3000"
echo "API documentation: http://localhost:8000/docs"