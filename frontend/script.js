const API_BASE_URL = 'http://localhost:8000';

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const features = {};
    
    for (let [key, value] of formData.entries()) {
        features[key] = parseFloat(value);
    }
    
    try {
        showLoading();
        const response = await fetch(`${API_BASE_URL}/predict/regression`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: features,
                explain: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayResult(result);
        
    } catch (error) {
        displayError(`Error making prediction: ${error.message}`);
    } finally {
        hideLoading();
    }
});

function showLoading() {
    const button = document.querySelector('button[type="submit"]');
    button.textContent = 'Predicting...';
    button.disabled = true;
}

function hideLoading() {
    const button = document.querySelector('button[type="submit"]');
    button.textContent = 'Predict Price';
    button.disabled = false;
}

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const predictionDiv = document.getElementById('prediction');
    const errorDiv = document.getElementById('error');
    
    errorDiv.classList.add('hidden');
    
    const formattedPrice = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(result.prediction);
    
    predictionDiv.innerHTML = `
        <div>Predicted Price: ${formattedPrice}</div>
        <div style="font-size: 14px; color: #666; margin-top: 5px;">
            Model: ${result.model_version} | Confidence: ${result.confidence_interval ? 
                `±$${Math.round((result.confidence_interval[1] - result.confidence_interval[0]) / 2).toLocaleString()}` : 'N/A'}
        </div>
    `;
    
    if (result.feature_importance) {
        displayFeatureImportance(result.feature_importance);
    }
    
    resultDiv.classList.remove('hidden');
}

function displayFeatureImportance(importance) {
    const importanceDiv = document.getElementById('featureImportance');
    const chartDiv = document.getElementById('importanceChart');
    
    const sortedFeatures = Object.entries(importance)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 6); // Show top 6 features
    
    const maxImportance = Math.max(...Object.values(importance));
    
    chartDiv.innerHTML = sortedFeatures.map(([feature, value]) => `
        <div class="importance-item">
            <span>${formatFeatureName(feature)}</span>
            <div class="importance-bar">
                <div class="importance-fill" style="width: ${(value / maxImportance) * 100}%"></div>
            </div>
            <span>${(value * 100).toFixed(1)}%</span>
        </div>
    `).join('');
    
    importanceDiv.classList.remove('hidden');
}

function formatFeatureName(feature) {
    const nameMap = {
        'bedrooms': 'Bedrooms',
        'bathrooms': 'Bathrooms',
        'sqft_living': 'Living Area',
        'sqft_lot': 'Lot Size',
        'floors': 'Floors',
        'grade': 'Grade'
    };
    return nameMap[feature] || feature;
}

function displayError(message) {
    const errorDiv = document.getElementById('error');
    const resultDiv = document.getElementById('result');
    
    resultDiv.classList.add('hidden');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

// Check API health on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error('API not available');
        }
        console.log('API is healthy');
    } catch (error) {
        displayError('API service is not available. Please ensure the backend is running on port 8000.');
    }
});