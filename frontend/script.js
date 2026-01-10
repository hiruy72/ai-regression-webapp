const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const form = document.getElementById('predictionForm');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const predictionDiv = document.getElementById('prediction');
const confidenceDiv = document.getElementById('confidence');
const featureImportanceDiv = document.getElementById('featureImportance');
const importanceChart = document.getElementById('importanceChart');
const btnText = document.querySelector('.btn-text');
const btnLoading = document.querySelector('.btn-loading');
const submitBtn = document.querySelector('.submit-btn');
const apiStatus = document.getElementById('apiStatus');
const errorMessage = document.getElementById('errorMessage');

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(form);
    const features = {};
    
    for (let [key, value] of formData.entries()) {
        features[key] = parseFloat(value);
    }
    
    try {
        showLoading();
        hideResults();
        
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
            const errorData = await response.json();
            throw new Error(errorData.detail || `Error: ${response.status}`);
        }
        
        const result = await response.json();
        displayResult(result);
        
    } catch (error) {
        displayError(error.message);
    } finally {
        hideLoading();
    }
});

function showLoading() {
    submitBtn.disabled = true;
    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');
}

function hideLoading() {
    submitBtn.disabled = false;
    btnText.classList.remove('hidden');
    btnLoading.classList.add('hidden');
}

function hideResults() {
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
}

function displayResult(result) {
    // Format price
    const price = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(result.prediction);
    
    predictionDiv.textContent = price;
    
    // Show confidence
    if (result.confidence_interval) {
        const margin = Math.round((result.confidence_interval[1] - result.confidence_interval[0]) / 2);
        confidenceDiv.textContent = `±$${margin.toLocaleString()} confidence range`;
    } else {
        confidenceDiv.textContent = 'High confidence prediction';
    }
    
    // Show feature importance
    if (result.feature_importance) {
        displayFeatureImportance(result.feature_importance);
    }
    
    resultDiv.classList.remove('hidden');
}

function displayFeatureImportance(importance) {
    const features = Object.entries(importance)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 4);
    
    const maxImportance = Math.max(...Object.values(importance));
    
    importanceChart.innerHTML = features.map(([feature, value]) => `
        <div class="importance-item">
            <span class="importance-name">${formatFeatureName(feature)}</span>
            <div class="importance-bar">
                <div class="importance-fill" style="width: ${(value / maxImportance) * 100}%"></div>
            </div>
            <span class="importance-value">${(value * 100).toFixed(0)}%</span>
        </div>
    `).join('');
    
    featureImportanceDiv.classList.remove('hidden');
}

function formatFeatureName(feature) {
    const names = {
        'bedrooms': 'Bedrooms',
        'bathrooms': 'Bathrooms', 
        'sqft_living': 'Living Area',
        'sqft_lot': 'Lot Size',
        'floors': 'Floors',
        'grade': 'Grade'
    };
    return names[feature] || feature;
}

function displayError(message) {
    errorMessage.textContent = message;
    errorDiv.classList.remove('hidden');
}

// API Health Check
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            apiStatus.textContent = 'API Ready';
            apiStatus.className = 'status healthy';
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        apiStatus.textContent = 'API Offline';
        apiStatus.className = 'status error';
    }
}

// Input validation
function setupInputValidation() {
    const inputs = form.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        input.addEventListener('input', () => {
            const value = parseFloat(input.value);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            
            if (value && (value < min || value > max)) {
                input.style.borderColor = '#ef4444';
            } else {
                input.style.borderColor = '#d1d5db';
            }
        });
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    setupInputValidation();
    
    // Check API health every 30 seconds
    setInterval(checkApiHealth, 30000);
});