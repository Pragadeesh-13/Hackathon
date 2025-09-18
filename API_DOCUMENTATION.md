# 🐄 Cattle Breed Recognition API Documentation

## 🚀 Quick Start for Mac React Developer

### Server Information
- **Server IP**: `172.16.45.105` (Windows PC)
- **API Port**: `5000`
- **Base URL**: `http://172.16.45.105:5000`
- **Model Accuracy**: 87.1%
- **Supported Breeds**: 11 (6 Buffalo + 5 Cattle)

---

## 📡 API Endpoints

### 1. **POST /predict** - Predict Cattle Breed
Main endpoint for cattle breed recognition.

**Request:**
```javascript
POST http://172.16.45.105:5000/predict
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "mode": "ensemble"  // optional: "ensemble", "trained", "fusion"
}
```

**Response:**
```javascript
{
  "success": true,
  "prediction": {
    "breed": "Murrah",
    "confidence": 0.891,
    "confidence_level": "VERY_HIGH",
    "is_cattle": true,
    "breed_info": {
      "type": "Buffalo",
      "origin": "Haryana/Punjab",
      "specialty": "Highest milk yield"
    },
    "method_used": "Advanced Ensemble"
  },
  "alternatives": [
    {
      "breed": "Murrah",
      "confidence": 0.891,
      "breed_info": { ... }
    },
    {
      "breed": "Surti", 
      "confidence": 0.067,
      "breed_info": { ... }
    }
  ],
  "metadata": {
    "prediction_time_seconds": 1.234,
    "timestamp": "2025-09-18T01:30:00",
    "model_accuracy": "87.1%",
    "total_breeds": 11
  },
  "ensemble_info": {
    "methods_agree": true,
    "trained_model_prediction": "Murrah",
    "smart_fusion_prediction": "Murrah"
  }
}
```

### 2. **GET /health** - Health Check
Check server and model status.

**Response:**
```javascript
{
  "status": "healthy",
  "timestamp": "2025-09-18T01:30:00",
  "uptime_hours": 2.5,
  "model": {
    "loaded": true,
    "accuracy": "87.1%",
    "supported_breeds": 11,
    "ensemble_system": true
  },
  "performance": {
    "total_predictions": 25,
    "successful_predictions": 24,
    "error_count": 1,
    "success_rate_percent": 96.0
  }
}
```

### 3. **GET /breeds** - List Supported Breeds
Get information about all supported breeds.

**Response:**
```javascript
{
  "total_breeds": 11,
  "breeds": [
    {
      "name": "Murrah",
      "type": "Buffalo", 
      "origin": "Haryana/Punjab",
      "specialty": "Highest milk yield",
      "category": "buffalo"
    },
    // ... more breeds
  ],
  "categories": {
    "cattle": ["Gir", "Kankrej", "Ongole", "Sahiwal", "Tharparkar"],
    "buffalo": ["Bhadawari", "Jaffarbadi", "Mehsana", "Murrah", "Nagpuri", "Surti"]
  }
}
```

### 4. **GET /stats** - Server Statistics
Get server performance metrics.

### 5. **GET /** - API Information
Get API overview and documentation.

---

## ⚛️ React.js Integration

### Installation
```bash
# No additional packages needed - uses fetch API
npm install  # your existing dependencies
```

### Complete React Component Example

```javascript
import React, { useState, useCallback } from 'react';

const CattleBreedRecognition = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [image, setImage] = useState(null);

  const API_BASE_URL = 'http://172.16.45.105:5000';

  // Convert file to base64
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  };

  // Predict cattle breed
  const predictBreed = useCallback(async (imageFile, mode = 'ensemble') => {
    setLoading(true);
    setError(null);
    
    try {
      // Convert image to base64
      const base64Image = await fileToBase64(imageFile);
      
      // Create preview
      setImage(URL.createObjectURL(imageFile));
      
      // Send prediction request
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          mode: mode
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setPrediction(result);
      } else {
        setError(result.error || 'Prediction failed');
      }

    } catch (err) {
      setError(`Error: ${err.message}`);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Check API health
  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const health = await response.json();
      console.log('API Health:', health);
      return health.status === 'healthy';
    } catch (err) {
      console.error('Health check failed:', err);
      return false;
    }
  };

  // Handle file input
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      predictBreed(file);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h1>🐄 Cattle Breed Recognition</h1>
      
      {/* Health Check Button */}
      <button onClick={checkHealth} style={{ marginBottom: '20px' }}>
        Check API Health
      </button>

      {/* File Upload */}
      <div style={{ marginBottom: '20px' }}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          disabled={loading}
        />
      </div>

      {/* Loading State */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <p>🔄 Analyzing cattle breed...</p>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div style={{ 
          background: '#ffebee', 
          color: '#c62828', 
          padding: '10px', 
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          ❌ {error}
        </div>
      )}

      {/* Image Preview */}
      {image && (
        <div style={{ marginBottom: '20px' }}>
          <img 
            src={image} 
            alt="Uploaded cattle" 
            style={{ 
              maxWidth: '300px', 
              height: 'auto', 
              borderRadius: '8px' 
            }} 
          />
        </div>
      )}

      {/* Prediction Results */}
      {prediction && prediction.success && (
        <div style={{ 
          background: '#e8f5e8', 
          padding: '20px', 
          borderRadius: '8px' 
        }}>
          <h2>🎯 Prediction Results</h2>
          
          {/* Main Prediction */}
          <div style={{ marginBottom: '15px' }}>
            <h3>🏆 Predicted Breed: {prediction.prediction.breed}</h3>
            <p><strong>Confidence:</strong> {(prediction.prediction.confidence * 100).toFixed(1)}%</p>
            <p><strong>Type:</strong> {prediction.prediction.breed_info.type}</p>
            <p><strong>Origin:</strong> {prediction.prediction.breed_info.origin}</p>
            <p><strong>Specialty:</strong> {prediction.prediction.breed_info.specialty}</p>
            <p><strong>Method:</strong> {prediction.prediction.method_used}</p>
          </div>

          {/* Confidence Level */}
          <div style={{ marginBottom: '15px' }}>
            <span style={{
              background: getConfidenceColor(prediction.prediction.confidence_level),
              color: 'white',
              padding: '5px 10px',
              borderRadius: '15px',
              fontSize: '12px'
            }}>
              {prediction.prediction.confidence_level} CONFIDENCE
            </span>
          </div>

          {/* Alternative Predictions */}
          <div>
            <h4>📊 Top 5 Predictions:</h4>
            <ul>
              {prediction.alternatives.map((alt, index) => (
                <li key={index}>
                  <strong>{alt.breed}</strong>: {(alt.confidence * 100).toFixed(1)}%
                  ({alt.breed_info.type})
                </li>
              ))}
            </ul>
          </div>

          {/* Ensemble Info */}
          {prediction.ensemble_info && (
            <div style={{ marginTop: '15px', fontSize: '14px' }}>
              <p><strong>Methods Agreement:</strong> {prediction.ensemble_info.methods_agree ? '✅ Yes' : '⚠️ No'}</p>
              <p><strong>Neural Network:</strong> {prediction.ensemble_info.trained_model_prediction}</p>
              <p><strong>Smart Fusion:</strong> {prediction.ensemble_info.smart_fusion_prediction}</p>
            </div>
          )}

          {/* Metadata */}
          <div style={{ marginTop: '15px', fontSize: '12px', color: '#666' }}>
            <p>Prediction time: {prediction.metadata.prediction_time_seconds}s</p>
            <p>Model accuracy: {prediction.metadata.model_accuracy}</p>
          </div>
        </div>
      )}

      {/* Not Cattle Warning */}
      {prediction && prediction.success && !prediction.prediction.is_cattle && (
        <div style={{
          background: '#fff3cd',
          color: '#856404',
          padding: '15px',
          borderRadius: '8px',
          marginTop: '10px'
        }}>
          ⚠️ <strong>Warning:</strong> This image might not be a cattle/buffalo. 
          Please upload a clear image of cattle for better results.
        </div>
      )}
    </div>
  );
};

// Helper function for confidence colors
const getConfidenceColor = (level) => {
  switch (level) {
    case 'VERY_HIGH': return '#28a745';
    case 'HIGH': return '#17a2b8';
    case 'MEDIUM': return '#ffc107';
    case 'LOW': return '#dc3545';
    default: return '#6c757d';
  }
};

export default CattleBreedRecognition;
```

### Simplified Hook Version

```javascript
import { useState, useCallback } from 'react';

const useCattleBreedAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const API_BASE_URL = 'http://172.16.45.105:5000';

  const predictBreed = useCallback(async (imageFile) => {
    setLoading(true);
    setError(null);

    try {
      const fileToBase64 = (file) => {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.readAsDataURL(file);
          reader.onload = () => resolve(reader.result);
          reader.onerror = error => reject(error);
        });
      };

      const base64Image = await fileToBase64(imageFile);
      
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image })
      });

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error);
      }

      return result;

    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { predictBreed, loading, error };
};

export default useCattleBreedAPI;
```

---

## 🔧 Testing & Debugging

### Test API Connection
```bash
# Test from Mac terminal
curl http://172.16.45.105:5000/health

# Expected response:
# {"status": "healthy", "model": {"loaded": true}, ...}
```

### Common Issues & Solutions

#### 1. **Connection Refused**
- Check if API server is running: `python api_server.py`
- Verify firewall settings on Windows PC
- Ensure both devices are on same network

#### 2. **CORS Issues**
- API already has CORS enabled for all origins
- Check browser developer console for errors

#### 3. **Large Image Files**
- Resize images before sending (recommended: < 2MB)
- API handles JPEG, PNG formats

#### 4. **Slow Predictions**
- First prediction may take 2-3 seconds (model loading)
- Subsequent predictions: ~1 second

### Network Configuration
- **Windows PC IP**: `172.16.45.105`
- **WiFi Network**: Both devices must be on same WiFi
- **Firewall**: Port 5000 must be open

---

## 📊 Model Information

### Supported Breeds (11 Total)

**Buffalo (6):**
- Bhadawari - Heat tolerant, Uttar Pradesh
- Jaffarbadi - Heavy draft work, Gujarat  
- Mehsana - High butterfat, Gujarat
- Murrah - Highest milk yield, Haryana/Punjab
- Nagpuri - Drought tolerant, Maharashtra
- Surti - Rich milk quality, Gujarat

**Cattle (5):**
- Gir - A2 milk production, Gujarat
- Kankrej - Drought resistant, Gujarat/Rajasthan
- Ongole - Heat tolerant, Andhra Pradesh
- Sahiwal - Quality milk, Punjab
- Tharparkar - Desert adapted, Rajasthan

### Model Performance
- **Overall Accuracy**: 87.1%
- **Architecture**: Enhanced ResNet50 + Smart Fusion
- **Training Data**: 11 breed dataset
- **Prediction Methods**: 3 (Neural Network, Smart Fusion, Ensemble)

---

## 🚀 Production Deployment Tips

1. **Error Handling**: Always check `result.success` before using data
2. **Loading States**: Show progress indicators for better UX
3. **Image Optimization**: Compress images client-side for faster uploads
4. **Caching**: Cache breed information to reduce API calls
5. **Offline Handling**: Detect network issues and show appropriate messages

---

## 📞 Support

If you encounter issues:
1. Check API health endpoint first
2. Verify network connectivity between Mac and Windows PC
3. Check browser console for detailed error messages
4. Test with curl commands to isolate React vs API issues

**API Server Status**: `GET http://172.16.45.105:5000/health`