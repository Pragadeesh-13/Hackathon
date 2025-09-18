import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { getText, getBreedName } from '../services/languageService';

const HomePage = ({ onBack, selectedLanguage, onViewBreedDetails, onShowHistory, onShowFeedback, onShowChatbot }) => {
  const [showCamera, setShowCamera] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const webcamRef = useRef(null);

  // API Configuration
  const API_BASE_URL = 'http://172.16.45.105:5000';

  const handleUploadPhoto = () => {
    console.log('Upload cattle photo clicked');
    // Create file input
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (event) => {
      const file = event.target.files[0];
      if (file) {
        // Check file size (max 10MB as per API)
        if (file.size > 10 * 1024 * 1024) {
          alert(getText('fileSizeError', selectedLanguage));
          return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
          setCapturedImage(e.target.result);
          console.log('Image uploaded:', file.name);
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  };

  const handleTakePhoto = () => {
    console.log('Take photo clicked');
    setCameraError(null);
    setShowCamera(true);
  };

  const stopCamera = () => {
    setShowCamera(false);
    setCameraError(null);
    setIsLoading(false);
  };

  const capturePhoto = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setCapturedImage(imageSrc);
        stopCamera();
        console.log('Photo captured successfully');
      } else {
        alert(getText('captureError', selectedLanguage));
      }
    }
  }, [webcamRef]);

  // Helper function to convert file to base64
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  };

  // Function to send image to API for prediction
  const analyzeImage = async (imageData) => {
    setIsAnalyzing(true);
    setPredictionResult(null);

    try {
      console.log('Sending image to API for analysis...');
      
      // Send to API with correct JSON format
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData, // imageData is already base64
          mode: 'ensemble'  // Use the 87.1% accuracy ensemble
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('API Response:', result);
      
      if (result.success) {
        console.log(`Predicted: ${result.prediction.breed}`);
        console.log(`Confidence: ${(result.prediction.confidence * 100).toFixed(1)}%`);
        console.log(`Type: ${result.prediction.breed_info.type}`);
        
        // Format the result for our UI
        const formattedResult = {
          breed: result.prediction.breed,
          confidence: result.prediction.confidence,
          type: result.prediction.breed_info.type,
          description: result.prediction.breed_info.description || '',
          characteristics: result.prediction.breed_info.characteristics || []
        };
        
        setPredictionResult(formattedResult);
      } else {
        throw new Error(result.error || 'Prediction failed');
      }
      
      setIsAnalyzing(false);
      
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert(`${getText('analysisError', selectedLanguage)}: ${error.message}`);
      setIsAnalyzing(false);
    }
  };

  const handleStart = () => {
    console.log('Start button clicked - Opening camera');
    setCameraError(null);
    setShowCamera(true);
  };

  const handleUploadFromCamera = () => {
    console.log('Upload from camera window clicked');
    // Create file input
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (event) => {
      const file = event.target.files[0];
      if (file) {
        // Check file size (max 10MB as per API)
        if (file.size > 10 * 1024 * 1024) {
          alert(getText('fileSizeError', selectedLanguage));
          return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
          setCapturedImage(e.target.result);
          stopCamera(); // Close camera window
          console.log('Image uploaded from camera window:', file.name);
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  };

  const handleNavigation = (tab) => {
    console.log(`Navigate to ${tab}`);
    // Add navigation functionality here
  };

  const handleViewBreedDetails = (breedData) => {
    console.log('View breed details clicked for:', breedData);
    // Navigate to breed details page
    if (onViewBreedDetails) {
      onViewBreedDetails(breedData);
    }
  };

  return (
    <div className="home-page">
      {/* Camera Modal */}
      {showCamera && (
        <div className="camera-modal">
          <div className="camera-container">
            <div className="camera-header">
              <button className="camera-close-btn" onClick={stopCamera}>
                ✕
              </button>
              <h3>{getText('cameraTitle', selectedLanguage)}</h3>
            </div>
            
            <div className="camera-content">
              {cameraError ? (
                <div className="camera-error">
                  <p>{getText('cameraError', selectedLanguage)}</p>
                  <button className="retry-camera-btn" onClick={() => setCameraError(null)}>
                    {getText('retakePhoto', selectedLanguage)}
                  </button>
                </div>
              ) : (
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  height="100%"
                  width="100%"
                  screenshotFormat="image/jpeg"
                  videoConstraints={{
                    width: 1280,
                    height: 720,
                    facingMode: "environment" // Use back camera
                  }}
                  className="camera-video"
                  onUserMedia={() => {
                    console.log('Camera started successfully');
                    setIsLoading(false);
                  }}
                  onUserMediaError={(error) => {
                    console.error('Camera error:', error);
                    setCameraError('Unable to access camera. Please check permissions.');
                    setIsLoading(false);
                  }}
                />
              )}
            </div>
            
            <div className="camera-controls">
              <button 
                className="capture-btn" 
                onClick={capturePhoto} 
                disabled={!!cameraError}
                title="Take Photo"
              >
                <div className="capture-circle"></div>
              </button>
              
              <button 
                className="upload-btn" 
                onClick={handleUploadFromCamera}
                title="Upload from Gallery"
              >
                📁
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Captured Image Preview */}
      {capturedImage && !showCamera && (
        <div className="image-preview-modal">
          <div className="image-preview-container">
            <div className="image-preview-header">
              <button className="preview-close-btn" onClick={() => {
                setCapturedImage(null);
                setPredictionResult(null);
              }}>
                ✕
              </button>
              <h3>{predictionResult ? getText('analysisResults', selectedLanguage) : getText('capturedImage', selectedLanguage)}</h3>
            </div>
            
            <div className="image-preview-content">
              <img src={capturedImage} alt="Captured cattle" className="preview-image" />
              
              {/* Analysis Loading */}
              {isAnalyzing && (
                <div className="analysis-overlay">
                  <div className="analysis-loading">
                    <div className="loading-spinner"></div>
                    <p>{getText('analyzingBreed', selectedLanguage)}</p>
                    <small>{getText('analysisTime', selectedLanguage)}</small>
                  </div>
                </div>
              )}
              
              {/* Prediction Results */}
              {predictionResult && (
                <div className="prediction-results">
                  <div className="prediction-card">
                    <h4>{getText('breedIdentification', selectedLanguage)}</h4>
                    {predictionResult.breed && (
                      <div className="breed-info">
                        <div className="breed-name">
                          {getBreedName(predictionResult.breed, selectedLanguage)}
                          {selectedLanguage !== 'en' && (
                            <div className="breed-english">({predictionResult.breed})</div>
                          )}
                        </div>
                        {predictionResult.confidence && (
                          <div className="confidence">
                            {getText('confidence', selectedLanguage)}: {(predictionResult.confidence * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>
                    )}
                    
                    {predictionResult.description && (
                      <div className="breed-description">
                        <p>{predictionResult.description}</p>
                      </div>
                    )}
                    
                    {predictionResult.characteristics && (
                      <div className="breed-characteristics">
                        <h5>Characteristics:</h5>
                        <ul>
                          {predictionResult.characteristics.map((char, index) => (
                            <li key={index}>{char}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                  
                  {/* Breed Details Button */}
                  <div className="breed-details-button-container">
                    <button 
                      className="breed-details-btn" 
                      disabled={!predictionResult}
                      onClick={() => {
                        if (!predictionResult) {
                          alert('Please upload and analyze an image first!');
                          return;
                        }
                        handleViewBreedDetails(predictionResult);
                      }}
                    >
                      View Breed Details
                    </button>
                  </div>
                </div>
              )}
            </div>
            
            <div className="image-preview-controls">
              <button className="retake-btn" onClick={() => {
                setCapturedImage(null);
                setPredictionResult(null);
                setShowCamera(true);
              }}>
                {getText('retakePhoto', selectedLanguage)}
              </button>
              
              {!predictionResult && !isAnalyzing && (
                <button className="analyze-btn" onClick={() => analyzeImage(capturedImage)}>
                  {getText('analyzeBreed', selectedLanguage)}
                </button>
              )}
              
              {predictionResult && (
                <button className="save-result-btn" onClick={() => {
                  console.log('Saving analysis result:', predictionResult);
                  // Add save functionality here
                  alert(getText('saveSuccess', selectedLanguage));
                  setCapturedImage(null);
                  setPredictionResult(null);
                }}>
                  Save Results
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Subtle background decorations */}
      <div className="subtle-bg-circle-1"></div>
      <div className="subtle-bg-circle-2"></div>
      
      {/* Header with back button and title */}
      <div className="home-header">
        <div className="home-top-left-corner"></div>
        <button className="home-back-button" onClick={onBack}>
          <span className="home-back-arrow">←</span>
        </button>
        <h1 className="home-title">{getText('welcome', selectedLanguage)}</h1>
      </div>

      {/* Main content area */}
      <div className="home-content">
        {/* Photo options section */}
        <div className="photo-options">
          <div className="photo-option" onClick={handleUploadPhoto}>
            <div className="photo-icon upload-icon">
              <div className="folder-icon"></div>
            </div>
            <span className="photo-label">{getText('uploadPhoto', selectedLanguage)}</span>
          </div>

          <div className="photo-option" onClick={handleTakePhoto}>
            <div className="photo-icon camera-icon">
              <div className="camera-body">
                <div className="camera-lens"></div>
              </div>
            </div>
                        <span className="photo-label">{getText('takePhoto', selectedLanguage)}</span>
          </div>
        </div>

        {/* Process arrows */}
        <div className="process-arrows">
          <div className="arrow-down">↓</div>
          <div className="arrow-down">↓</div>
        </div>

        {/* Solution section */}
        <div className="solution-section">
          <div className="solution-icon">
            <div className="checklist-icon">
              <div className="checklist-lines"></div>
              <div className="checkmark">✓</div>
            </div>
          </div>
          <div className="solution-text">
            <span>{getText('breedIdentification', selectedLanguage)}</span>
          </div>
        </div>

        {/* Final arrow */}
        <div className="final-arrow">
          <div className="arrow-down">↓</div>
        </div>

        {/* Start button */}
        <button className="start-button" onClick={handleStart}>
          {getText('start', selectedLanguage)}
        </button>
      </div>

      {/* Bottom navigation */}
      <div className="bottom-navigation">
        <div className="nav-item active" onClick={() => {}}>
          <div className="nav-icon home-nav-icon">🏠</div>
          <span className="nav-label">{getText('welcome', selectedLanguage).split(' ').slice(-2).join(' ')}</span>
        </div>
        <div className="nav-item" onClick={() => {}}>
          <div className="nav-icon breed-icon">🐄</div>
          <span className="nav-label">{getText('breedDetails', selectedLanguage)}</span>
        </div>
        <div className="nav-item" onClick={() => (typeof onShowHistory === 'function' ? onShowHistory() : undefined)}>
          <div className="nav-icon history-icon">🔍</div>
          <span className="nav-label">{getText('history', selectedLanguage)}</span>
        </div>
        <div className="nav-item" onClick={() => (typeof onShowFeedback === 'function' ? onShowFeedback() : undefined)}>
          <div className="nav-icon feedback-icon">📝</div>
          <span className="nav-label">Feedback</span>
        </div>
        <div className="nav-item" onClick={() => (typeof onShowChatbot === 'function' ? onShowChatbot() : undefined)}>
          <div className="nav-icon chatbot-icon">💬</div>
          <span className="nav-label">Chatbot</span>
        </div>
      </div>

      {/* Bottom decorative corners */}
      <div className="home-bottom-corners">
        <div className="home-bottom-left-corner"></div>
        <div className="home-bottom-right-corner"></div>
      </div>
    </div>
  );
};

export default HomePage;