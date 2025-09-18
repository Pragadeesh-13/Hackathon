import React, { useState, useEffect } from 'react';
import { getBreedDetails } from '../services/geminiService';
import { getText } from '../services/languageService';

const BreedDetailsPage = ({ breedData, onBack, selectedLanguage = 'ta' }) => {
  const [breedInfo, setBreedInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [availableVoices, setAvailableVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [showVoiceSelector, setShowVoiceSelector] = useState(false);

  // Check if speech synthesis is supported and load voices
  useEffect(() => {
    if ('speechSynthesis' in window) {
      setSpeechSupported(true);
      
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        const languageCodes = {
          'ta': ['ta-IN', 'ta'],
          'hi': ['hi-IN', 'hi'],
          'en': ['en-US', 'en-GB', 'en-AU', 'en-IN', 'en'],
          'bn': ['bn-IN', 'bn-BD', 'bn'],
          'te': ['te-IN', 'te'],
          'kn': ['kn-IN', 'kn']
        };
        
        // Filter voices for current language
        const currentLangCodes = languageCodes[selectedLanguage] || ['en-US', 'en'];
        const languageFilteredVoices = voices.filter(voice => 
          currentLangCodes.some(code => voice.lang.startsWith(code))
        );
        
        // Filter to only include voices with "Aman" in the name
        const amanVoices = languageFilteredVoices.filter(voice => 
          voice.name.toLowerCase().includes('aman')
        );
        
        // If no Aman voices for selected language, check all languages for Aman voices
        const finalVoices = amanVoices.length > 0 ? amanVoices : 
          voices.filter(voice => voice.name.toLowerCase().includes('aman'));
        
        setAvailableVoices(finalVoices);
        
        // Auto-select first available voice for the language
        if (finalVoices.length > 0 && !selectedVoice) {
          setSelectedVoice(finalVoices[0]);
        }
      };
      
      // Load voices immediately if available
      loadVoices();
      
      // Load voices when they become available (some browsers load them asynchronously)
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }
    }
  }, [selectedLanguage, selectedVoice]);

  // Text-to-Speech functionality
  const speakText = () => {
    if (!speechSupported || !breedInfo?.information) return;

    // Stop any ongoing speech
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    // Create speech utterance
    const utterance = new SpeechSynthesisUtterance(breedInfo.information);
    
    // Use selected voice or fallback to language-based selection
    if (selectedVoice) {
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang;
    } else {
      // Set language based on selected language as fallback
      const languageCodes = {
        'ta': 'ta-IN',
        'hi': 'hi-IN',
        'en': 'en-US',
        'bn': 'bn-IN',
        'te': 'te-IN',
        'kn': 'kn-IN'
      };
      utterance.lang = languageCodes[selectedLanguage] || 'en-US';
    }
    
    utterance.rate = 0.8; // Slower rate for better comprehension
    utterance.pitch = 1;
    utterance.volume = 1;

    // Event listeners
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    // Start speaking
    window.speechSynthesis.speak(utterance);
  };

  useEffect(() => {
    const fetchBreedDetails = async () => {
      if (breedData?.breed) {
        setLoading(true);
        setError(null);
        
        try {
          console.log('Fetching details for breed:', breedData.breed, 'in language:', selectedLanguage);
          const details = await getBreedDetails(breedData.breed, breedData.type, selectedLanguage);
          setBreedInfo(details);
        } catch (err) {
          console.error('Error fetching breed details:', err);
          setError('Failed to load breed information. Please try again.');
        } finally {
          setLoading(false);
        }
      }
    };

    fetchBreedDetails();
  }, [breedData, selectedLanguage]);

  // Close voice selector when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showVoiceSelector && !event.target.closest('.voice-selector-container')) {
        setShowVoiceSelector(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showVoiceSelector]);

  const renderBreedInformation = () => {
    if (loading) {
      return (
        <div className="loading-container">
          <div className="loading-spinner-large"></div>
          <h3>{getText('loadingBreedInfo', selectedLanguage)}</h3>
          <p>{getText('fetchingInsights', selectedLanguage)}</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <h3>{getText('unableToLoad', selectedLanguage)}</h3>
          <p>{error}</p>
          <button 
            className="retry-btn" 
            onClick={() => window.location.reload()}
          >
            {getText('tryAgain', selectedLanguage)}
          </button>
        </div>
      );
    }

    if (!breedInfo) {
      return (
        <div className="no-data-container">
          <h3>{getText('noBreedInfo', selectedLanguage)}</h3>
          <p>{getText('goBackTryAgain', selectedLanguage)}</p>
        </div>
      );
    }

    return (
      <div className="breed-info-container">
        {/* Breed Header */}
        <div className="breed-header">
          <h2>🐄 {breedInfo.breedName}</h2>
          {breedData?.confidence && (
            <div className="confidence-badge">
              {(breedData.confidence * 100).toFixed(1)}% Match
            </div>
          )}
          {breedData?.type && (
            <div className="breed-type-badge">
              {breedData.type} Breed
            </div>
          )}
        </div>

        {/* AI Generated Content */}
        <div className="breed-content">
          <div className="ai-powered-notice">
            <span className="ai-icon">🤖</span>
            <span>{getText('aiVetInsights', selectedLanguage)}</span>
          </div>
          
          {/* Render structured information */}
          {breedInfo.sections && Object.keys(breedInfo.sections).length > 0 ? (
            <div className="structured-sections">
              {Object.entries(breedInfo.sections).map(([sectionName, content], index) => {
                // Add icons based on section type
                let icon = '📋';
                if (sectionName.toLowerCase().includes('milk') || sectionName.toLowerCase().includes('production')) {
                  icon = '🥛';
                } else if (sectionName.toLowerCase().includes('disease') && sectionName.toLowerCase().includes('resistance')) {
                  icon = '🛡️';
                } else if (sectionName.toLowerCase().includes('health')) {
                  icon = '🏥';
                } else if (sectionName.toLowerCase().includes('breeding')) {
                  icon = '🐄';
                } else if (sectionName.toLowerCase().includes('financial') || sectionName.toLowerCase().includes('economic')) {
                  icon = '💰';
                } else if (sectionName.toLowerCase().includes('smart') || sectionName.toLowerCase().includes('decision')) {
                  icon = '🧠';
                } else if (sectionName.toLowerCase().includes('productivity')) {
                  icon = '📈';
                }
                
                return (
                  <div key={index}>
                    <div className="info-section">
                      <h3>{icon} {sectionName}</h3>
                      <div className="section-content">
                        {content.split('\n').map((paragraph, pIndex) => {
                          const cleanParagraph = paragraph.trim()
                            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                            .replace(/\*\*/g, '')
                            .replace(/\*/g, '•');
                          
                          if (cleanParagraph) {
                            return (
                              <p 
                                key={pIndex} 
                                dangerouslySetInnerHTML={{__html: cleanParagraph}}
                              />
                            );
                          }
                          return null;
                        })}
                      </div>
                    </div>
                    {/* Add divider between sections (not after the last one) */}
                    {index < Object.entries(breedInfo.sections).length - 1 && (
                      <div className="section-divider"></div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            // Fallback to raw information
            <div className="raw-information">
              {breedInfo.information.split('\n').map((line, index) => {
                const trimmedLine = line.trim();
                
                // Clean line of ** symbols and format properly
                const cleanLine = trimmedLine
                  .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Convert **text** to <strong>text</strong>
                  .replace(/\*\*/g, '') // Remove any remaining **
                  .replace(/\*/g, '•'); // Convert * to bullet points
                
                if (cleanLine.startsWith('#')) {
                  return <h2 key={index} dangerouslySetInnerHTML={{__html: cleanLine.replace(/^#+\s*/, '')}} />;
                } else if (cleanLine.includes(':') && cleanLine.length < 50 && !cleanLine.includes('•')) {
                  // Treat as section header
                  return <h3 key={index} dangerouslySetInnerHTML={{__html: cleanLine}} />;
                } else if (cleanLine.startsWith('•')) {
                  return <p key={index} className="bullet-point" dangerouslySetInnerHTML={{__html: cleanLine}} />;
                } else if (cleanLine) {
                  return <p key={index} dangerouslySetInnerHTML={{__html: cleanLine}} />;
                }
                return null;
              })}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="breed-actions">
          <button className="action-btn primary" onClick={() => window.print()}>
            📄 {getText('saveAsPdf', selectedLanguage)}
          </button>
          <button className="action-btn secondary" onClick={() => {
            if (navigator.share) {
              navigator.share({
                title: `${breedInfo.breedName} Breed Information`,
                text: breedInfo.information.substring(0, 200) + '...',
                url: window.location.href
              });
            }
          }}>
            📤 {getText('shareInfo', selectedLanguage)}
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="breed-details-page">
      {/* Header */}
      <div className="breed-details-header">
        <button className="back-button" onClick={onBack}>
          <span className="back-arrow">←</span>
        </button>
        <h2>{getText('breedDetails', selectedLanguage)}</h2>
        {speechSupported && breedInfo && (
          <div className="voice-controls">
            <div className="voice-selector-container">
              <button 
                className="voice-selector-btn"
                onClick={() => setShowVoiceSelector(!showVoiceSelector)}
                title="Select voice"
              >
                🎤
              </button>
              {showVoiceSelector && (
                <div className="voice-dropdown">
                  {availableVoices.length > 0 ? (
                    availableVoices.map((voice, index) => (
                      <div
                        key={index}
                        className={`voice-option ${selectedVoice?.name === voice.name ? 'selected' : ''}`}
                        onClick={() => {
                          setSelectedVoice(voice);
                          setShowVoiceSelector(false);
                        }}
                      >
                        <span className="voice-name">{voice.name}</span>
                        <span className="voice-lang">({voice.lang})</span>
                      </div>
                    ))
                  ) : (
                    <div className="voice-option no-voices">
                      <span className="voice-name">No Aman voices available</span>
                      <span className="voice-lang">Using default voice</span>
                    </div>
                  )}
                </div>
              )}
            </div>
            <button 
              className="speaker-button" 
              onClick={speakText}
              title={isSpeaking ? getText('stopSpeaking', selectedLanguage) : getText('speakText', selectedLanguage)}
            >
              <span className="speaker-icon">
                {isSpeaking ? "🔊" : "🔈"}
              </span>
            </button>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="breed-details-content">
        {renderBreedInformation()}
      </div>
    </div>
  );
};

export default BreedDetailsPage;