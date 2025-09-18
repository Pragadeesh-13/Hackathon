import React, { useState } from 'react';
import LanguageSelector from './components/LanguageSelector';
import LanguagePage from './components/LanguagePage';
import HomePage from './components/HomePage';
import BreedDetailsPage from './components/BreedDetailsPage';
import FeedbackForm from './components/FeedbackForm';
import HistoryPage from './components/HistoryPage';
import './App.css';
import Chatbot from './components/Chatbot';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [selectedLanguage, setSelectedLanguage] = useState(null);
  const [breedData, setBreedData] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [lastFeedback, setLastFeedback] = useState(null);
  const [showFeedbackShare, setShowFeedbackShare] = useState(false);
  const [history, setHistory] = useState([]);

  const handleLanguageSelect = () => {
    setCurrentPage('language');
  };

  const handleBackToHome = () => {
    setCurrentPage('home');
  };

  const handleBackToLanguage = () => {
    setCurrentPage('language');
  };

  const handleLanguageChoice = (language) => {
    setSelectedLanguage(language);
    console.log('Selected language:', language);
    setCurrentPage('homepage');
  };

  const handleViewBreedDetails = (data) => {
    setBreedData(data);
    setCurrentPage('breedDetails');
    // Add to history
    setHistory((prev) => [
      ...prev,
      {
        image: data.image || '',
        breed: data.breed || '',
        confidence: data.confidence || 0,
        timestamp: Date.now(),
      },
    ]);
  };
  const handleShowHistory = () => {
    setCurrentPage('history');
  };

  const handleBackToHomePage = () => {
    setCurrentPage('homepage');
  };

  const handleShowFeedbackForm = () => {
    setShowFeedback(true);
    setCurrentPage('feedback');
  };

  const handleFeedbackSubmit = (formData) => {
    setLastFeedback(formData);
    setShowFeedback(false);
    setShowFeedbackShare(true);
    setCurrentPage('homepage');
  };
  // Feedback share modal
  if (showFeedbackShare && lastFeedback) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        background: 'rgba(0,0,0,0.18)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999
      }}>
        <div style={{
          background: '#fff',
          borderRadius: 18,
          boxShadow: '0 4px 24px rgba(34,139,34,0.12)',
          padding: '32px 28px',
          maxWidth: 420,
          width: '90vw',
          textAlign: 'center',
          fontFamily: 'Poppins, sans-serif'
        }}>
          <h2 style={{ color: '#228B22', marginBottom: 18 }}>Your Feedback</h2>
          <div style={{ textAlign: 'left', marginBottom: 18 }}>
            {Object.entries(lastFeedback).map(([key, value]) => (
              <div key={key} style={{ marginBottom: 8 }}>
                <strong style={{ color: '#228B22', fontWeight: 500 }}>{key}:</strong> {Array.isArray(value) ? value.join(', ') : value}
              </div>
            ))}
          </div>
          <button
            style={{
              background: '#228B22',
              color: '#fff',
              border: 'none',
              borderRadius: 10,
              padding: '10px 22px',
              fontWeight: 600,
              fontSize: '1.08rem',
              cursor: 'pointer',
              marginTop: 8
            }}
            onClick={() => setShowFeedbackShare(false)}
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  if (currentPage === 'breedDetails') {
    return (
      <div>
        <BreedDetailsPage 
          breedData={breedData}
          onBack={handleBackToHomePage}
          selectedLanguage={selectedLanguage}
        />
        <div style={{ textAlign: 'center', margin: '2rem' }}>
          <button className="feedback-btn" onClick={handleShowFeedbackForm}>
            Give Feedback
          </button>
        </div>
      </div>
    );
  }

  if (currentPage === 'history') {
    return <HistoryPage history={history} onBack={handleBackToHomePage} selectedLanguage={selectedLanguage} />;
  }

  if (currentPage === 'feedback') {
    return (
      <FeedbackForm onSubmit={handleFeedbackSubmit} selectedLanguage={selectedLanguage} onBack={handleBackToHomePage} />
    );
  }

  if (currentPage === 'chatbot') {
    return <Chatbot selectedLanguage={selectedLanguage} onBack={handleBackToHomePage} />;
  }

  if (currentPage === 'homepage') {
    return (
      <HomePage 
        onBack={handleBackToLanguage}
        selectedLanguage={selectedLanguage}
        onViewBreedDetails={handleViewBreedDetails}
        onShowHistory={handleShowHistory}
        onShowFeedback={handleShowFeedbackForm}
        onShowChatbot={() => setCurrentPage('chatbot')}
      />
    );
  }

  if (currentPage === 'language') {
    return (
      <LanguagePage 
        onBack={handleBackToHome}
        onLanguageSelect={handleLanguageChoice}
      />
    );
  }

  return (
    <div className="app">
      <img 
        src="/src/assets/top_left.png" 
        alt="Top left decoration" 
        className="top-left-image"
      />
      <div className="main-content">
        <LanguageSelector onLanguageSelect={handleLanguageSelect} />
      </div>
    </div>
  );
}

export default App;
