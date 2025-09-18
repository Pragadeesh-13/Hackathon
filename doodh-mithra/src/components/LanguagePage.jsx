import React from 'react';
import { getText } from '../services/languageService';

const LanguagePage = ({ onBack, onLanguageSelect }) => {
  const languages = [
    { code: 'tamil', name: 'tamil' },
    { code: 'english', name: 'english' },
    { code: 'hindi', name: 'hindi' },
    { code: 'bangla', name: 'bangla' },
    { code: 'telugu', name: 'telugu' },
    { code: 'kannada', name: 'kannada' }
  ];

  const handleLanguageClick = (language) => {
    if (onLanguageSelect) {
      onLanguageSelect(language.code);
    }
  };

  return (
    <div className="language-page">
      {/* Top Left Corner */}
      <div className="language-top-left-corner"></div>
      
      {/* Back Button */}
      <button className="back-button" onClick={onBack}>
        <span className="back-arrow">←</span>
      </button>

      {/* Welcome Text */}
      <div className="welcome-container">
        <h1 className="welcome-text">{getText('appTitle', 'english')}</h1>
      </div>

      {/* Language Selection Text */}
      <div className="language-instruction">
        <h2>{getText('chooseLanguage', 'english')}</h2>
      </div>

      {/* Language Buttons Grid */}
      <div className="language-grid">
        {languages.map((language, index) => (
          <button
            key={language.code}
            className="language-button-option"
            onClick={() => handleLanguageClick(language)}
          >
            {getText(language.name, language.code)}
          </button>
        ))}
      </div>

      {/* Bottom Right Corner */}
      <div className="language-bottom-right-corner"></div>
    </div>
  );
};

export default LanguagePage;