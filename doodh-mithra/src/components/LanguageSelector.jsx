import React from 'react';

const LanguageSelector = ({ onLanguageSelect }) => {
  const handleLanguageSelect = () => {
    // This will be expanded to show language options
    if (onLanguageSelect) {
      onLanguageSelect();
    }
  };

  return (
    <button 
      className="language-button" 
      onClick={handleLanguageSelect}
    >
      Select Your Language
    </button>
  );
};

export default LanguageSelector;