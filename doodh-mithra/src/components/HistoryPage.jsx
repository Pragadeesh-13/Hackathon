import React from 'react';
import { getText } from '../services/languageService';

const HistoryPage = ({ history, onBack, selectedLanguage = 'english' }) => {
  return (
    <div className="history-page">
      <div className="history-header">
        <button className="history-back-btn" onClick={onBack}>←</button>
        <h2 className="history-title">{getText('historyTitle', selectedLanguage) || 'Tried Images History'}</h2>
      </div>
      {history && history.length > 0 ? (
        <ul className="history-list">
          {history.map((item, idx) => (
            <li key={idx} className="history-item">
              <img src={item.image} alt={`Tried ${idx + 1}`} className="history-img" />
              <div className="history-details">
                <div><strong>{getText('breed', selectedLanguage) || 'Breed'}:</strong> {item.breed}</div>
                <div><strong>{getText('confidence', selectedLanguage) || 'Confidence'}:</strong> {(item.confidence * 100).toFixed(1)}%</div>
                <div><strong>{getText('date', selectedLanguage) || 'Date'}:</strong> {new Date(item.timestamp).toLocaleString()}</div>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p>{getText('noTriedImages', selectedLanguage) || 'No tried images yet.'}</p>
      )}
    </div>
  );
};

export default HistoryPage;
