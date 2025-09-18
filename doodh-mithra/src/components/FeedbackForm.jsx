import React, { useState } from 'react';
import { getText } from '../services/languageService';

const initialState = {
  easyToUse: '',
  photoExperience: '',
  correctAnswer: '',
  realBreed: '',
  helpful: '',
  moreInfo: [],
  phoneWorks: '',
  internetProblem: '',
  offlineNeeded: '',
  overall: '',
  recommend: '',
  otherIdeas: '',
};

const infoOptions = [
  'Milk details',
  'Health care',
  'Selling price/market info',
  'Other',
];

const FeedbackForm = ({ onSubmit, selectedLanguage = 'english', onBack }) => {
  const [form, setForm] = useState(initialState);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (type === 'checkbox') {
      setForm((prev) => {
        if (checked) {
          return { ...prev, moreInfo: [...prev.moreInfo, value] };
        } else {
          return { ...prev, moreInfo: prev.moreInfo.filter((v) => v !== value) };
        }
      });
    } else {
      setForm((prev) => ({ ...prev, [name]: value }));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onSubmit) onSubmit(form);
    alert('Thank you for your feedback!');
  };

  return (
    <form className="feedback-form" onSubmit={handleSubmit}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '18px' }}>
        {onBack && (
          <button type="button" className="history-back-btn" style={{ fontSize: '1.7rem', marginRight: '10px', background: 'none', border: 'none', color: '#228B22', cursor: 'pointer', fontWeight: 'bold', padding: 0 }} onClick={onBack}>←</button>
        )}
        <h2 style={{ margin: 0 }}>{getText('feedbackFormTitle', selectedLanguage) || 'Farmer Feedback Form'}</h2>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackEasyToUse', selectedLanguage) || 'Was the app simple to use?'}</label>
        <div className="option-group">
          <label><input type="radio" name="easyToUse" value="Yes" onChange={handleChange} /> {getText('yes', selectedLanguage) || 'Yes'}</label>
          <label><input type="radio" name="easyToUse" value="No" onChange={handleChange} /> {getText('no', selectedLanguage) || 'No'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackPhotoExperience', selectedLanguage) || 'Taking photo and getting answer:'}</label>
        <div className="option-group">
          <label><input type="radio" name="photoExperience" value="Easy" onChange={handleChange} /> 🙂 {getText('easy', selectedLanguage) || 'Easy'}</label>
          <label><input type="radio" name="photoExperience" value="Okay" onChange={handleChange} /> 😐 {getText('okay', selectedLanguage) || 'Okay'}</label>
          <label><input type="radio" name="photoExperience" value="Difficult" onChange={handleChange} /> 🙁 {getText('difficult', selectedLanguage) || 'Difficult'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackCorrectAnswer', selectedLanguage) || 'Did the app tell the right breed of your cow/buffalo?'}</label>
        <div className="option-group">
          <label><input type="radio" name="correctAnswer" value="Yes" onChange={handleChange} /> {getText('yes', selectedLanguage) || 'Yes'}</label>
          <label><input type="radio" name="correctAnswer" value="No" onChange={handleChange} /> {getText('no', selectedLanguage) || 'No'}</label>
          <label><input type="radio" name="correctAnswer" value="Don’t know" onChange={handleChange} /> {getText('dontKnow', selectedLanguage) || 'Don’t know'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackRealBreed', selectedLanguage) || 'If wrong – What is the real breed?'}</label>
        <input type="text" name="realBreed" value={form.realBreed} onChange={handleChange} placeholder={getText('feedbackRealBreedPlaceholder', selectedLanguage) || 'Farmer can say name / skip if unsure'} />
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackHelpful', selectedLanguage) || 'Did the app give you useful information?'}</label>
        <div className="option-group">
          <label><input type="radio" name="helpful" value="Yes" onChange={handleChange} /> {getText('yes', selectedLanguage) || 'Yes'}</label>
          <label><input type="radio" name="helpful" value="No" onChange={handleChange} /> {getText('no', selectedLanguage) || 'No'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackMoreInfo', selectedLanguage) || 'Do you want more info like:'}</label>
        <div className="option-group">
          {infoOptions.map((opt) => (
            <label key={opt}>
              <input
                type="checkbox"
                name="moreInfo"
                value={opt}
                checked={form.moreInfo.includes(opt)}
                onChange={handleChange}
              /> {getText(opt.replace(/ /g, ''), selectedLanguage) || opt}
            </label>
          ))}
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackPhoneWorks', selectedLanguage) || 'Did the app work well on your phone?'}</label>
        <div className="option-group">
          <label><input type="radio" name="phoneWorks" value="Yes" onChange={handleChange} /> {getText('yes', selectedLanguage) || 'Yes'}</label>
          <label><input type="radio" name="phoneWorks" value="No" onChange={handleChange} /> {getText('no', selectedLanguage) || 'No'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackInternetProblem', selectedLanguage) || 'Internet was a problem?'}</label>
        <div className="option-group">
          <label><input type="radio" name="internetProblem" value="Yes" onChange={handleChange} /> {getText('yes', selectedLanguage) || 'Yes'}</label>
          <label><input type="radio" name="internetProblem" value="No" onChange={handleChange} /> {getText('no', selectedLanguage) || 'No'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackOfflineNeeded', selectedLanguage) || 'Do you want the app without internet also?'}</label>
        <div className="option-group">
          <label><input type="radio" name="offlineNeeded" value="Yes" onChange={handleChange} /> {getText('yes', selectedLanguage) || 'Yes'}</label>
          <label><input type="radio" name="offlineNeeded" value="No" onChange={handleChange} /> {getText('no', selectedLanguage) || 'No'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackOverall', selectedLanguage) || 'Happy with the app?'}</label>
        <div className="option-group">
          <label><input type="radio" name="overall" value="Happy" onChange={handleChange} /> 🙂 {getText('happy', selectedLanguage) || 'Happy'}</label>
          <label><input type="radio" name="overall" value="Okay" onChange={handleChange} /> 😐 {getText('okay', selectedLanguage) || 'Okay'}</label>
          <label><input type="radio" name="overall" value="Not happy" onChange={handleChange} /> 🙁 {getText('notHappy', selectedLanguage) || 'Not happy'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackRecommend', selectedLanguage) || 'Will you tell your friends/farmers to use this app?'}</label>
        <div className="option-group">
          <label><input type="radio" name="recommend" value="Yes" onChange={handleChange} /> {getText('yes', selectedLanguage) || 'Yes'}</label>
          <label><input type="radio" name="recommend" value="No" onChange={handleChange} /> {getText('no', selectedLanguage) || 'No'}</label>
        </div>
      </div>
      <div className="feedback-section">
        <label>{getText('feedbackOtherIdeas', selectedLanguage) || 'Any other idea you want in this app?'}</label>
        <textarea name="otherIdeas" value={form.otherIdeas} onChange={handleChange} placeholder={getText('feedbackOtherIdeasPlaceholder', selectedLanguage) || 'Farmer speaks, helper notes it down'} />
      </div>
      <button type="submit">{getText('submitFeedback', selectedLanguage) || 'Submit Feedback'}</button>
    </form>
  );
};

export default FeedbackForm;
