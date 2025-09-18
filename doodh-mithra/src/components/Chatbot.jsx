import React, { useState } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';

const GEMINI_API_KEY = 'AIzaSyCougoUVPWcFdqaOFr_UZX9KUFFH_AYMGg';
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

const Chatbot = ({ selectedLanguage = 'english', onBack }) => {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hi! How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    setMessages([...messages, { sender: 'user', text: input }]);
    setLoading(true);
    try {
      const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
      const prompt = `You are a helpful assistant for farmers. Answer in ${selectedLanguage} language.\nUser: ${input}`;
      const result = await model.generateContent(prompt);
      const response = await result.response;
  let text = response.text();
  // Remove markdown formatting like asterisks and triple asterisks
  text = text.replace(/\*{1,3}/g, '').replace(/_/g, '').replace(/`/g, '');
  // Limit to max 3 lines
  const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
  text = lines.slice(0, 3).join(' ');
  setMessages((prev) => [...prev, { sender: 'bot', text }]);
    } catch (err) {
      setMessages((prev) => [...prev, { sender: 'bot', text: 'Sorry, I could not process your request.' }]);
    }
    setInput('');
    setLoading(false);
  };

  return (
    <div className="chatbot-container" style={{
      maxWidth: 440,
      margin: '40px auto',
      background: 'linear-gradient(135deg, #e8f5e9 0%, #f8f8f8 100%)',
      borderRadius: 24,
      boxShadow: '0 4px 24px rgba(34,139,34,0.08)',
      padding: 0,
      position: 'relative',
      minHeight: 780
    }}>
      {/* Header Bar with Back Button */}
      <div style={{
        background: '#228B22',
        color: '#fff',
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        padding: '18px 0 18px 0',
        textAlign: 'center',
        fontSize: '1.5rem',
        fontWeight: 700,
        letterSpacing: 1,
        position: 'relative'
      }}>
        {typeof onBack === 'function' && (
          <button
            onClick={onBack}
            style={{
              position: 'absolute',
              left: 18,
              top: '50%',
              transform: 'translateY(-50%)',
              background: 'rgba(255,255,255,0.85)',
              color: '#228B22',
              border: 'none',
              borderRadius: 8,
              padding: '6px 14px',
              fontWeight: 600,
              fontSize: '1.08rem',
              cursor: 'pointer',
              boxShadow: '0 1px 4px rgba(34,139,34,0.09)'
            }}
          >
            ←
          </button>
        )}
        <span style={{ fontFamily: 'Poppins, sans-serif' }}>🌾 Farmer Chatbot</span>
      </div>
      {/* Messages */}
      <div className="chatbot-messages" style={{
        maxHeight: 620,
        overflowY: 'auto',
        background: 'transparent',
        padding: '40px 18px 20px 18px',
        marginBottom: 0,
        minHeight: 620
      }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{
            display: 'flex',
            justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
            margin: '10px 0'
          }}>
            <span style={{
              background: msg.sender === 'user' ? 'linear-gradient(90deg,#e0f7fa 60%,#b2dfdb 100%)' : 'linear-gradient(90deg,#e8f5e9 60%,#c8e6c9 100%)',
              color: '#228B22',
              borderRadius: msg.sender === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
              padding: '10px 16px',
              maxWidth: '75%',
              fontWeight: msg.sender === 'bot' ? 500 : 400,
              fontSize: '1.08rem',
              boxShadow: '0 2px 8px rgba(34,139,34,0.07)',
              fontFamily: 'Poppins, sans-serif',
              wordBreak: 'break-word'
            }}>{msg.text}</span>
          </div>
        ))}
        {loading && <div style={{ textAlign: 'center', color: '#228B22', fontWeight: 600, fontSize: '1.1rem' }}>...</div>}
      </div>
      {/* Input Area */}
      <form onSubmit={sendMessage} style={{
        display: 'flex',
        gap: 10,
        padding: '18px',
        borderBottomLeftRadius: 24,
        borderBottomRightRadius: 24,
        background: '#f8f8f8',
        boxShadow: '0 -2px 8px rgba(34,139,34,0.04)'
      }}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type your question..."
          style={{
            flex: 1,
            padding: '12px 14px',
            borderRadius: 12,
            border: '1.5px solid #228B22',
            fontSize: '1.08rem',
            fontFamily: 'Poppins, sans-serif',
            outline: 'none',
            background: '#fff',
            boxShadow: '0 1px 4px rgba(34,139,34,0.04)'
          }}
        />
        <button type="submit" style={{
          background: '#228B22',
          color: '#fff',
          border: 'none',
          borderRadius: 12,
          padding: '12px 22px',
          fontWeight: 600,
          fontSize: '1.08rem',
          cursor: 'pointer',
          fontFamily: 'Poppins, sans-serif',
          boxShadow: '0 2px 8px rgba(34,139,34,0.09)'
        }} disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
};

export default Chatbot;
