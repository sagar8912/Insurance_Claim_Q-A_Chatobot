import React from 'react';
import { Bot, Sparkles } from 'lucide-react';

export default function LoadingIndicator() {
  return (
    <div className="message-wrapper assistant-wrapper loading-wrapper">
      <div className="avatar bot-avatar">
        <Bot size={18} />
      </div>
      <div className="message-bubble assistant-bubble loading-bubble">
        <div className="loading-header">
          <Sparkles size={14} className="spin-icon" />
          <span>Analyzing insurance policy knowledge base...</span>
        </div>
        <div className="typing-dots">
          <span className="dot"></span>
          <span className="dot"></span>
          <span className="dot"></span>
        </div>
      </div>
    </div>
  );
}
