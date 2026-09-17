import React, { useState } from 'react';
import { Copy, Check, RotateCw, ThumbsUp, ThumbsDown, Share2 } from 'lucide-react';

export default function MessageActions({ content, onRegenerate }) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const handleCopy = () => {
    if (!content) return;
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedback = (type) => {
    setFeedback(prev => prev === type ? null : type);
  };

  return (
    <div className="message-action-toolbar" role="toolbar" aria-label="Message options">
      <button 
        type="button"
        className={`action-icon-btn ${copied ? 'is-success' : ''}`}
        onClick={handleCopy} 
        title={copied ? "Copied to clipboard!" : "Copy response"}
        aria-label="Copy response"
      >
        {copied ? <Check size={14} className="success-icon" /> : <Copy size={14} />}
        {copied && <span className="action-tooltip">Copied!</span>}
      </button>

      {onRegenerate && (
        <button 
          type="button"
          className="action-icon-btn" 
          onClick={onRegenerate} 
          title="Regenerate answer"
          aria-label="Regenerate answer"
        >
          <RotateCw size={14} />
        </button>
      )}

      <div className="action-divider" />

      <button 
        type="button"
        className={`action-icon-btn ${feedback === 'up' ? 'is-active-up' : ''}`}
        onClick={() => handleFeedback('up')} 
        title="Helpful response"
        aria-label="Mark response as helpful"
      >
        <ThumbsUp size={14} />
      </button>

      <button 
        type="button"
        className={`action-icon-btn ${feedback === 'down' ? 'is-active-down' : ''}`}
        onClick={() => handleFeedback('down')} 
        title="Inaccurate or unhelpful"
        aria-label="Mark response as unhelpful"
      >
        <ThumbsDown size={14} />
      </button>
    </div>
  );
}
