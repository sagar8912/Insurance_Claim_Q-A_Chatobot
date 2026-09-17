import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, ArrowUp, CornerDownLeft } from 'lucide-react';
import VoiceRecorder from './VoiceRecorder';

export default function ChatInput({ onSendMessage, disabled }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [text]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!text.trim() || disabled) return;
    onSendMessage(text);
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTranscription = (transcribedText) => {
    if (!transcribedText) return;
    setText((prev) => {
      const trimmedPrev = prev.trim();
      return trimmedPrev ? `${trimmedPrev} ${transcribedText.trim()}` : transcribedText.trim();
    });
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }, 50);
  };

  const isSendActive = text.trim().length > 0 && !disabled;

  return (
    <div className="chat-input-wrapper-outer">
      <div className="chat-input-card">
        <div className="chat-input-field-row">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about coverage limits, exclusions, waiting periods, claim steps..."
            rows={1}
            maxLength={2000}
            disabled={disabled}
            className="chat-textarea-modern"
            aria-label="Ask your insurance policy question"
          />

          <div className="chat-input-actions-dock">
            <VoiceRecorder
              onTranscriptionSuccess={handleTranscription}
              disabled={disabled}
            />

            <button
              type="button"
              onClick={handleSubmit}
              disabled={!isSendActive}
              className={`send-button-elevated ${isSendActive ? 'is-ready' : ''}`}
              title="Send question (Enter)"
              aria-label="Send message"
            >
              <ArrowUp size={16} strokeWidth={2.6} />
            </button>
          </div>
        </div>

        {/* Input Footer with Guidelines and Shortcuts */}
        <div className="chat-input-subfooter">
          <div className="subfooter-grounding-hint">
            <span className="grounding-dot" />
            <span>Answers strictly grounded in indexed policy documents</span>
          </div>

          <div className="subfooter-shortcut-hint">
            <span>Press</span>
            <kbd className="kbd-key">↵ Enter</kbd>
            <span>to ask</span>
          </div>
        </div>
      </div>
    </div>
  );
}
