import React from 'react';
import { Menu, Zap, Shield, RotateCcw, FileText, Sparkles } from 'lucide-react';

export default function AppHeader({ onToggleSidebar, onClearChat, messageCount }) {
  return (
    <header className="app-header">
      <div className="header-left">
        <button 
          type="button"
          className="mobile-menu-btn" 
          onClick={onToggleSidebar} 
          aria-label="Toggle navigation menu"
        >
          <Menu size={18} />
        </button>

        <div className="header-breadcrumbs">
          <div className="breadcrumb-logo">
            <Shield size={16} className="text-accent" />
          </div>
          <span className="breadcrumb-slash">/</span>
          <span className="breadcrumb-item">Insurance Knowledge Hub</span>
          <span className="breadcrumb-slash">/</span>
          <span className="breadcrumb-active">Policy Q&A Copilot</span>
        </div>
      </div>
      
      <div className="header-right">
        {messageCount > 0 && onClearChat && (
          <button
            type="button"
            onClick={onClearChat}
            className="header-action-btn"
            title="Clear current conversation"
          >
            <RotateCcw size={13} />
            <span>Reset Session</span>
          </button>
        )}

        <div className="header-docs-badge" title="Indexed insurance repository">
          <FileText size={12} />
          <span>76 Policy Documents</span>
        </div>

        <div className="header-status-pill" title="Live connection to Groq RAG Engine">
          <span className="status-pulse-dot" />
          <span className="status-label">Groq RAG Active</span>
        </div>
      </div>
    </header>
  );
}
