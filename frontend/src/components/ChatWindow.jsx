import React, { useRef, useEffect, useState } from 'react';
import { 
  ShieldCheck, 
  ArrowDown, 
  Sparkles, 
  HeartPulse, 
  Car, 
  FileText, 
  Clock, 
  AlertTriangle, 
  ShieldAlert,
  HelpCircle,
  ChevronRight
} from 'lucide-react';
import ChatMessage from './ChatMessage';
import LoadingState from './LoadingState';
import ChatInput from './ChatInput';

const SUGGESTION_CATEGORIES = [
  { id: 'all', label: 'All Topics' },
  { id: 'health', label: 'Health & Mediclaim' },
  { id: 'motor', label: 'Motor & Theft' },
  { id: 'claims', label: 'Claims & Process' }
];

const CURATED_QUESTIONS = [
  {
    category: 'health',
    icon: Clock,
    color: '#10B981',
    tag: 'Waiting Periods',
    title: 'Pre-existing diseases waiting period',
    prompt: 'What is the waiting period for pre-existing diseases and illnesses under health policies?'
  },
  {
    category: 'motor',
    icon: Car,
    color: '#3B82F6',
    tag: 'Vehicle Theft',
    title: 'Comprehensive motor theft coverage',
    prompt: 'Does comprehensive motor insurance cover vehicle theft, and what documents are required?'
  },
  {
    category: 'claims',
    icon: FileText,
    color: '#8B5CF6',
    tag: 'Filing Procedure',
    title: 'How to file an insurance claim step-by-step',
    prompt: 'What is the step-by-step procedure to submit an insurance claim for hospital reimbursement?'
  },
  {
    category: 'health',
    icon: HeartPulse,
    color: '#EC4899',
    tag: 'Maternity',
    title: 'Maternity coverage terms & limits',
    prompt: 'What is the waiting period and maximum coverage limit for maternity benefits?'
  },
  {
    category: 'motor',
    icon: ShieldAlert,
    color: '#F59E0B',
    tag: 'Exclusions',
    title: 'Accidental damage exclusions',
    prompt: 'What specific accidental damages and conditions are excluded from motor claim payouts?'
  },
  {
    category: 'claims',
    icon: AlertTriangle,
    color: '#EF4444',
    tag: 'Policy Lapse',
    title: 'Premium grace period & policy lapse',
    prompt: 'What happens if I miss a premium payment, and what is the grace period before policy lapse?'
  }
];

export default function ChatWindow({ messages, loading, onSendMessage }) {
  const scrollAreaRef = useRef(null);
  const messagesEndRef = useRef(null);
  const [showScrollBottom, setShowScrollBottom] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    // Smooth scroll on incoming messages
    if (scrollAreaRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollAreaRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 200;
      if (isNearBottom || messages.length <= 1) {
        scrollToBottom();
      }
    } else {
      scrollToBottom();
    }
  }, [messages, loading]);

  const handleScroll = () => {
    if (scrollAreaRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollAreaRef.current;
      setShowScrollBottom(scrollHeight - scrollTop - clientHeight > 200);
    }
  };

  const isEmpty = messages.length === 0;

  const filteredQuestions = selectedCategory === 'all'
    ? CURATED_QUESTIONS
    : CURATED_QUESTIONS.filter(q => q.category === selectedCategory);

  return (
    <main className="chat-main-container">
      <div 
        className="chat-scrollable-viewport" 
        ref={scrollAreaRef}
        onScroll={handleScroll}
      >
        {isEmpty ? (
          <div className="welcome-hero-container">
            {/* Hero Badge */}
            <div className="welcome-hero-badge">
              <ShieldCheck size={14} className="badge-shield-icon" />
              <span>Enterprise Insurance Policy Intelligence</span>
            </div>

            {/* Main Hero Header */}
            <h2 className="welcome-hero-title">
              What insurance questions can I verify for you?
            </h2>
            <p className="welcome-hero-subtitle">
              Ground your decisions with instant answers sourced directly from verified health, motor, and claim policy documents.
            </p>

            {/* Category Filter Tabs */}
            <div className="hero-filter-tabs">
              {SUGGESTION_CATEGORIES.map(cat => (
                <button
                  key={cat.id}
                  type="button"
                  className={`filter-tab-btn ${selectedCategory === cat.id ? 'is-active' : ''}`}
                  onClick={() => setSelectedCategory(cat.id)}
                >
                  {cat.label}
                </button>
              ))}
            </div>

            {/* Curated Question Cards */}
            <div className="hero-suggestions-grid">
              {filteredQuestions.map((q, idx) => {
                const Icon = q.icon;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => onSendMessage(q.prompt)}
                    className="hero-suggestion-card"
                    disabled={loading}
                  >
                    <div className="card-top-row">
                      <div className="card-icon-box" style={{ backgroundColor: `${q.color}15`, color: q.color }}>
                        <Icon size={16} />
                      </div>
                      <span className="card-tag-pill">{q.tag}</span>
                    </div>

                    <div className="card-title-text">{q.title}</div>
                    <div className="card-prompt-snippet">{q.prompt}</div>

                    <div className="card-hover-arrow">
                      <span>Ask policy</span>
                      <ChevronRight size={13} />
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="chat-conversation-flow">
            {messages.map((msg) => (
              <ChatMessage 
                key={msg.id} 
                message={msg} 
                onRegenerate={() => onSendMessage(msg.prompt || messages[messages.length - 2]?.content)}
              />
            ))}

            {loading && (
              <div className="message-row assistant">
                <div className="message-bubble assistant">
                  <LoadingState />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} className="scroll-anchor" />
          </div>
        )}
      </div>

      {/* Floating Scroll to Bottom Button (positioned safely above the input dock) */}
      {showScrollBottom && !isEmpty && (
        <button 
          type="button"
          className="floating-scroll-bottom-btn" 
          onClick={scrollToBottom}
          aria-label="Scroll to newest response"
        >
          <ArrowDown size={14} />
          <span>Latest response</span>
        </button>
      )}

      {/* Modern Fixed Input Dock */}
      <ChatInput onSendMessage={onSendMessage} disabled={loading} />
    </main>
  );
}
