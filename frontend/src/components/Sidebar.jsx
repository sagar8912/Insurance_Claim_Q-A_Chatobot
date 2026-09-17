import React, { useState } from 'react';
import { 
  ShieldCheck, 
  Plus, 
  Database, 
  ChevronDown,
  ChevronRight,
  Sparkles,
  Zap,
  Layers,
  FileText,
  Activity,
  Car,
  HeartPulse,
  FileQuestion,
  HelpCircle,
  ExternalLink,
  Cpu
} from 'lucide-react';
import StatusIndicator from './StatusIndicator';

const POLICY_TOPICS = [
  {
    category: "Health & Mediclaim",
    icon: HeartPulse,
    color: "#10B981",
    questions: [
      "What is the waiting period for pre-existing diseases?",
      "Does health insurance cover maternity and childbirth?",
      "What are the room rent limits in mediclaim?",
    ]
  },
  {
    category: "Motor & Auto Insurance",
    icon: Car,
    color: "#3B82F6",
    questions: [
      "Does comprehensive motor insurance cover theft?",
      "What documents are required to file a car accident claim?",
      "Is third-party damage covered under motor policy?",
    ]
  },
  {
    category: "Claims & Settlements",
    icon: FileQuestion,
    color: "#8B5CF6",
    questions: [
      "What is the timeline for insurance claim settlement?",
      "Why do insurance claims get rejected?",
      "How do I apply for cashless hospitalization?",
    ]
  }
];

export default function Sidebar({
  isOpen,
  onNewChat,
  kbStatus,
  backendHealth,
  messageCount,
  onSelectQuestion,
}) {
  const [kbExpanded, setKbExpanded] = useState(false);
  const [activeTopic, setActiveTopic] = useState("Health & Mediclaim");

  const isServerHealthy = backendHealth?.status === 'healthy';
  const isDbReady = kbStatus?.database_status === 'ready';

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      {/* Brand Header */}
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <div className="brand-logo-container">
            <ShieldCheck size={24} className="brand-logo-icon" />
            <div className="brand-logo-glow" />
          </div>
          <div className="brand-text-block">
            <div className="brand-title-row">
              <h1 className="brand-title">SecureLife AI</h1>
              <span className="brand-badge">RAG 2.0</span>
            </div>
            <span className="brand-subtitle">Insurance Intelligence Copilot</span>
          </div>
        </div>

        {/* New Chat Primary Action */}
        <button 
          type="button"
          onClick={onNewChat} 
          className="btn-new-chat-elevated"
          title="Start a new consultation session"
        >
          <div className="btn-new-chat-left">
            <Plus size={16} strokeWidth={2.5} />
            <span>New Consultation</span>
          </div>
          <span className="shortcut-tag">Ctrl K</span>
        </button>
      </div>

      {/* Middle Navigation / Policy Knowledge Domains */}
      <div className="sidebar-content">
        <div className="sidebar-section-header">
          <span className="section-label">Policy Knowledge Domains</span>
        </div>

        <div className="policy-topic-accordion">
          {POLICY_TOPICS.map((topic) => {
            const Icon = topic.icon;
            const isExpanded = activeTopic === topic.category;

            return (
              <div key={topic.category} className="topic-group">
                <button
                  type="button"
                  className={`topic-header-btn ${isExpanded ? 'is-active' : ''}`}
                  onClick={() => setActiveTopic(isExpanded ? null : topic.category)}
                >
                  <div className="topic-title-left">
                    <span className="topic-icon-badge" style={{ backgroundColor: `${topic.color}18`, color: topic.color }}>
                      <Icon size={14} />
                    </span>
                    <span className="topic-name">{topic.category}</span>
                  </div>
                  {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                </button>

                {isExpanded && (
                  <div className="topic-questions-list">
                    {topic.questions.map((q, idx) => (
                      <button
                        key={idx}
                        type="button"
                        className="topic-question-item"
                        onClick={() => onSelectQuestion && onSelectQuestion(q)}
                      >
                        <span className="question-bullet">›</span>
                        <span className="question-text">{q}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Knowledge Base & System Telemetry Card */}
      <div className="sidebar-status-container">
        <div className="kb-telemetry-card">
          <div className="kb-card-header">
            <div className="kb-header-left">
              <Database size={15} className="text-accent" />
              <span className="kb-card-title">Vector Knowledge Store</span>
            </div>
            <span className="kb-docs-badge">
              {kbStatus?.document_count || 76} Docs
            </span>
          </div>

          <div className="kb-metrics-grid">
            <div className="metric-chip">
              <span className="metric-chip-label">Vector DB</span>
              <span className="metric-chip-value">ChromaDB Rust</span>
            </div>
            <div className="metric-chip">
              <span className="metric-chip-label">Model Engine</span>
              <span className="metric-chip-value">Groq Fast LLM</span>
            </div>
          </div>

          <button 
            type="button"
            className="kb-details-toggle" 
            onClick={() => setKbExpanded(!kbExpanded)}
          >
            <span>{kbExpanded ? "Hide indexing metrics" : "View repository breakdown"}</span>
            {kbExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </button>

          {kbExpanded && (
            <div className="kb-details-dropdown">
              <div className="kb-detail-row">
                <span>PDF Policy Files</span>
                <strong className="text-white">{kbStatus?.pdf_files || '18 files'}</strong>
              </div>
              <div className="kb-detail-row">
                <span>TXT Policy Files</span>
                <strong className="text-white">{kbStatus?.txt_files || '58 files'}</strong>
              </div>
              <div className="kb-detail-row">
                <span>Indexed Chunks</span>
                <strong className="text-white">{kbStatus?.chunk_count || '1,420 chunks'}</strong>
              </div>
              <div className="kb-detail-row">
                <span>Semantic Response Cache</span>
                <span className="badge-cache-on">Enabled</span>
              </div>
            </div>
          )}
        </div>

        {/* System Health Indicators */}
        <div className="system-health-rows">
          <div className="health-item">
            <span className={`health-dot ${isServerHealthy ? 'online' : 'error'}`} />
            <span className="health-label">API Gateway</span>
            <span className="health-status">{isServerHealthy ? 'Connected' : 'Reconnecting'}</span>
          </div>

          <div className="health-item">
            <span className={`health-dot ${isDbReady ? 'online' : 'warning'}`} />
            <span className="health-label">Index Sync</span>
            <span className="health-status">{isDbReady ? 'Ready' : 'Standby'}</span>
          </div>
        </div>
      </div>

      {/* Footer Branding */}
      <div className="sidebar-footer">
        <div className="footer-shield-row">
          <ShieldCheck size={12} className="footer-shield-icon" />
          <span>HIPAA & IRDAI Policy Grounded</span>
        </div>
        <div className="footer-subtext">SecureLife Intelligence Hub © 2026</div>
      </div>
    </aside>
  );
}
