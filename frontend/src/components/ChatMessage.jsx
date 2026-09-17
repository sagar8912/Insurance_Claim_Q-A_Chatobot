import React, { useState } from 'react';
import { 
  ShieldCheck, 
  Sparkles, 
  Clock, 
  Zap, 
  CheckCircle2, 
  AlertCircle, 
  BookOpen, 
  ChevronDown, 
  ChevronUp,
  ExternalLink
} from 'lucide-react';
import SourceCard from './SourceCard';
import MessageActions from './MessageActions';

/**
 * Lightweight, robust markdown & insurance structured text parser.
 * Renders headers, numbered steps, bullet points, bold/italic formatting,
 * inline code tags, blockquotes, and tables without external library dependencies.
 */
function MarkdownRenderer({ content }) {
  if (!content) return null;

  const lines = content.split('\n');
  const elements = [];
  let currentList = null;
  let listType = null; // 'bullet' or 'numbered'
  let inTable = false;
  let tableRows = [];

  const flushTable = () => {
    if (tableRows.length > 0) {
      elements.push(
        <div key={`table-${elements.length}`} className="md-table-wrapper">
          <table className="md-table">
            <thead>
              <tr>
                {tableRows[0].map((cell, cIdx) => (
                  <th key={cIdx}>{parseInline(cell)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.slice(1).map((row, rIdx) => (
                <tr key={rIdx}>
                  {row.map((cell, cIdx) => (
                    <td key={cIdx}>{parseInline(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableRows = [];
    }
    inTable = false;
  };

  const flushList = () => {
    if (currentList && currentList.length > 0) {
      if (listType === 'numbered') {
        elements.push(
          <div key={`nlist-${elements.length}`} className="md-numbered-list">
            {currentList.map((item, idx) => (
              <div key={idx} className="md-numbered-item">
                <span className="md-number-badge">{item.number}</span>
                <div className="md-item-body">{item.content}</div>
              </div>
            ))}
          </div>
        );
      } else {
        elements.push(
          <div key={`blist-${elements.length}`} className="md-bullet-list">
            {currentList.map((item, idx) => (
              <div key={idx} className="md-bullet-item">
                <span className="md-bullet-dot" />
                <div className="md-item-body">{item.content}</div>
              </div>
            ))}
          </div>
        );
      }
      currentList = null;
      listType = null;
    }
  };

  const parseInline = (text) => {
    if (!text) return '';
    // Format bold (**text**), inline code (`code`), italic (*text*), and key-value tags (e.g., "Police: ...")
    const segments = [];
    // Match **bold**, `code`, or *italic*
    const regex = /(\*\*.*?\*\*|`.*?`|\*.*?\*)/g;
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        segments.push(text.substring(lastIndex, match.index));
      }
      const token = match[0];
      if (token.startsWith('**') && token.endsWith('**')) {
        segments.push(<strong key={match.index} className="md-bold">{token.slice(2, -2)}</strong>);
      } else if (token.startsWith('`') && token.endsWith('`')) {
        segments.push(<code key={match.index} className="md-code">{token.slice(1, -1)}</code>);
      } else if (token.startsWith('*') && token.endsWith('*')) {
        segments.push(<em key={match.index} className="md-italic">{token.slice(1, -1)}</em>);
      }
      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      segments.push(text.substring(lastIndex));
    }

    return segments.length > 0 ? segments : text;
  };

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const trimmed = rawLine.trim();

    // Table detection: | col1 | col2 |
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      flushList();
      // Skip divider lines like |---|---|
      if (/^\|[-:\s|]+\|$/.test(trimmed)) {
        continue;
      }
      const cells = trimmed
        .split('|')
        .slice(1, -1)
        .map((c) => c.trim());
      tableRows.push(cells);
      inTable = true;
      continue;
    } else if (inTable) {
      flushTable();
    }

    // Blank line
    if (!trimmed) {
      flushList();
      continue;
    }

    // Headings
    if (trimmed.startsWith('#### ')) {
      flushList();
      elements.push(
        <h4 key={`h4-${i}`} className="md-h4">
          {parseInline(trimmed.substring(5))}
        </h4>
      );
      continue;
    }
    if (trimmed.startsWith('### ')) {
      flushList();
      elements.push(
        <h3 key={`h3-${i}`} className="md-h3">
          <span className="md-h3-bar" />
          {parseInline(trimmed.substring(4))}
        </h3>
      );
      continue;
    }
    if (trimmed.startsWith('## ')) {
      flushList();
      elements.push(
        <h2 key={`h2-${i}`} className="md-h2">
          {parseInline(trimmed.substring(3))}
        </h2>
      );
      continue;
    }
    if (trimmed.startsWith('# ')) {
      flushList();
      elements.push(
        <h1 key={`h1-${i}`} className="md-h1">
          {parseInline(trimmed.substring(2))}
        </h1>
      );
      continue;
    }

    // Blockquote: > text
    if (trimmed.startsWith('> ')) {
      flushList();
      elements.push(
        <blockquote key={`quote-${i}`} className="md-blockquote">
          {parseInline(trimmed.substring(2))}
        </blockquote>
      );
      continue;
    }

    // Numbered List: e.g. "1. ", "2. ", "10. "
    const numberedMatch = trimmed.match(/^(\d+)[\.\)]\s+(.*)/);
    if (numberedMatch) {
      if (listType !== 'numbered') {
        flushList();
        listType = 'numbered';
        currentList = [];
      }
      currentList.push({
        number: numberedMatch[1],
        content: parseInline(numberedMatch[2]),
      });
      continue;
    }

    // Bullet List: "- ", "* ", "• "
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
      if (listType !== 'bullet') {
        flushList();
        listType = 'bullet';
        currentList = [];
      }
      const itemText = trimmed.replace(/^([-*•]\s+)/, '');
      currentList.push({
        content: parseInline(itemText),
      });
      continue;
    }

    // Standard paragraph
    flushList();
    elements.push(
      <p key={`p-${i}`} className="md-paragraph">
        {parseInline(trimmed)}
      </p>
    );
  }

  flushList();
  flushTable();

  return <div className="markdown-body">{elements}</div>;
}

export default function ChatMessage({ message, onRegenerate }) {
  const isUser = message.role === 'user';
  const [sourcesOpen, setSourcesOpen] = useState(true);

  if (isUser) {
    return (
      <div className="message-row user" role="article" aria-label="User inquiry">
        <div className="message-bubble user">
          <div className="user-message-content">
            <p className="user-text">{message.content}</p>
          </div>
          <div className="user-message-meta">
            <span className="message-timestamp">{message.timestamp || 'Just now'}</span>
          </div>
        </div>
      </div>
    );
  }

  // Assistant Message (AI Copilot)
  const hasSources = message.sources && message.sources.length > 0;
  const isError = message.isError;
  const isGrounded = message.grounded !== false && !isError;
  const processingSeconds = message.processingTime ? `${message.processingTime}s` : null;

  return (
    <div className={`message-row assistant ${isError ? 'has-error' : ''}`} role="article" aria-label="Insurance Copilot Response">
      <div className="message-bubble assistant">
        
        {/* Header: Identity, Model, Grounding Beacon */}
        <div className="assistant-card-header">
          <div className="assistant-identity-group">
            <div className="assistant-avatar-badge">
              <ShieldCheck size={18} strokeWidth={2.4} />
            </div>
            <div className="assistant-title-group">
              <div className="assistant-name-row">
                <span className="assistant-name">SecureLife AI</span>
                <span className="verified-badge">
                  <Sparkles size={11} /> Verified Policy Intelligence
                </span>
              </div>
              <span className="assistant-sublabel">Grounded Insurance RAG Engine</span>
            </div>
          </div>

          <div className="assistant-metrics-group">
            {message.cacheHit && (
              <span className="metric-pill cache-pill" title="Instant retrieval from verified semantic cache">
                <Zap size={11} className="pulse-icon" /> Cached
              </span>
            )}
            {processingSeconds && (
              <span className="metric-pill latency-pill" title="Response generation latency">
                <Clock size={11} /> {processingSeconds}
              </span>
            )}
            <span className="message-timestamp">{message.timestamp || 'Just now'}</span>
          </div>
        </div>

        {/* Rewritten Query notification if contextual memory rephrased the query */}
        {message.rewrittenQuery && (
          <div className="contextual-rewrite-hint">
            <span className="rewrite-label">Context Resolved:</span>
            <span className="rewrite-query">"{message.rewrittenQuery}"</span>
          </div>
        )}

        {/* Formatted Content Body */}
        <div className="assistant-body">
          <MarkdownRenderer content={message.content} />
        </div>

        {/* Grounding & Source Documents Section */}
        {hasSources && (
          <div className="sources-container">
            <button 
              type="button"
              className="sources-toggle-header"
              onClick={() => setSourcesOpen(!sourcesOpen)}
              aria-expanded={sourcesOpen}
            >
              <div className="sources-toggle-left">
                <BookOpen size={14} className="sources-book-icon" />
                <span className="sources-toggle-title">
                  Referenced Policy Clauses ({message.sources.length})
                </span>
                <span className="sources-grounded-tag">
                  <CheckCircle2 size={12} /> Grounded
                </span>
              </div>
              <div className="sources-toggle-right">
                {sourcesOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </div>
            </button>

            {sourcesOpen && (
              <div className="sources-grid">
                {message.sources.map((src, idx) => (
                  <SourceCard key={idx} source={src} index={idx + 1} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Grounding Status Callout for queries with no sources */}
        {!hasSources && !isError && (
          <div className="unreferenced-notice">
            <AlertCircle size={13} />
            <span>Response formulated based on policy guidelines and general coverage provisions.</span>
          </div>
        )}

        {/* Footer Actions: Copy, Thumbs, Regenerate */}
        <div className="assistant-card-footer">
          <div className="footer-grounding-indicator">
            {isGrounded ? (
              <span className="grounding-badge success">
                <CheckCircle2 size={13} /> Verified Policy Grounded
              </span>
            ) : isError ? (
              <span className="grounding-badge error">
                <AlertCircle size={13} /> Notice: Action Required
              </span>
            ) : null}
          </div>

          <MessageActions 
            content={message.content} 
            onRegenerate={onRegenerate} 
          />
        </div>

      </div>
    </div>
  );
}
