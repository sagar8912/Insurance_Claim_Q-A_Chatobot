import React, { useState } from 'react';
import { 
  FileText, 
  FileCode, 
  ChevronRight, 
  ChevronDown, 
  Copy, 
  Check, 
  Bookmark, 
  Layers
} from 'lucide-react';

export default function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const isPdf = source.file_type?.toLowerCase() === 'pdf' || source.filename?.toLowerCase().endsWith('.pdf');
  const similarityPercent = source.similarity_score ? Math.round(source.similarity_score * 100) : null;
  const pageLabel = source.page ? `Page ${source.page}` : null;
  const docType = source.document_type || (isPdf ? 'PDF Policy' : 'Policy Doc');

  const handleCopySnippet = (e) => {
    e.stopPropagation();
    if (!source.preview) return;
    navigator.clipboard.writeText(source.preview);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className={`source-citation-card ${expanded ? 'is-expanded' : ''}`}>
      {/* Header Pill */}
      <button 
        type="button" 
        className="source-card-header"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <div className="source-card-left">
          <span className="source-index-tag">#{index || 1}</span>
          <div className="source-icon-wrapper">
            {isPdf ? (
              <Layers size={13} className="doc-icon pdf" />
            ) : (
              <FileText size={13} className="doc-icon txt" />
            )}
          </div>
          <span className="source-filename-text" title={source.filename}>
            {source.filename}
          </span>
        </div>

        <div className="source-card-right">
          {pageLabel && <span className="source-page-tag">{pageLabel}</span>}
          {similarityPercent !== null && (
            <span className="source-relevance-chip" title="Semantic retrieval confidence">
              <span className="relevance-dot" />
              {similarityPercent}% match
            </span>
          )}
          <span className="source-expand-icon">
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </span>
        </div>
      </button>

      {/* Expanded Clause Excerpt Panel */}
      {expanded && (
        <div className="source-drawer">
          <div className="source-drawer-top">
            <div className="source-drawer-tags">
              <span className="drawer-type-badge">{docType}</span>
              {source.file_type && (
                <span className="drawer-format-badge">{source.file_type.toUpperCase()}</span>
              )}
              {pageLabel && <span className="drawer-page-badge">{pageLabel}</span>}
            </div>

            <button
              type="button"
              className="copy-snippet-btn"
              onClick={handleCopySnippet}
              title="Copy policy excerpt"
            >
              {copied ? (
                <>
                  <Check size={12} color="#10B981" />
                  <span className="copied-label">Copied</span>
                </>
              ) : (
                <>
                  <Copy size={12} />
                  <span>Copy clause</span>
                </>
              )}
            </button>
          </div>

          <div className="source-excerpt-box">
            <div className="excerpt-quote-mark">“</div>
            <p className="source-excerpt-text">
              {source.preview || 'No clause preview text available.'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
