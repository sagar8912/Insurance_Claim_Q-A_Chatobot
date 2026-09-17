import React, { useState, useEffect } from 'react';
import { Shield, Sparkles, Search, FileCheck, Cpu } from 'lucide-react';

const STAGES = [
  { icon: Search, label: "Scanning 76 indexed insurance policy documents..." },
  { icon: FileCheck, label: "Matching relevant clauses & waiting period terms..." },
  { icon: Cpu, label: "Grounded reasoning with Groq intelligence engine..." },
  { icon: Sparkles, label: "Finalizing synthesized policy answer..." },
];

export default function LoadingState() {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStage((prev) => (prev + 1) % STAGES.length);
    }, 1800);
    return () => clearInterval(interval);
  }, []);

  const StageIcon = STAGES[currentStage].icon;

  return (
    <div className="copilot-loading-card" role="status" aria-live="polite">
      {/* Top Beacon & Current Step */}
      <div className="copilot-loading-header">
        <div className="loading-beacon-wrapper">
          <div className="beacon-ring" />
          <div className="beacon-core">
            <StageIcon size={14} className="beacon-icon" />
          </div>
        </div>

        <div className="loading-text-group">
          <span className="loading-label">SecureLife AI Copilot</span>
          <span className="loading-status-text">{STAGES[currentStage].label}</span>
        </div>
      </div>

      {/* Modern Shimmer Skeleton Lines */}
      <div className="loading-shimmer-container">
        <div className="shimmer-line line-long" />
        <div className="shimmer-line line-med" />
        <div className="shimmer-line line-short" />
      </div>

      {/* Micro Progress Bar */}
      <div className="loading-progress-bar">
        <div className="loading-progress-fill" style={{ width: `${((currentStage + 1) / STAGES.length) * 100}%` }} />
      </div>
    </div>
  );
}
