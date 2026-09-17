import React from 'react';

export default function StatusIndicator({ status, label }) {
  let statusClass = 'offline';
  if (status === 'online' || status === 'ready' || status === 'healthy') {
    statusClass = 'online';
  } else if (status === 'warning') {
    statusClass = 'warning';
  }

  return (
    <div className="status-row">
      <div className={`status-dot ${statusClass}`}></div>
      <span>{label}</span>
    </div>
  );
}
