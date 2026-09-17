import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import AppHeader from './components/AppHeader';
import ChatWindow from './components/ChatWindow';
import { useChat } from './hooks/useChat';
import { getHealth, getKnowledgeBaseStatus } from './services/api';

export default function App() {
  const { messages, loading, sendMessage, clearChat } = useChat();

  const [kbStatus, setKbStatus] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const fetchSystemStatus = useCallback(async () => {
    try {
      const [healthRes, statusRes] = await Promise.all([
        getHealth(),
        getKnowledgeBaseStatus(),
      ]);
      setBackendHealth(healthRes);
      setKbStatus(statusRes);
    } catch (err) {
      console.error('Failed to retrieve system status:', err);
    }
  }, []);

  useEffect(() => {
    fetchSystemStatus();
    // Periodic health check poll every 30 seconds
    const interval = setInterval(fetchSystemStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchSystemStatus]);

  // Global Ctrl+K shortcut to start a new chat
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        clearChat();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [clearChat]);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="app-container">
      <Sidebar
        isOpen={isSidebarOpen}
        onNewChat={() => {
          clearChat();
          setIsSidebarOpen(false); // Close mobile sidebar
        }}
        onSelectQuestion={(q) => {
          sendMessage(q);
          setIsSidebarOpen(false);
        }}
        kbStatus={kbStatus}
        backendHealth={backendHealth}
        messageCount={messages.length}
      />

      <div className="app-main-viewport">
        <AppHeader 
          onToggleSidebar={toggleSidebar} 
          onClearChat={clearChat}
          messageCount={messages.length}
        />
        
        <ChatWindow
          messages={messages}
          loading={loading}
          onSendMessage={sendMessage}
        />
      </div>
    </div>
  );
}
