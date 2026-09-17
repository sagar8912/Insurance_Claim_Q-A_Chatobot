import { useState, useCallback, useEffect } from 'react';
import { sendMessage as apiSendMessage, startNewChat } from '../services/api';

function generateUUID() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'conv_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
}

export function useChat() {
  const [conversationId, setConversationId] = useState(() => {
    try {
      const saved = localStorage.getItem('securelife_conversation_id');
      if (saved) return saved;
    } catch {
      // Ignore localStorage issues
    }
    const freshId = generateUUID();
    try {
      localStorage.setItem('securelife_conversation_id', freshId);
    } catch {
      // Ignore localStorage issues
    }
    return freshId;
  });

  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Sync conversation ID with localStorage
  useEffect(() => {
    if (conversationId) {
      try {
        localStorage.setItem('securelife_conversation_id', conversationId);
      } catch {
        // Ignore localStorage issues
      }
    }
  }, [conversationId]);

  const sendMessage = useCallback(
    async (text) => {
      const trimmed = text?.trim();
      if (!trimmed || loading) return;

      setError(null);

      const userMessageId = `user_${Date.now()}`;
      const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      const userMessage = {
        id: userMessageId,
        role: 'user',
        content: trimmed,
        sources: [],
        timestamp: now,
      };

      setMessages((prev) => [...prev, userMessage]);
      setLoading(true);

      try {
        const response = await apiSendMessage(trimmed, conversationId);

        // Keep conversationId updated if returned by backend
        if (response.conversation_id && response.conversation_id !== conversationId) {
          setConversationId(response.conversation_id);
        }

        if (response.success === false) {
          const errorMsg = {
            id: `assistant_err_${Date.now()}`,
            role: 'assistant',
            content: `⚠️ **Service Notice:** ${response.error || 'Failed to get a valid response from AI.'}`,
            sources: response.sources || [],
            isError: true,
            grounded: false,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          };
          setMessages((prev) => [...prev, errorMsg]);
          setError(response.error);
          return;
        }

        const assistantMessage = {
          id: `assistant_${Date.now()}`,
          role: 'assistant',
          content: response.answer,
          sources: response.sources || [],
          processingTime: response.processing_time,
          grounded: response.grounded,
          cacheHit: response.cache_hit || false,
          cacheType: response.cache_type || null,
          rewrittenQuery: response.rewritten_query || null,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };

        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err) {
        const errorMessage = err.message || 'Failed to get answer from AI.';
        setError(errorMessage);

        // Add assistant error message bubble so user clearly sees what happened
        const assistantErrorMessage = {
          id: `assistant_err_${Date.now()}`,
          role: 'assistant',
          content: `⚠️ **Service Notice:** ${errorMessage}\n\nPlease verify backend connectivity or configuration.`,
          sources: [],
          isError: true,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setMessages((prev) => [...prev, assistantErrorMessage]);
      } finally {
        setLoading(false);
      }
    },
    [conversationId, loading]
  );

  const clearChat = useCallback(async () => {
    setMessages([]);
    setError(null);
    try {
      const res = await startNewChat();
      const newId = res.conversation_id || generateUUID();
      setConversationId(newId);
      localStorage.setItem('securelife_conversation_id', newId);
    } catch {
      const newId = generateUUID();
      setConversationId(newId);
      localStorage.setItem('securelife_conversation_id', newId);
    }
  }, []);

  return {
    conversationId,
    messages,
    loading,
    error,
    sendMessage,
    clearChat,
  };
}
