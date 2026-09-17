import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Submit an insurance policy question to the RAG backend with conversation ID tracking.
 * @param {string} question
 * @param {string} [conversationId]
 * @returns {Promise<{ answer: string, sources: Array, processing_time: number, conversation_id?: string, rewritten_query?: string }>}
 */
export async function sendMessage(question, conversationId) {
  try {
    const payload = { question };
    if (conversationId) {
      payload.conversation_id = conversationId;
    }
    const response = await client.post('/api/v1/chat', payload);
    return response.data;
  } catch (error) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('Request timed out. The server took too long to respond.');
    } else if (!error.response) {
      throw new Error('Cannot connect to the backend server. Please verify FastAPI is running at ' + API_BASE_URL);
    }
    throw new Error(error.message || 'An unexpected error occurred while communicating with the server.');
  }
}

/**
 * Initialize a new conversation session on the backend.
 */
export async function startNewChat() {
  try {
    const response = await client.post('/api/v1/conversations/new');
    return response.data;
  } catch (error) {
    const fallbackId = (typeof crypto !== 'undefined' && crypto.randomUUID)
      ? crypto.randomUUID()
      : `conv_${Date.now()}`;
    return { conversation_id: fallbackId };
  }
}

/**
 * Check backend health status.
 */
export async function getHealth() {
  try {
    const response = await client.get('/health');
    return response.data;
  } catch (error) {
    return { status: 'offline', error: error.message };
  }
}

/**
 * Retrieve ChromaDB knowledge base metrics.
 */
export async function getKnowledgeBaseStatus() {
  try {
    const response = await client.get('/api/v1/knowledge-base/status');
    return response.data;
  } catch (error) {
    return { database_status: 'offline', document_count: 0, chunk_count: 0, error: error.message };
  }
}

/**
 * Trigger backend document ingestion.
 */
export async function triggerIngestion(reset = false) {
  try {
    const response = await client.post('/api/v1/ingest', { reset });
    return response.data;
  } catch (error) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error(error.message || 'Failed to trigger ingestion.');
  }
}

/**
 * Upload an audio recording Blob to FastAPI for local Whisper transcription.
 * @param {Blob} audioBlob
 * @returns {Promise<{ success: boolean, text?: string, error?: string }>}
 */
export async function transcribeAudio(audioBlob) {
  const formData = new FormData();
  let ext = 'webm';
  if (audioBlob.type) {
    if (audioBlob.type.includes('wav')) ext = 'wav';
    else if (audioBlob.type.includes('mp4') || audioBlob.type.includes('m4a')) ext = 'm4a';
    else if (audioBlob.type.includes('ogg')) ext = 'ogg';
    else if (audioBlob.type.includes('mp3')) ext = 'mp3';
  }
  formData.append('audio', audioBlob, `recording.${ext}`);

  try {
    const response = await client.post('/api/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    } else if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error(error.message || 'Unable to transcribe audio. Please try again.');
  }
}
