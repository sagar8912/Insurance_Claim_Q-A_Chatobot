import React, { useState } from 'react';
import { Mic, Square, X, Loader2, Check } from 'lucide-react';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';
import { transcribeAudio } from '../services/api';

/**
 * Voice recording control with audio transcription using faster-whisper.
 */
export default function VoiceRecorder({ onTranscriptionSuccess, disabled }) {
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcriptionError, setTranscriptionError] = useState(null);

  const handleRecordingComplete = async (audioBlob) => {
    setIsTranscribing(true);
    setTranscriptionError(null);

    try {
      const result = await transcribeAudio(audioBlob);
      if (result.success && result.text) {
        onTranscriptionSuccess(result.text);
      } else {
        setTranscriptionError(result.error || "We couldn't understand the audio. Please try again.");
      }
    } catch (err) {
      setTranscriptionError(err.message || 'Speech recognition failed. Please try again.');
    } finally {
      setIsTranscribing(false);
    }
  };

  const {
    isSupported,
    isRecording,
    formattedTime,
    error: recorderError,
    clearError,
    startRecording,
    stopRecording,
    cancelRecording,
  } = useVoiceRecorder({
    onRecordingComplete: handleRecordingComplete,
    maxDuration: 60,
  });

  const activeError = transcriptionError || recorderError;

  const handleDismissError = () => {
    setTranscriptionError(null);
    clearError();
  };

  if (!isSupported) {
    return null; // Gracefully hide if browser has no MediaRecorder
  }

  // 1. Transcribing state
  if (isTranscribing) {
    return (
      <div className="voice-transcribing-pill" role="status" aria-live="polite">
        <Loader2 size={14} className="spin-icon" />
        <span className="transcribing-text">Transcribing your question...</span>
      </div>
    );
  }

  // 2. Live Recording state
  if (isRecording) {
    return (
      <div className="voice-recording-bar" role="region" aria-label="Audio recording in progress">
        <div className="recording-indicator">
          <span className="recording-dot" />
          <span className="recording-time">{formattedTime}</span>
        </div>

        <div className="recording-actions">
          <button
            type="button"
            onClick={stopRecording}
            className="voice-action-btn complete-btn"
            title="Finish and transcribe"
            aria-label="Finish recording and transcribe"
          >
            <Check size={14} strokeWidth={2.5} />
          </button>

          <button
            type="button"
            onClick={cancelRecording}
            className="voice-action-btn cancel-btn"
            title="Cancel recording"
            aria-label="Cancel recording"
          >
            <X size={14} strokeWidth={2.5} />
          </button>
        </div>
      </div>
    );
  }

  // 3. Idle state with optional error banner
  return (
    <div className="voice-recorder-wrapper">
      {activeError && (
        <div className="voice-error-toast" role="alert">
          <span>{activeError}</span>
          <button
            type="button"
            onClick={handleDismissError}
            className="error-dismiss-btn"
            aria-label="Dismiss error"
          >
            <X size={12} />
          </button>
        </div>
      )}

      <button
        type="button"
        onClick={() => {
          handleDismissError();
          startRecording();
        }}
        disabled={disabled || isTranscribing}
        className="voice-mic-btn"
        title="Ask using voice"
        aria-label="Start voice recording"
      >
        <Mic size={16} strokeWidth={2} />
      </button>
    </div>
  );
}
