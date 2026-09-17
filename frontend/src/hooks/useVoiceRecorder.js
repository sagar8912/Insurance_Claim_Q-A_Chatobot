import { useState, useRef, useCallback, useEffect } from 'react';

const MAX_DURATION_SECONDS = 60;

/**
 * Custom React hook for microphone audio recording using the browser MediaRecorder API.
 */
export function useVoiceRecorder({ onRecordingComplete, maxDuration = MAX_DURATION_SECONDS } = {}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const isCancelledRef = useRef(false);

  // Check browser capability
  const isSupported = typeof window !== 'undefined' &&
    !!(navigator?.mediaDevices?.getUserMedia) &&
    !!(window?.MediaRecorder);

  const cleanupStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_) {}
      });
      streamRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const cancelRecording = useCallback(() => {
    isCancelledRef.current = true;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (_) {}
    }
    cleanupStream();
    setIsRecording(false);
    setRecordingSeconds(0);
    audioChunksRef.current = [];
  }, [cleanupStream]);

  const stopRecording = useCallback(() => {
    isCancelledRef.current = false;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (_) {}
    }
    cleanupStream();
    setIsRecording(false);
  }, [cleanupStream]);

  const startRecording = useCallback(async () => {
    if (!isSupported) {
      setError('Voice recording is not supported in this browser. Please use Chrome, Edge, or Firefox.');
      return;
    }

    setError(null);
    audioChunksRef.current = [];
    isCancelledRef.current = false;
    setRecordingSeconds(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      // Determine best supported container format
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
        'audio/wav',
      ];
      let selectedMime = '';
      for (const m of mimeTypes) {
        if (MediaRecorder.isTypeSupported(m)) {
          selectedMime = m;
          break;
        }
      }

      const options = selectedMime ? { mimeType: selectedMime } : {};
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        cleanupStream();
        if (isCancelledRef.current) {
          audioChunksRef.current = [];
          setRecordingSeconds(0);
          return;
        }

        const currentMime = recorder.mimeType || selectedMime || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: currentMime });
        audioChunksRef.current = [];
        setRecordingSeconds(0);

        if (audioBlob.size === 0) {
          setError('No audio captured. Please check your microphone and try again.');
          return;
        }

        if (onRecordingComplete) {
          onRecordingComplete(audioBlob);
        }
      };

      recorder.start(250);
      setIsRecording(true);

      // Start duration counter
      const startTime = Date.now();
      timerRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        setRecordingSeconds(elapsed);
        if (elapsed >= maxDuration) {
          stopRecording();
        }
      }, 500);

    } catch (err) {
      cleanupStream();
      setIsRecording(false);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('Microphone permission was denied. Please allow microphone access in your browser settings.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No microphone found. Please connect an audio input device.');
      } else {
        setError(err.message || 'Could not start audio recording.');
      }
    }
  }, [isSupported, maxDuration, onRecordingComplete, cleanupStream, stopRecording]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      cleanupStream();
    };
  }, [cleanupStream]);

  // Format mm:ss
  const formatTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const remainingSecs = secs % 60;
    return `${String(mins).padStart(2, '0')}:${String(remainingSecs).padStart(2, '0')}`;
  };

  return {
    isSupported,
    isRecording,
    recordingSeconds,
    formattedTime: formatTime(recordingSeconds),
    error,
    clearError: () => setError(null),
    startRecording,
    stopRecording,
    cancelRecording,
  };
}
