"""Speech-to-Text service using local faster-whisper model."""
import os
import time
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.speech_service")


class SpeechService:
    """Singleton service for local audio transcription using faster-whisper."""

    _instance: Optional["SpeechService"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "SpeechService":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super(SpeechService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._model = None
        self._load_lock = threading.Lock()
        self._initialized = True

    def is_enabled(self) -> bool:
        """Check if voice input feature is enabled in configuration."""
        return settings.ENABLE_VOICE_INPUT

    def load_model(self) -> None:
        """Load the faster-whisper model once into memory."""
        if not self.is_enabled():
            logger.info("Voice input is disabled in settings (ENABLE_VOICE_INPUT=False). Skipping Whisper load.")
            return

        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            try:
                from faster_whisper import WhisperModel

                logger.info(
                    f"Loading local faster-whisper model: '{settings.WHISPER_MODEL}' "
                    f"(device={settings.WHISPER_DEVICE}, compute_type={settings.WHISPER_COMPUTE_TYPE})..."
                )
                start_time = time.perf_counter()

                self._model = WhisperModel(
                    model_size_or_path=settings.WHISPER_MODEL,
                    device=settings.WHISPER_DEVICE,
                    compute_type=settings.WHISPER_COMPUTE_TYPE,
                )

                load_elapsed = round(time.perf_counter() - start_time, 2)
                logger.info(f"faster-whisper model '{settings.WHISPER_MODEL}' loaded successfully in {load_elapsed}s.")
            except Exception as e:
                logger.error(f"Failed to load faster-whisper model: {str(e)}", exc_info=True)
                self._model = None
                raise RuntimeError(f"Could not initialize Whisper model: {e}")

    def validate_audio(self, file_bytes: bytes, filename: str) -> Optional[str]:
        """Validate audio file properties. Returns error string if invalid, None if valid."""
        if not file_bytes or len(file_bytes) == 0:
            return "Uploaded audio file is empty."

        # File size check
        max_bytes = settings.MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            return (
                f"Audio file size ({len(file_bytes) / (1024 * 1024):.1f}MB) exceeds "
                f"maximum allowed limit of {settings.MAX_AUDIO_FILE_SIZE_MB}MB."
            )

        # Extension check
        ext = Path(filename).suffix.lstrip(".").lower()
        if not ext:
            ext = "webm"  # Default fallback for browser MediaRecorder blobs

        supported = settings.SUPPORTED_AUDIO_FORMATS
        if ext not in supported:
            return f"Unsupported audio format '.{ext}'. Supported formats: {', '.join(supported)}"

        return None

    def transcribe_audio(
        self,
        file_bytes: bytes,
        filename: str = "audio.webm",
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate, temporarily save, and transcribe an audio file using faster-whisper.

        Guarantees deletion of the temporary audio file immediately upon completion.
        """
        if not self.is_enabled():
            return {
                "success": False,
                "error": "Voice input is currently disabled on this server.",
            }

        start_time = time.perf_counter()

        # 1. Validation
        val_error = self.validate_audio(file_bytes, filename)
        if val_error:
            logger.warning(f"Audio validation failed: {val_error}")
            return {"success": False, "error": val_error}

        # 2. Ensure model is ready
        if self._model is None:
            try:
                self.load_model()
            except Exception as e:
                return {
                    "success": False,
                    "error": "Speech recognition service could not be initialized.",
                }

        # 3. Create secure temporary file
        ext = Path(filename).suffix.lstrip(".").lower() or "webm"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        temp_path = temp_file.name

        try:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_file.close()

            logger.info(f"Transcribing audio file '{filename}' ({len(file_bytes)} bytes) using Whisper...")

            # 4. Transcribe with VAD (voice activity detection) filtering
            segments, info = self._model.transcribe(
                temp_path,
                language=settings.WHISPER_LANGUAGE or None,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Assemble transcribed text from segments
            text_segments = []
            for seg in segments:
                text_segments.append(seg.text.strip())

            transcribed_text = " ".join(text_segments).strip()
            elapsed = round(time.perf_counter() - start_time, 2)
            audio_duration = round(getattr(info, "duration", 0.0), 2)

            if audio_duration > settings.MAX_AUDIO_DURATION_SECONDS:
                logger.warning(f"Audio duration {audio_duration}s exceeds maximum {settings.MAX_AUDIO_DURATION_SECONDS}s.")
                return {
                    "success": False,
                    "error": f"Audio duration ({audio_duration}s) exceeds maximum allowed {settings.MAX_AUDIO_DURATION_SECONDS}s.",
                }

            if not transcribed_text:
                logger.info("Transcription completed with empty speech text (silence or inaudible).")
                return {
                    "success": False,
                    "error": "No clear speech detected. Please speak closer to the microphone and try again.",
                    "duration": audio_duration,
                    "processing_time": elapsed,
                }

            logger.info(
                f"Transcription successful in {elapsed}s (audio duration: {audio_duration}s): "
                f"'{transcribed_text[:60]}...'"
            )

            return {
                "success": True,
                "text": transcribed_text,
                "language": getattr(info, "language", settings.WHISPER_LANGUAGE or "en"),
                "duration": audio_duration,
                "processing_time": elapsed,
            }

        except Exception as e:
            logger.error(f"Error during audio transcription: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": "Unable to transcribe audio. Please try speaking again.",
            }
        finally:
            # 5. Security & Privacy: Clean up temporary audio file immediately
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"Deleted temporary audio file '{temp_path}'")
            except Exception as ce:
                logger.warning(f"Could not remove temporary audio file '{temp_path}': {str(ce)}")


_speech_service: Optional[SpeechService] = None


def get_speech_service() -> SpeechService:
    """Singleton getter for SpeechService."""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service
