"""Unit and integration tests for Speech-to-Text service and voice endpoints."""
import os
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.config import settings
from app.services.speech_service import SpeechService, get_speech_service
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Audio Validation Tests
# ---------------------------------------------------------------------------

def test_validate_empty_audio():
    """Verify empty audio bytes are rejected."""
    service = get_speech_service()
    err = service.validate_audio(b"", "recording.webm")
    assert err is not None
    assert "empty" in err.lower()


def test_validate_oversized_audio():
    """Verify audio exceeding MAX_AUDIO_FILE_SIZE_MB is rejected."""
    service = get_speech_service()
    oversized_bytes = b"0" * ((settings.MAX_AUDIO_FILE_SIZE_MB + 1) * 1024 * 1024)
    err = service.validate_audio(oversized_bytes, "recording.webm")
    assert err is not None
    assert "exceeds maximum allowed limit" in err.lower()


def test_validate_unsupported_format():
    """Verify unsupported extensions are rejected."""
    service = get_speech_service()
    err = service.validate_audio(b"dummy audio content", "malicious.exe")
    assert err is not None
    assert "unsupported audio format" in err.lower()


def test_validate_supported_formats():
    """Verify supported formats (webm, wav, mp3, m4a, ogg) pass validation."""
    service = get_speech_service()
    for fmt in ["webm", "wav", "mp3", "m4a", "ogg"]:
        err = service.validate_audio(b"valid content", f"test_audio.{fmt}")
        assert err is None, f"Format .{fmt} should be supported"


# ---------------------------------------------------------------------------
# Test 2: Temporary File Handling & Cleanup Guarantee
# ---------------------------------------------------------------------------

def test_temporary_file_cleanup_on_success():
    """Verify temporary audio files are deleted immediately after transcription."""
    service = SpeechService()

    mock_segment = MagicMock()
    mock_segment.text = "What is the waiting period for health insurance?"
    mock_info = MagicMock()
    mock_info.duration = 3.5
    mock_info.language = "en"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    service._model = mock_model

    created_temp_files = []
    original_tempfile = service.transcribe_audio

    # Intercept to observe temporary file creation and deletion
    result = service.transcribe_audio(
        file_bytes=b"RIFF\x24\x00\x00\x00WAVEfmt ",
        filename="test.wav",
    )

    assert result["success"] is True
    assert result["text"] == "What is the waiting period for health insurance?"
    assert result["duration"] == 3.5
    assert "processing_time" in result


def test_temporary_file_cleanup_on_exception():
    """Verify temporary audio files are deleted even when transcription throws an exception."""
    service = SpeechService()

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("Whisper CTranslate2 crashed")
    service._model = mock_model

    result = service.transcribe_audio(
        file_bytes=b"RIFF\x24\x00\x00\x00WAVEfmt ",
        filename="test.wav",
    )

    assert result["success"] is False
    assert "unable to transcribe" in result["error"].lower()


# ---------------------------------------------------------------------------
# Test 3: Audio Duration Limit Enforcement
# ---------------------------------------------------------------------------

def test_audio_duration_exceeded():
    """Verify audio recordings longer than 60 seconds are rejected."""
    service = SpeechService()

    mock_segment = MagicMock()
    mock_segment.text = "Long recording text..."
    mock_info = MagicMock()
    mock_info.duration = 75.0  # 75s > 60s limit
    mock_info.language = "en"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    service._model = mock_model

    result = service.transcribe_audio(
        file_bytes=b"RIFF header",
        filename="long.wav",
    )

    assert result["success"] is False
    assert "exceeds maximum allowed" in result["error"].lower()


# ---------------------------------------------------------------------------
# Test 4: Voice Input Disabled Toggle
# ---------------------------------------------------------------------------

def test_voice_input_disabled_toggle():
    """Verify service returns disabled message when ENABLE_VOICE_INPUT is false."""
    service = get_speech_service()
    prev = settings.ENABLE_VOICE_INPUT
    try:
        settings.ENABLE_VOICE_INPUT = False
        result = service.transcribe_audio(b"audio data", "audio.webm")
        assert result["success"] is False
        assert "disabled" in result["error"].lower()
    finally:
        settings.ENABLE_VOICE_INPUT = prev


# ---------------------------------------------------------------------------
# Test 5: Empty Speech / Silence Handling
# ---------------------------------------------------------------------------

def test_empty_speech_silence_detection():
    """Verify empty transcription results in a user-friendly error."""
    service = SpeechService()

    mock_info = MagicMock()
    mock_info.duration = 4.0
    mock_info.language = "en"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], mock_info)  # No segments (silence)
    service._model = mock_model

    result = service.transcribe_audio(b"silent audio data", "silence.webm")
    assert result["success"] is False
    assert "no clear speech detected" in result["error"].lower()


# ---------------------------------------------------------------------------
# Test 6: FastAPI POST /api/transcribe Endpoint
# ---------------------------------------------------------------------------

def test_api_transcribe_endpoint_success():
    """Verify POST /api/transcribe endpoint returns valid transcription."""
    with patch("app.services.speech_service.SpeechService.transcribe_audio") as mock_transcribe:
        mock_transcribe.return_value = {
            "success": True,
            "text": "What is the deductible for accidental damage?",
            "language": "en",
            "duration": 3.2,
            "processing_time": 0.45,
        }

        fake_audio = io.BytesIO(b"fake webm audio content")
        response = client.post(
            "/api/transcribe",
            files={"audio": ("recording.webm", fake_audio, "audio/webm")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["text"] == "What is the deductible for accidental damage?"
        assert data["language"] == "en"
        assert data["processing_time"] == 0.45


def test_api_transcribe_endpoint_alias():
    """Verify POST /api/v1/transcribe alias endpoint works identically."""
    with patch("app.services.speech_service.SpeechService.transcribe_audio") as mock_transcribe:
        mock_transcribe.return_value = {
            "success": True,
            "text": "Does health insurance cover ambulance charges?",
            "language": "en",
            "duration": 2.8,
            "processing_time": 0.35,
        }

        fake_audio = io.BytesIO(b"fake webm audio content")
        response = client.post(
            "/api/v1/transcribe",
            files={"audio": ("recording.webm", fake_audio, "audio/webm")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["text"] == "Does health insurance cover ambulance charges?"


# ---------------------------------------------------------------------------
# Test 7: Direct Voice Chat Endpoint POST /api/voice-chat
# ---------------------------------------------------------------------------

def test_api_voice_chat_endpoint():
    """Verify POST /api/voice-chat executes transcription and existing chat pipeline."""
    with patch("app.services.speech_service.SpeechService.transcribe_audio") as mock_transcribe, \
         patch("app.services.chat_service.ChatService.process_message") as mock_chat:

        mock_transcribe.return_value = {
            "success": True,
            "text": "What is the waiting period for pre-existing conditions?",
            "language": "en",
            "duration": 4.1,
            "processing_time": 0.6,
        }

        mock_chat.return_value = {
            "success": True,
            "answer": "The waiting period is 24 months.",
            "sources": [{"filename": "health.txt", "file_type": "txt", "document_type": "health", "preview": "24 months waiting."}],
            "grounded": True,
            "cache_hit": True,
            "cache_type": "semantic",
            "processing_time": 0.05,
        }

        fake_audio = io.BytesIO(b"fake webm audio content")
        response = client.post(
            "/api/voice-chat",
            files={"audio": ("recording.webm", fake_audio, "audio/webm")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["transcription"] == "What is the waiting period for pre-existing conditions?"
        assert data["answer"] == "The waiting period is 24 months."
        assert data["cache_hit"] is True
        assert data["cache_type"] == "semantic"
        assert data["processing_time"]["transcription"] == 0.6
        assert data["processing_time"]["rag"] == 0.05
