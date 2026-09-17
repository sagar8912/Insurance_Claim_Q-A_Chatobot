"""FastAPI routing endpoints for Insurance RAG application."""
from typing import Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from .schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeBaseStatusResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    CacheStatsResponse,
    CacheClearResponse,
    TranscriptionResponse,
    VoiceChatResponse,
    VoiceChatProcessingTime,
    NewChatResponse,
    ConversationResponse,
    MessageItem,
)
from ..services.chat_service import get_chat_service
from ..services.speech_service import get_speech_service
from ..cache import get_response_cache
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.routes")

# Root & Health Router
root_router = APIRouter()

# API v1 Router
api_router = APIRouter(prefix="/api/v1")

# Dedicated Voice Router for /api/transcribe and /api/voice-chat
voice_router = APIRouter(prefix="/api")


@root_router.get("/", tags=["Root"])
def root_info():
    """Root status verification."""
    return {"message": "Insurance RAG API is running"}


@root_router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint for liveness and readiness monitoring."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@api_router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Submit insurance question and retrieve answers with source citations",
)
def chat_endpoint(request: ChatRequest):
    """Processes policy question, queries semantic cache / RAG, and returns AI answer with citations."""
    try:
        service = get_chat_service()
        result = service.process_message(request.question, conversation_id=request.conversation_id)
        return ChatResponse(
            success=result.get("success", True),
            answer=result.get("answer"),
            error=result.get("error"),
            sources=result.get("sources", []),
            grounded=result.get("grounded", False),
            cache_hit=result.get("cache_hit", False),
            cache_type=result.get("cache_type"),
            processing_time=result.get("processing_time", 0.0),
            conversation_id=result.get("conversation_id"),
            rewritten_query=result.get("rewritten_query"),
            debug_info=result.get("debug_info") if settings.ENABLE_RAG_DEBUG else None,
        )
    except Exception as e:
        logger.error(f"Error during chat endpoint execution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat query: {str(e)}",
        )


@api_router.post(
    "/conversations/new",
    response_model=NewChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Conversations"],
    summary="Create a new conversation session",
)
def create_new_conversation():
    """Generates a new conversation session ID for clean multi-turn dialogue."""
    service = get_chat_service()
    new_conv_id = service.session_service.get_or_create_session(None)
    return NewChatResponse(
        conversation_id=new_conv_id,
        message="New conversation session created successfully.",
    )


@api_router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Conversations"],
    summary="Fetch conversation history messages",
)
def get_conversation_history(conversation_id: str):
    """Retrieves all messages for a specific conversation session."""
    service = get_chat_service()
    messages_data = service.session_service.get_messages(conversation_id, limit=50)
    return ConversationResponse(
        conversation_id=conversation_id,
        created_at=messages_data[0]["created_at"] if messages_data else "",
        message_count=len(messages_data),
        messages=[
            MessageItem(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"],
            )
            for m in messages_data
        ],
    )


@api_router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    tags=["Conversations"],
    summary="Clear conversation history",
)
def clear_conversation(conversation_id: str):
    """Clears all messages for a conversation while retaining session identifier."""
    service = get_chat_service()
    service.session_service.clear_session(conversation_id)
    return {"status": "success", "message": f"Conversation {conversation_id} cleared."}


@api_router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    tags=["Knowledge Base"],
    summary="Trigger document ingestion from knowledge base directory into ChromaDB",
)
def trigger_ingest(request: IngestRequest = IngestRequest()):
    """Ingests raw policy documents, splits text into chunks, and saves embeddings in ChromaDB."""
    try:
        service = get_chat_service()
        result = service.run_ingestion(reset=request.reset)
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge base ingestion failed: {str(e)}",
        )


@api_router.get(
    "/knowledge-base/status",
    response_model=KnowledgeBaseStatusResponse,
    status_code=status.HTTP_200_OK,
    tags=["Knowledge Base"],
    summary="Retrieve current status of ChromaDB vector store collection",
)
def get_kb_status():
    """Returns collection name, document count, chunk count, and readiness status."""
    try:
        service = get_chat_service()
        status_data = service.get_knowledge_base_status()
        return KnowledgeBaseStatusResponse(**status_data)
    except Exception as e:
        logger.error(f"Failed to fetch KB status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch knowledge base status: {str(e)}",
        )


@api_router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Response Cache"],
    summary="Retrieve real-time metrics of the Semantic Response Cache",
)
def get_cache_statistics():
    """Returns total requests, exact/semantic hits, misses, hit rate, and latency averages."""
    try:
        cache_service = get_response_cache()
        stats = cache_service.get_statistics()
        return CacheStatsResponse(**stats.model_dump())
    except Exception as e:
        logger.error(f"Failed to retrieve cache stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cache statistics: {str(e)}",
        )


@api_router.post(
    "/cache/clear",
    response_model=CacheClearResponse,
    status_code=status.HTTP_200_OK,
    tags=["Response Cache"],
    summary="Clear or invalidate the Semantic Response Cache collection",
)
def clear_response_cache(new_version: str = None):
    """Purges the response cache collection or increments the active version."""
    try:
        cache_service = get_response_cache()
        if new_version:
            result = cache_service.invalidate_cache(new_version=new_version)
        else:
            result = cache_service.clear_cache()
        return CacheClearResponse(
            success=result.get("success", True),
            message=result.get("message", "Cache cleared successfully"),
            deleted_count=result.get("deleted_count", 0),
        )
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear response cache: {str(e)}",
        )


# ==============================================================================
# Speech-to-Text & Voice Endpoints
# ==============================================================================

@voice_router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Speech to Text"],
    summary="Transcribe audio to text using local faster-whisper",
)
@api_router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Speech to Text"],
    summary="Transcribe audio to text using local faster-whisper (v1 alias)",
)
async def transcribe_audio_endpoint(audio: UploadFile = File(...)):
    """Accepts multipart audio file, runs local faster-whisper, and returns text."""
    try:
        content = await audio.read()
        speech_service = get_speech_service()
        result = speech_service.transcribe_audio(
            file_bytes=content,
            filename=audio.filename or "recording.webm",
            content_type=audio.content_type,
        )
        return TranscriptionResponse(
            success=result.get("success", False),
            text=result.get("text"),
            language=result.get("language"),
            duration=result.get("duration"),
            processing_time=result.get("processing_time", 0.0),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /api/transcribe: {str(e)}", exc_info=True)
        return TranscriptionResponse(
            success=False,
            error="Unable to transcribe audio",
            processing_time=0.0,
        )


@voice_router.post(
    "/voice-chat",
    response_model=VoiceChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Speech to Text"],
    summary="Directly submit voice audio and execute RAG pipeline",
)
@api_router.post(
    "/voice-chat",
    response_model=VoiceChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Speech to Text"],
    summary="Directly submit voice audio and execute RAG pipeline (v1 alias)",
)
async def voice_chat_endpoint(
    audio: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
):
    """End-to-end voice query: transcribe audio and execute chat pipeline with caching."""
    try:
        content = await audio.read()
        speech_service = get_speech_service()
        transcribe_res = speech_service.transcribe_audio(
            file_bytes=content,
            filename=audio.filename or "recording.webm",
            content_type=audio.content_type,
        )

        if not transcribe_res.get("success"):
            return VoiceChatResponse(
                success=False,
                error=transcribe_res.get("error", "Unable to transcribe audio"),
                processing_time=None,
            )

        transcribed_text = transcribe_res.get("text", "")
        chat_service = get_chat_service()
        chat_res = chat_service.process_message(transcribed_text, conversation_id=conversation_id)

        transcribe_time = transcribe_res.get("processing_time", 0.0)
        rag_time = chat_res.get("processing_time", 0.0)
        total_time = round(transcribe_time + rag_time, 3)

        return VoiceChatResponse(
            success=chat_res.get("success", True),
            transcription=transcribed_text,
            answer=chat_res.get("answer"),
            sources=chat_res.get("sources", []),
            grounded=chat_res.get("grounded", False),
            cache_hit=chat_res.get("cache_hit", False),
            cache_type=chat_res.get("cache_type"),
            processing_time=VoiceChatProcessingTime(
                transcription=transcribe_time,
                rag=rag_time,
                total=total_time,
            ),
            error=chat_res.get("error"),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /api/voice-chat: {str(e)}", exc_info=True)
        return VoiceChatResponse(
            success=False,
            error="An error occurred while processing your voice question.",
        )
