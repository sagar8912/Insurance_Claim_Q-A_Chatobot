"""Pydantic schemas for API requests, responses, and data contracts."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    """User question payload with strict input validation and conversational session tracking."""

    question: Optional[str] = Field(
        None,
        description="The insurance policy question to ask the AI assistant.",
        examples=["What is the waiting period for pre-existing diseases?"],
    )
    message: Optional[str] = Field(
        None,
        description="Alternative field name for user question (compatible with chat API formats).",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Unique UUID identifying the multi-turn conversation session.",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_question_and_message(cls, data: Any) -> Any:
        if isinstance(data, dict):
            q = data.get("question")
            m = data.get("message")
            if (not q or not str(q).strip()) and m and str(m).strip():
                data["question"] = str(m).strip()
            elif q:
                data["question"] = str(q).strip()

            if not data.get("question"):
                raise ValueError("Question cannot be empty or whitespace only.")

            trimmed = str(data["question"]).strip()
            if len(trimmed) < 2:
                raise ValueError("Question must be at least 2 characters long.")
            if len(trimmed) > 1000:
                raise ValueError("Question cannot exceed 1000 characters.")
            data["question"] = trimmed
        return data


class SourceCitation(BaseModel):
    """Structured representation of a retrieved document citation."""

    filename: str = Field(..., description="Name of the source text file.")
    file_type: str = Field("txt", description="File type, e.g., txt or pdf.")
    document_type: str = Field(..., description="Classified category of insurance policy.")
    page: Optional[int] = Field(None, description="Page number for PDF documents.")
    preview: str = Field(..., description="Short text snippet from the retrieved document chunk.")
    similarity_score: Optional[float] = Field(None, description="Normalized similarity score.")


class ChatResponse(BaseModel):
    """AI Assistant response with cited policy documents, conversational context, and caching metrics."""

    success: bool = Field(True, description="Whether the request was successful.")
    answer: Optional[str] = Field(None, description="Generated insurance policy answer.")
    error: Optional[str] = Field(None, description="Error message if success is false.")
    sources: List[SourceCitation] = Field(default_factory=list, description="List of source policy files.")
    grounded: bool = Field(False, description="Whether the answer was successfully grounded in context.")
    cache_hit: bool = Field(False, description="Whether response was served from cache.")
    cache_type: Optional[str] = Field(None, description="Cache type: 'exact' or 'semantic' or None.")
    processing_time: float = Field(..., description="Query processing duration in seconds.")
    conversation_id: Optional[str] = Field(None, description="Unique conversation session UUID.")
    rewritten_query: Optional[str] = Field(None, description="Rewritten standalone query if follow-up contextualization was applied.")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Internal debug details if ENABLE_RAG_DEBUG is true.")


class MessageItem(BaseModel):
    """Individual conversational message turn."""

    id: str
    role: str
    content: str
    created_at: str


class ConversationResponse(BaseModel):
    """Structured full conversation history."""

    conversation_id: str
    created_at: str
    updated_at: Optional[str] = None
    message_count: int = 0
    messages: List[MessageItem] = Field(default_factory=list)


class NewChatResponse(BaseModel):
    """Response returned when initiating a fresh conversation session."""

    conversation_id: str
    message: str = "New conversation session created successfully."


class CacheStatsResponse(BaseModel):
    """Real-time performance metrics of the response cache."""

    total_requests: int
    exact_cache_hits: int
    semantic_cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    estimated_groq_calls_saved: int
    total_cached_entries: int
    average_cached_response_time: float
    average_rag_response_time: float
    knowledge_base_version: str


class CacheClearResponse(BaseModel):
    """Response when clearing response cache."""

    success: bool
    message: str
    deleted_count: Optional[int] = 0


class KnowledgeBaseStatusResponse(BaseModel):
    """Real-time status metrics of the ChromaDB knowledge base."""

    database_status: str = Field(..., description="Current status: 'ready', 'empty', or 'error'.")
    collection_name: str = Field(..., description="Chroma collection name.")
    document_count: int = Field(..., description="Number of distinct documents indexed.")
    txt_files: int = Field(0, description="Number of text documents indexed.")
    pdf_files: int = Field(0, description="Number of PDF documents indexed.")
    total_files: int = Field(0, description="Total number of documents indexed.")
    chunk_count: int = Field(..., description="Total number of chunks stored in vector database.")
    embedding_model: Optional[str] = None
    groq_model: Optional[str] = None
    has_groq_key: Optional[bool] = None


class IngestRequest(BaseModel):
    """Optional parameters for ingestion trigger."""

    reset: bool = Field(False, description="Whether to clear existing collection before ingestion.")


class IngestResponse(BaseModel):
    """Summary metrics of the ingestion run."""

    status: str
    message: str
    documents_loaded: int
    chunks_generated: int
    chunks_added: int
    total_chunks_in_db: int
    total_documents_in_db: Optional[int] = None


class HealthResponse(BaseModel):
    """Service health response."""

    status: str = Field("healthy", description="Service health state.")
    app_name: str
    version: str


class TranscriptionResponse(BaseModel):
    """Audio transcription response."""

    success: bool = Field(..., description="Whether transcription succeeded.")
    text: Optional[str] = Field(None, description="Transcribed question text.")
    language: Optional[str] = Field(None, description="Detected or configured spoken language.")
    duration: Optional[float] = Field(None, description="Audio length in seconds.")
    processing_time: float = Field(..., description="Transcription latency in seconds.")
    error: Optional[str] = Field(None, description="User-friendly error message if failed.")


class VoiceChatProcessingTime(BaseModel):
    transcription: float
    rag: float
    total: float


class VoiceChatResponse(BaseModel):
    """Direct speech-to-answer response."""

    success: bool = Field(..., description="Whether voice chat request succeeded.")
    transcription: Optional[str] = Field(None, description="Transcribed question text.")
    answer: Optional[str] = Field(None, description="RAG generated insurance answer.")
    sources: List[SourceCitation] = Field(default_factory=list, description="Cited policy files.")
    grounded: bool = Field(False, description="Grounding status.")
    cache_hit: bool = Field(False, description="Response cache hit status.")
    cache_type: Optional[str] = Field(None, description="exact or semantic cache hit.")
    processing_time: Optional[VoiceChatProcessingTime] = None
    error: Optional[str] = Field(None, description="Error message if failed.")
