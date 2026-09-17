"""Centralized configuration module for Insurance RAG Q&A backend."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent

# Load environment files: prefer backend/.env, fallback to root .env
backend_env = BACKEND_DIR / ".env"
root_env = ROOT_DIR / ".env"
if backend_env.exists():
    load_dotenv(dotenv_path=backend_env, override=False)
elif root_env.exists():
    load_dotenv(dotenv_path=root_env, override=False)
else:
    load_dotenv(override=False)


class Settings:
    """Application settings with environment variable fallback and validation."""

    def __init__(self) -> None:
        self.APP_NAME: str = os.getenv("APP_NAME", "Insurance RAG Q&A API")
        self.APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
        self.DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))
        self.FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

        # Groq LLM Configuration
        self.GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "").strip() or None
        self.GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        # Embeddings & ChromaDB
        self.EMBEDDING_MODEL: str = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.CHROMA_COLLECTION_NAME: str = os.getenv(
            "CHROMA_COLLECTION_NAME", "insurance_knowledge_base"
        )

        # Directory Paths (resolved to absolute paths)
        raw_chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        if os.path.isabs(raw_chroma_dir):
            self.CHROMA_PERSIST_DIRECTORY = Path(raw_chroma_dir).resolve()
        else:
            self.CHROMA_PERSIST_DIRECTORY = (BACKEND_DIR / raw_chroma_dir).resolve()

        raw_doc_dir = os.getenv("DOCUMENT_DIRECTORY", "../insurance_rag_knowledge_base")
        if os.path.isabs(raw_doc_dir):
            self.DOCUMENT_DIRECTORY = Path(raw_doc_dir).resolve()
        else:
            self.DOCUMENT_DIRECTORY = (BACKEND_DIR / raw_doc_dir).resolve()

        # Chunking & Retrieval
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
        self.SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

        # Context Budget Manager
        self.RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "15"))
        self.FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "5"))
        self.MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
        self.MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
        self.MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "2500"))
        self.MAX_CHUNKS_PER_DOCUMENT: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "2"))
        self.MAX_CONTEXT_RETRY: int = int(os.getenv("MAX_CONTEXT_RETRY", "1"))
        self.ENABLE_RAG_DEBUG: bool = os.getenv("ENABLE_RAG_DEBUG", "False").lower() in ("true", "1", "yes")

        # Semantic Response Cache
        self.ENABLE_RESPONSE_CACHE: bool = os.getenv("ENABLE_RESPONSE_CACHE", "True").lower() in ("true", "1", "yes")
        self.SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
        self.CACHE_MAX_RESULTS: int = int(os.getenv("CACHE_MAX_RESULTS", "3"))
        self.CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "168"))
        self.KNOWLEDGE_BASE_VERSION: str = os.getenv("KNOWLEDGE_BASE_VERSION", "v1").strip()
        self.CHROMA_CACHE_COLLECTION_NAME: str = os.getenv(
            "CHROMA_CACHE_COLLECTION_NAME", "insurance_response_cache"
        )

        # Voice Input / Speech-to-Text
        self.ENABLE_VOICE_INPUT: bool = os.getenv("ENABLE_VOICE_INPUT", "True").lower() in ("true", "1", "yes")
        self.WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base").strip()
        self.WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu").strip()
        self.WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip()
        self.WHISPER_LANGUAGE: Optional[str] = os.getenv("WHISPER_LANGUAGE", "en").strip() or None
        self.MAX_AUDIO_DURATION_SECONDS: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "60"))
        self.MAX_AUDIO_FILE_SIZE_MB: int = int(os.getenv("MAX_AUDIO_FILE_SIZE_MB", "10"))
        self.SUPPORTED_AUDIO_FORMATS: list = [
            fmt.strip().lower()
            for fmt in os.getenv("SUPPORTED_AUDIO_FORMATS", "webm,wav,mp3,m4a,ogg").split(",")
            if fmt.strip()
        ]

        # Conversational Memory & Contextualization
        self.MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
        raw_conv_db = os.getenv("CONVERSATION_DB_PATH", "./conversations.db")
        if os.path.isabs(raw_conv_db):
            self.CONVERSATION_DB_PATH: Path = Path(raw_conv_db).resolve()
        else:
            self.CONVERSATION_DB_PATH: Path = (BACKEND_DIR / raw_conv_db).resolve()

    def validate(self) -> None:
        """Validate required configuration and paths."""
        if not self.DOCUMENT_DIRECTORY.exists():
            # Try workspace root fallback
            alt_doc_dir = ROOT_DIR / "insurance_rag_knowledge_base"
            if alt_doc_dir.exists():
                self.DOCUMENT_DIRECTORY = alt_doc_dir

        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0")
        if self.CHUNK_OVERLAP < 0 or self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE")
        if self.TOP_K_RESULTS <= 0:
            raise ValueError("TOP_K_RESULTS must be greater than 0")

    @property
    def has_groq_key(self) -> bool:
        return bool(self.GROQ_API_KEY and len(self.GROQ_API_KEY) > 5)

    def to_safe_dict(self) -> dict:
        """Return configuration dictionary without sensitive credentials."""
        return {
            "APP_NAME": self.APP_NAME,
            "APP_VERSION": self.APP_VERSION,
            "DEBUG": self.DEBUG,
            "HOST": self.HOST,
            "PORT": self.PORT,
            "FRONTEND_URL": self.FRONTEND_URL,
            "GROQ_MODEL": self.GROQ_MODEL,
            "HAS_GROQ_KEY": self.has_groq_key,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "CHROMA_COLLECTION_NAME": self.CHROMA_COLLECTION_NAME,
            "CHROMA_PERSIST_DIRECTORY": str(self.CHROMA_PERSIST_DIRECTORY),
            "DOCUMENT_DIRECTORY": str(self.DOCUMENT_DIRECTORY),
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "CHUNK_OVERLAP": self.CHUNK_OVERLAP,
            "TOP_K_RESULTS": self.TOP_K_RESULTS,
            "SIMILARITY_THRESHOLD": self.SIMILARITY_THRESHOLD,
            "RETRIEVAL_CANDIDATES": self.RETRIEVAL_CANDIDATES,
            "FINAL_TOP_K": self.FINAL_TOP_K,
            "MAX_CONTEXT_CHARS": self.MAX_CONTEXT_CHARS,
            "MAX_CONTEXT_TOKENS": self.MAX_CONTEXT_TOKENS,
            "MAX_CHUNK_CHARS": self.MAX_CHUNK_CHARS,
            "MAX_CHUNKS_PER_DOCUMENT": self.MAX_CHUNKS_PER_DOCUMENT,
            "MAX_CONTEXT_RETRY": self.MAX_CONTEXT_RETRY,
            "ENABLE_RAG_DEBUG": self.ENABLE_RAG_DEBUG,
            "ENABLE_RESPONSE_CACHE": self.ENABLE_RESPONSE_CACHE,
            "SEMANTIC_CACHE_THRESHOLD": self.SEMANTIC_CACHE_THRESHOLD,
            "CACHE_MAX_RESULTS": self.CACHE_MAX_RESULTS,
            "CACHE_TTL_HOURS": self.CACHE_TTL_HOURS,
            "KNOWLEDGE_BASE_VERSION": self.KNOWLEDGE_BASE_VERSION,
            "CHROMA_CACHE_COLLECTION_NAME": self.CHROMA_CACHE_COLLECTION_NAME,
            "ENABLE_VOICE_INPUT": self.ENABLE_VOICE_INPUT,
            "WHISPER_MODEL": self.WHISPER_MODEL,
            "WHISPER_DEVICE": self.WHISPER_DEVICE,
            "WHISPER_COMPUTE_TYPE": self.WHISPER_COMPUTE_TYPE,
            "WHISPER_LANGUAGE": self.WHISPER_LANGUAGE,
            "MAX_AUDIO_DURATION_SECONDS": self.MAX_AUDIO_DURATION_SECONDS,
            "MAX_AUDIO_FILE_SIZE_MB": self.MAX_AUDIO_FILE_SIZE_MB,
            "SUPPORTED_AUDIO_FORMATS": self.SUPPORTED_AUDIO_FORMATS,
            "MAX_HISTORY_MESSAGES": self.MAX_HISTORY_MESSAGES,
            "CONVERSATION_DB_PATH": str(self.CONVERSATION_DB_PATH),
        }


# Global settings singleton
settings = Settings()
settings.validate()
