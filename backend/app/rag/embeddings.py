"""Embedding service using HuggingFace Sentence Transformers."""
import threading
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.embeddings")


class EmbeddingService:
    """Thread-safe Singleton embedding service utilizing HuggingFace Sentence Transformers."""

    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.model_name = settings.EMBEDDING_MODEL
        logger.info(f"Initializing SentenceTransformer embeddings with model: {self.model_name}")
        try:
            # We configure device to CPU by default for portability, automatically uses GPU if configured
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}

            self._embedder: Embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            self._initialized = True
            logger.info("SentenceTransformer embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {str(e)}", exc_info=True)
            raise RuntimeError(f"Could not load embedding model: {e}")

    @property
    def embedder(self) -> Embeddings:
        return self._embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        if not texts:
            return []
        try:
            return self._embedder.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}", exc_info=True)
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty for embedding.")
        try:
            return self._embedder.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}", exc_info=True)
            raise


def get_embedding_service() -> EmbeddingService:
    """Factory helper to obtain the singleton embedding service."""
    return EmbeddingService()
