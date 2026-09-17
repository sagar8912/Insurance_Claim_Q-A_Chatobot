"""Context Budget Manager to prevent LLM payload overflow."""
from typing import List, Dict, Any, Set
from langchain_core.documents import Document

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.context_manager")

class ContextBudgetManager:
    """Manages the size and quality of the context passed to the LLM."""

    def __init__(self):
        self.max_chars = settings.MAX_CONTEXT_CHARS
        self.max_tokens = settings.MAX_CONTEXT_TOKENS
        self.final_top_k = settings.FINAL_TOP_K
        self.max_chunks_per_doc = settings.MAX_CHUNKS_PER_DOCUMENT
        self.debug = settings.ENABLE_RAG_DEBUG

    def estimate_tokens(self, text: str) -> int:
        """Lightweight token approximation."""
        return len(text) // 4

    def apply_budget(
        self, 
        documents: List[Document], 
        char_limit_override: int = None,
        top_k_override: int = None
    ) -> List[Document]:
        """Filter, deduplicate, and enforce budget constraints on retrieved documents."""
        if not documents:
            return []

        # Sort strictly by highest similarity score first
        # Handle cases where score might be missing, defaulting to 0
        sorted_docs = sorted(
            documents, 
            key=lambda d: d.metadata.get("similarity_score", 0.0), 
            reverse=True
        )

        final_docs: List[Document] = []
        seen_chunks: Set[str] = set()
        seen_texts: Set[str] = set()
        doc_counts: Dict[str, int] = {}
        
        current_chars = 0
        current_tokens = 0
        
        limit_chars = char_limit_override if char_limit_override else self.max_chars
        limit_top_k = top_k_override if top_k_override else self.final_top_k

        for doc in sorted_docs:
            if len(final_docs) >= limit_top_k:
                break

            content = doc.page_content.strip()
            chunk_id = doc.metadata.get("chunk_id", "")
            filename = doc.metadata.get("filename", "unknown")

            # 1. Exact text duplicate removal
            if content in seen_texts:
                if self.debug:
                    logger.debug(f"Discarding duplicate text from {filename}")
                continue
                
            # 2. Chunk ID duplicate removal
            if chunk_id and chunk_id in seen_chunks:
                if self.debug:
                    logger.debug(f"Discarding duplicate chunk_id {chunk_id}")
                continue

            # 3. Limit excessive chunks from one source
            if doc_counts.get(filename, 0) >= self.max_chunks_per_doc:
                if self.debug:
                    logger.debug(f"Discarding chunk from {filename} (hit MAX_CHUNKS_PER_DOCUMENT={self.max_chunks_per_doc})")
                continue

            doc_chars = len(content)
            doc_tokens = self.estimate_tokens(content)

            # 4. Enforce Character / Token Budget
            if current_chars + doc_chars > limit_chars or current_tokens + doc_tokens > self.max_tokens:
                if self.debug:
                    logger.debug(
                        f"Budget exceeded adding {chunk_id}. "
                        f"Current: {current_chars}c/{current_tokens}t. "
                        f"Attempted: +{doc_chars}c/+{doc_tokens}t"
                    )
                # We skip adding this one, but continue to see if a smaller chunk fits?
                # Usually we just break to prioritize the best context.
                # Let's break to maintain strict relevance order.
                break

            # Accept chunk
            final_docs.append(doc)
            seen_texts.add(content)
            if chunk_id:
                seen_chunks.add(chunk_id)
            doc_counts[filename] = doc_counts.get(filename, 0) + 1
            
            current_chars += doc_chars
            current_tokens += doc_tokens

        if self.debug:
            logger.info(
                f"RAG CONTEXT METRICS: Retrieved candidates: {len(documents)} | "
                f"Final chunks: {len(final_docs)} | "
                f"Context characters: {current_chars} | "
                f"Estimated tokens: {current_tokens} | "
                f"Context limit: {limit_chars} characters. Payload status: SAFE"
            )

        return final_docs
