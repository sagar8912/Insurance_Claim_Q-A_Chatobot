"""Chat service layer orchestrating Semantic Response Cache, RAG pipeline, and administrative operations."""
import time
from typing import Dict, Any, Optional
from pathlib import Path

from ..config import settings
from ..rag.rag_pipeline import get_rag_pipeline, InsuranceRAGPipeline
from ..rag.vector_store import get_vector_store_manager, ChromaVectorStoreManager
from ..rag.document_loader import InsuranceDocumentLoader
from ..rag.text_splitter import InsuranceTextSplitter
from ..rag.hallucination_checker import get_hallucination_checker, HallucinationChecker
from ..cache import get_response_cache, ResponseCacheService
from .session_service import get_session_service, SessionService
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.chat_service")


class ChatService:
    """Service handling chat interactions, conversational memory, response caching, and administrative operations."""

    def __init__(
        self,
        rag_pipeline: Optional[InsuranceRAGPipeline] = None,
        vector_store_manager: Optional[ChromaVectorStoreManager] = None,
        response_cache: Optional[ResponseCacheService] = None,
        hallucination_checker: Optional[HallucinationChecker] = None,
        session_service: Optional[SessionService] = None,
    ) -> None:
        self.rag_pipeline = rag_pipeline or get_rag_pipeline()
        self.vector_store_manager = vector_store_manager or get_vector_store_manager()
        self.response_cache = response_cache or get_response_cache()
        self.hallucination_checker = hallucination_checker or get_hallucination_checker()
        self.session_service = session_service or get_session_service()

    def process_message(
        self, question: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute question answering through Session Memory -> Cache -> Contextualized RAG -> Validation -> Cache & Session Save."""
        start_time = time.perf_counter()
        self.response_cache.record_request_start()

        # Step 0: Ensure Conversation Session and Fetch History
        conv_id = self.session_service.get_or_create_session(conversation_id)
        history_messages = self.session_service.get_messages(conv_id, limit=settings.MAX_HISTORY_MESSAGES)
        history_text = self.session_service.format_chat_history_for_prompt(
            conv_id, max_messages=settings.MAX_HISTORY_MESSAGES
        )

        logger.info(f"[CHAT REQUEST] conversation_id: {conv_id}")
        logger.info(f"[USER QUERY] original_query: '{question}'")
        logger.info(f"[CHAT HISTORY] number_of_messages: {len(history_messages)}")

        # Step 1: Cache Check (Context-Aware)
        # For first-turn questions, use standard exact and semantic matching.
        # For follow-ups, use conversation-scoped key to avoid context collision.
        cache_query_key = f"{conv_id}::{question}" if history_text else question
        exact_hit = self.response_cache.check_exact_cache(cache_query_key)
        if not exact_hit and not history_text:
            exact_hit = self.response_cache.check_exact_cache(question)

        if exact_hit and exact_hit.is_valid and exact_hit.entry:
            elapsed = round(time.perf_counter() - start_time, 4)
            self.response_cache.record_cached_latency(elapsed)
            entry = exact_hit.entry
            logger.info(f"[CACHE] hit (exact)")

            # Record turn in persistent session storage
            self.session_service.add_message(conv_id, role="user", content=question)
            self.session_service.add_message(conv_id, role="assistant", content=entry.answer)

            debug_info = None
            if settings.ENABLE_RAG_DEBUG:
                debug_info = {
                    "cache_hit": True,
                    "cache_type": "exact",
                    "semantic_similarity": 1.0,
                    "cache_age_seconds": round(entry.age_seconds, 2),
                    "knowledge_base_version": entry.knowledge_base_version,
                }

            return {
                "success": True,
                "answer": entry.answer,
                "sources": entry.sources,
                "grounded": entry.grounded,
                "cache_hit": True,
                "cache_type": "exact",
                "processing_time": elapsed,
                "conversation_id": conv_id,
                "rewritten_query": None,
                "debug_info": debug_info,
            }

        # For first-turn questions without history, check semantic cache
        embedding = None
        if not history_text:
            try:
                normalized_q = self.response_cache.normalize_question(question)
                embedding = self.response_cache.embedding_service.embed_query(normalized_q)
            except Exception as e:
                logger.warning(f"Could not pre-embed query for semantic cache: {str(e)}")

            semantic_hit = self.response_cache.check_semantic_cache(question, embedding=embedding)
            if semantic_hit and semantic_hit.is_valid and semantic_hit.entry:
                elapsed = round(time.perf_counter() - start_time, 4)
                self.response_cache.record_cached_latency(elapsed)
                entry = semantic_hit.entry
                logger.info(f"[CACHE] hit (semantic)")

                # Record turn in persistent session storage
                self.session_service.add_message(conv_id, role="user", content=question)
                self.session_service.add_message(conv_id, role="assistant", content=entry.answer)

                debug_info = None
                if settings.ENABLE_RAG_DEBUG:
                    debug_info = {
                        "cache_hit": True,
                        "cache_type": "semantic",
                        "semantic_similarity": semantic_hit.similarity,
                        "cache_age_seconds": round(entry.age_seconds, 2),
                        "knowledge_base_version": entry.knowledge_base_version,
                    }

                return {
                    "success": True,
                    "answer": entry.answer,
                    "sources": entry.sources,
                    "grounded": entry.grounded,
                    "cache_hit": True,
                    "cache_type": "semantic",
                    "processing_time": elapsed,
                    "conversation_id": conv_id,
                    "rewritten_query": None,
                    "debug_info": debug_info,
                }

        # Step 2: Cache Miss -> Conversational RAG Execution
        self.response_cache.record_cache_miss()
        logger.info("[CACHE] miss")
        logger.info("RAG PIPELINE STARTED: Executing contextualized retrieval and Groq generation.")

        rag_result = self.rag_pipeline.ask(
            question=question,
            conversation_id=conv_id,
            chat_history_str=history_text,
        )
        elapsed = round(time.perf_counter() - start_time, 4)
        self.response_cache.record_rag_latency(elapsed)

        if not rag_result.get("success", False):
            return {
                "success": False,
                "answer": None,
                "error": rag_result.get("error", "Error processing request."),
                "sources": [],
                "grounded": False,
                "cache_hit": False,
                "cache_type": None,
                "processing_time": elapsed,
                "conversation_id": conv_id,
                "rewritten_query": rag_result.get("rewritten_query"),
            }

        answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])
        is_grounded = rag_result.get("grounded", False)
        standalone_q = rag_result.get("standalone_query", question)
        rewritten_q = rag_result.get("rewritten_query")

        # Step 3: Hallucination & Grounding Validation
        raw_docs = self.rag_pipeline.retrieve_context(standalone_q)
        val_result = self.hallucination_checker.validate(answer=answer, documents=raw_docs, sources=sources)
        final_grounded = is_grounded and val_result.is_grounded and not val_result.is_hallucination

        # Step 4: Cache Save (Only when grounded and hallucination-free)
        if val_result.is_cacheable and final_grounded:
            if history_text:
                # Cache under conversation-scoped key for exact repeated follow-up
                self.response_cache.save_response(
                    question=cache_query_key,
                    answer=answer,
                    sources=sources,
                    grounded=True,
                    hallucination_score=val_result.hallucination_score,
                )
                # Also cache under rewritten standalone question if available
                if rewritten_q and rewritten_q != question:
                    self.response_cache.save_response(
                        question=rewritten_q,
                        answer=answer,
                        sources=sources,
                        grounded=True,
                        hallucination_score=val_result.hallucination_score,
                    )
            else:
                self.response_cache.save_response(
                    question=question,
                    answer=answer,
                    sources=sources,
                    grounded=True,
                    hallucination_score=val_result.hallucination_score,
                    embedding=embedding,
                )
        else:
            logger.info(f"Answer NOT cached: {val_result.reason}")

        # Step 5: Save turn to persistent session history
        self.session_service.add_message(conv_id, role="user", content=question)
        self.session_service.add_message(conv_id, role="assistant", content=answer)

        debug_info = None
        if settings.ENABLE_RAG_DEBUG:
            debug_info = {
                "cache_hit": False,
                "cache_type": None,
                "semantic_similarity": 0.0,
                "hallucination_score": val_result.hallucination_score,
                "is_hallucination": val_result.is_hallucination,
                "knowledge_base_version": settings.KNOWLEDGE_BASE_VERSION,
                "rewritten_query": rewritten_q,
            }

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "grounded": final_grounded,
            "cache_hit": False,
            "cache_type": None,
            "processing_time": elapsed,
            "conversation_id": conv_id,
            "rewritten_query": rewritten_q,
            "debug_info": debug_info,
        }

    def get_knowledge_base_status(self) -> Dict[str, Any]:
        """Fetch real-time metrics on the local ChromaDB vector store."""
        try:
            chunk_count = self.vector_store_manager.count_chunks()
            doc_count = self.vector_store_manager.get_indexed_document_count()

            kb_dir = Path(settings.DOCUMENT_DIRECTORY)
            txt_files = len(list(kb_dir.rglob("*.txt"))) if kb_dir.exists() else 0
            pdf_files = len(list(kb_dir.rglob("*.pdf"))) if kb_dir.exists() else 0
            total_files = txt_files + pdf_files

            status = "ready" if chunk_count > 0 else "empty"

            return {
                "database_status": status,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "document_count": doc_count,
                "txt_files": txt_files,
                "pdf_files": pdf_files,
                "total_files": total_files,
                "chunk_count": chunk_count,
                "persist_directory": str(settings.CHROMA_PERSIST_DIRECTORY),
                "embedding_model": settings.EMBEDDING_MODEL,
                "groq_model": settings.GROQ_MODEL,
                "has_groq_key": settings.has_groq_key,
            }
        except Exception as e:
            logger.error(f"Error checking knowledge base status: {str(e)}", exc_info=True)
            return {
                "database_status": "error",
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "document_count": 0,
                "chunk_count": 0,
                "error": str(e),
            }

    def run_ingestion(self, reset: bool = False, clear_cache: bool = True) -> Dict[str, Any]:
        """Run document loading, text splitting, embedding, vector persistence, and cache invalidation."""
        logger.info(f"Starting ingestion process (reset={reset}, clear_cache={clear_cache})...")

        if reset:
            self.vector_store_manager.reset_database()

        loader = InsuranceDocumentLoader(settings.DOCUMENT_DIRECTORY)
        documents = loader.load()

        if not documents:
            logger.warning(f"No documents found in {settings.DOCUMENT_DIRECTORY}")
            return {
                "status": "warning",
                "message": f"No documents found in {settings.DOCUMENT_DIRECTORY}",
                "documents_loaded": 0,
                "chunks_generated": 0,
                "chunks_added": 0,
            }

        splitter = InsuranceTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        chunks_added = self.vector_store_manager.add_documents(chunks)
        total_chunks = self.vector_store_manager.count_chunks()
        total_docs = self.vector_store_manager.get_indexed_document_count()

        # Automatically clear/invalidate response cache on re-index so outdated answers are purged
        if clear_cache:
            self.response_cache.clear_cache()
            logger.info("Semantic response cache cleared following document re-indexing.")

        logger.info(
            f"Ingestion finished: {len(documents)} docs loaded, "
            f"{len(chunks)} chunks created, {chunks_added} newly indexed. Total in DB: {total_chunks}"
        )

        return {
            "status": "success",
            "message": "Knowledge base ingested and persisted successfully. Response cache invalidated.",
            "documents_loaded": len(documents),
            "chunks_generated": len(chunks),
            "chunks_added": chunks_added,
            "total_chunks_in_db": total_chunks,
            "total_documents_in_db": total_docs,
        }


_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
