"""Semantic Response Cache Service using existing ChromaDB and HuggingFace Embeddings."""
import json
import time
import threading
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import settings
from ..rag.embeddings import get_embedding_service, EmbeddingService
from ..rag.vector_store import get_vector_store_manager, ChromaVectorStoreManager
from .cache_models import CacheEntry, CacheSearchResult, CacheMetrics
from .exact_cache import normalize_question, generate_cache_id, ExactCacheManager
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.response_cache")


class ResponseCacheService:
    """Production-grade Semantic Response Cache with exact and embedding similarity lookups."""

    _instance: Optional["ResponseCacheService"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "ResponseCacheService":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super(ResponseCacheService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.collection_name = settings.CHROMA_CACHE_COLLECTION_NAME
        self.embedding_service = get_embedding_service()
        self.vector_store_manager = get_vector_store_manager()
        self.exact_cache = ExactCacheManager(max_capacity=2000)

        # Metrics tracking
        self._metrics_lock = threading.Lock()
        self._total_requests = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._cache_misses = 0
        self._cached_latencies: List[float] = []
        self._rag_latencies: List[float] = []

        self._collection = None
        self._init_collection()
        self._initialized = True

    def _init_collection(self) -> None:
        """Initialize separate ChromaDB collection for response cache with cosine distance."""
        try:
            self.vector_store_manager._ensure_connected()
            client = self.vector_store_manager._client
            if client is None:
                try:
                    from chromadb.api.client import SharedSystemClient
                    SharedSystemClient.clear_system_cache()
                except Exception:
                    pass
                client = chromadb.PersistentClient(
                    path=str(settings.CHROMA_PERSIST_DIRECTORY),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Connected to response cache collection '{self.collection_name}' "
                f"({self._collection.count()} entries present)."
            )
        except Exception as e:
            logger.error(f"Failed to initialize response cache collection: {str(e)}", exc_info=True)
            self._collection = None

    @property
    def collection(self):
        try:
            if self._collection is None:
                self._init_collection()
            else:
                # Validate that rust bindings are still active
                self._collection.count()
        except Exception:
            self._init_collection()
        return self._collection

    def normalize_question(self, question: str) -> str:
        """Expose question normalization helper."""
        return normalize_question(question)

    def validate_cache_entry(
        self, entry: CacheEntry, similarity: float, query_category: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate cached entry against all 6 safety and freshness criteria.

        Criteria:
        1. Similarity is above threshold.
        2. Cached answer was grounded.
        3. Cached answer has no hallucination detected (score <= 0.3).
        4. Knowledge base version matches current system setting.
        5. Cache entry is not expired (age <= CACHE_TTL_HOURS).
        6. Cached answer contains valid sources.
        7. Safety check: Policy category compatibility.
        """
        threshold = settings.SEMANTIC_CACHE_THRESHOLD
        if similarity < threshold and similarity < 0.999:  # Allow 1.0 for exact
            return False, f"Similarity {similarity:.4f} below threshold {threshold}"

        if not entry.grounded:
            return False, "Cached response is marked ungrounded"

        if entry.hallucination_score > 0.3:
            return False, f"Cached response has high hallucination score ({entry.hallucination_score})"

        if entry.knowledge_base_version != settings.KNOWLEDGE_BASE_VERSION:
            return False, (
                f"Knowledge base version mismatch (cached: {entry.knowledge_base_version}, "
                f"current: {settings.KNOWLEDGE_BASE_VERSION})"
            )

        if entry.age_hours > settings.CACHE_TTL_HOURS:
            return False, f"Cache entry expired (age: {entry.age_hours:.1f}h, TTL: {settings.CACHE_TTL_HOURS}h)"

        if not entry.sources or len(entry.sources) == 0:
            return False, "Cached response has empty sources"

        if not entry.answer or not entry.answer.strip():
            return False, "Cached answer is blank or empty"

        # Check policy category mismatch if specified
        if query_category and entry.policy_category and entry.policy_category != "general":
            if query_category.lower() != entry.policy_category.lower():
                return False, f"Category mismatch: query is '{query_category}' but cached is '{entry.policy_category}'"

        return True, None

    def check_exact_cache(self, question: str) -> Optional[CacheSearchResult]:
        """Perform exact match check using normalized question and deterministic ID."""
        if not settings.ENABLE_RESPONSE_CACHE:
            return None

        normalized_q = normalize_question(question)
        if not normalized_q:
            return None

        cache_id = generate_cache_id(normalized_q)

        # 1. Check in-memory exact cache
        entry = self.exact_cache.get(normalized_q)

        # 2. If not in memory, check Chroma collection by deterministic ID
        if entry is None and self.collection is not None:
            try:
                res = self.collection.get(ids=[cache_id], include=["metadatas", "documents"])
                if res and res.get("ids") and len(res["ids"]) > 0:
                    meta = res["metadatas"][0]
                    entry = self._deserialize_entry(cache_id, res["documents"][0], meta)
                    if entry:
                        self.exact_cache.put(normalized_q, entry)
            except Exception as e:
                logger.error(f"Error reading exact cache from Chroma: {str(e)}")

        if entry is None:
            return None

        # Validate entry
        is_valid, reason = self.validate_cache_entry(entry, similarity=1.0)
        if not is_valid:
            logger.info(f"Exact cache entry found but failed validation: {reason}")
            return None

        logger.info(f"CACHE HIT - EXACT: Returning cached response for '{normalized_q}' (ID: {cache_id})")
        with self._metrics_lock:
            self._exact_hits += 1

        return CacheSearchResult(
            entry=entry,
            similarity=1.0,
            cache_type="exact",
            is_valid=True,
        )

    def check_semantic_cache(
        self, question: str, embedding: Optional[List[float]] = None
    ) -> Optional[CacheSearchResult]:
        """Perform semantic similarity search on the response cache collection."""
        if not settings.ENABLE_RESPONSE_CACHE or self.collection is None:
            return None

        try:
            if self.collection.count() == 0:
                return None
        except Exception:
            return None

        normalized_q = normalize_question(question)

        # Generate embedding if not passed
        if embedding is None:
            try:
                embedding = self.embedding_service.embed_query(normalized_q)
            except Exception as e:
                logger.error(f"Failed to generate query embedding for cache search: {str(e)}")
                return None

        try:
            k = max(1, settings.CACHE_MAX_RESULTS)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["metadatas", "documents", "distances"],
            )

            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            documents = results.get("documents", [[]])[0]

            for i in range(len(ids)):
                cache_id = ids[i]
                meta = metadatas[i]
                dist = distances[i] if distances else 1.0
                doc = documents[i] if documents else ""

                # With hnsw:space: cosine, distance = 1 - cosine_similarity
                # Bound similarity to [0.0, 1.0]
                similarity = max(0.0, min(1.0, 1.0 - dist))

                if similarity < settings.SEMANTIC_CACHE_THRESHOLD:
                    logger.debug(
                        f"Semantic candidate {cache_id} similarity {similarity:.4f} "
                        f"below threshold {settings.SEMANTIC_CACHE_THRESHOLD}"
                    )
                    continue

                entry = self._deserialize_entry(cache_id, doc, meta)
                if not entry:
                    continue

                is_valid, reason = self.validate_cache_entry(entry, similarity=similarity)
                if not is_valid:
                    logger.info(f"Semantic candidate {cache_id} (sim: {similarity:.4f}) failed validation: {reason}")
                    continue

                logger.info(
                    f"CACHE HIT - SEMANTIC: Matched '{entry.normalized_question}' "
                    f"(similarity: {similarity:.4f} >= {settings.SEMANTIC_CACHE_THRESHOLD}) for '{normalized_q}'"
                )
                with self._metrics_lock:
                    self._semantic_hits += 1

                return CacheSearchResult(
                    entry=entry,
                    similarity=round(similarity, 4),
                    cache_type="semantic",
                    is_valid=True,
                )

        except Exception as e:
            logger.error(f"Error during semantic cache search: {str(e)}", exc_info=True)

        return None

    def save_response(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        grounded: bool,
        hallucination_score: float = 0.0,
        model_name: Optional[str] = None,
        policy_category: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Validate and persist grounded answer into ChromaDB response cache and memory cache."""
        if not settings.ENABLE_RESPONSE_CACHE or self.collection is None:
            return False

        # STRICT SAFETY CHECK: Never cache invalid, ungrounded, or blank responses
        if not grounded:
            logger.info("CACHE NOT SAVED: Response is ungrounded.")
            return False

        if hallucination_score > 0.3:
            logger.info(f"CACHE NOT SAVED: Hallucination score ({hallucination_score}) too high.")
            return False

        if not answer or not answer.strip():
            logger.info("CACHE NOT SAVED: Answer is empty.")
            return False

        if not sources or len(sources) == 0:
            logger.info("CACHE NOT SAVED: Sources list is empty.")
            return False

        normalized_q = normalize_question(question)
        if not normalized_q:
            return False

        cache_id = generate_cache_id(normalized_q)

        # Detect policy category from sources if not provided
        if not policy_category and sources:
            categories = {s.get("document_type") for s in sources if s.get("document_type")}
            if len(categories) == 1:
                policy_category = list(categories)[0]
            else:
                policy_category = "general"

        entry = CacheEntry(
            id=cache_id,
            question=question,
            normalized_question=normalized_q,
            answer=answer,
            sources=sources,
            grounded=grounded,
            hallucination_score=hallucination_score,
            created_at=time.time(),
            knowledge_base_version=settings.KNOWLEDGE_BASE_VERSION,
            model_name=model_name or settings.GROQ_MODEL,
            cache_type="verified_rag",
            policy_category=policy_category or "general",
        )

        try:
            # Generate embedding for normalized question
            if embedding is None:
                embedding = self.embedding_service.embed_query(normalized_q)

            # Metadata must be primitives for ChromaDB
            chroma_meta = {
                "question": question,
                "normalized_question": normalized_q,
                "answer": answer,
                "sources": json.dumps(sources),
                "grounded": grounded,
                "hallucination_score": float(hallucination_score),
                "created_at": float(entry.created_at),
                "knowledge_base_version": settings.KNOWLEDGE_BASE_VERSION,
                "model_name": entry.model_name,
                "cache_type": entry.cache_type,
                "policy_category": entry.policy_category,
            }

            self.collection.upsert(
                ids=[cache_id],
                embeddings=[embedding],
                documents=[normalized_q],
                metadatas=[chroma_meta],
            )

            # Also store in fast memory exact cache
            self.exact_cache.put(normalized_q, entry)

            logger.info(f"CACHE SAVED: Verified grounded response stored with ID '{cache_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to persist response cache entry '{cache_id}': {str(e)}", exc_info=True)
            return False

    def record_request_start(self) -> None:
        with self._metrics_lock:
            self._total_requests += 1

    def record_cache_miss(self) -> None:
        logger.info("CACHE MISS: No valid cache entry found. Proceeding to RAG pipeline.")
        with self._metrics_lock:
            self._cache_misses += 1

    def record_cached_latency(self, elapsed: float) -> None:
        with self._metrics_lock:
            self._cached_latencies.append(elapsed)
            if len(self._cached_latencies) > 200:
                self._cached_latencies.pop(0)

    def record_rag_latency(self, elapsed: float) -> None:
        with self._metrics_lock:
            self._rag_latencies.append(elapsed)
            if len(self._rag_latencies) > 200:
                self._rag_latencies.pop(0)

    def clear_cache(self) -> Dict[str, Any]:
        """Purge all entries from the response cache collection and in-memory cache.

        CRITICAL: Only clears 'insurance_response_cache'; DOES NOT touch knowledge base!
        """
        logger.warning(f"Clearing response cache collection '{self.collection_name}'...")
        deleted_count = 0
        try:
            if self.collection is not None:
                deleted_count = self.collection.count()
                client = self.vector_store_manager._client
                if client:
                    try:
                        client.delete_collection(self.collection_name)
                    except Exception:
                        pass
                self._collection = None
                self._init_collection()

            self.exact_cache.clear()
            logger.info(f"Response cache cleared ({deleted_count} entries purged).")
            return {
                "success": True,
                "message": f"Response cache cleared. Purged {deleted_count} entries.",
                "deleted_count": deleted_count,
            }
        except Exception as e:
            logger.error(f"Error clearing response cache: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def invalidate_cache(self, new_version: Optional[str] = None) -> Dict[str, Any]:
        """Invalidate cache entries by version or clearing collection."""
        if new_version:
            logger.info(f"Updating knowledge base version from {settings.KNOWLEDGE_BASE_VERSION} to {new_version}")
            settings.KNOWLEDGE_BASE_VERSION = new_version
            self.exact_cache.clear()
            return {
                "success": True,
                "message": f"Knowledge base version updated to {new_version}. Previous version cache invalidated.",
                "current_version": new_version,
            }
        else:
            return self.clear_cache()

    def get_statistics(self) -> CacheMetrics:
        """Fetch real-time cache performance and operational metrics."""
        with self._metrics_lock:
            total = self._total_requests
            hits = self._exact_hits + self._semantic_hits
            rate = round((hits / total) if total > 0 else 0.0, 4)
            avg_cached = (
                round(sum(self._cached_latencies) / len(self._cached_latencies), 3)
                if self._cached_latencies
                else 0.0
            )
            avg_rag = (
                round(sum(self._rag_latencies) / len(self._rag_latencies), 3)
                if self._rag_latencies
                else 0.0
            )

            total_entries = 0
            if self.collection is not None:
                try:
                    total_entries = self.collection.count()
                except Exception:
                    total_entries = 0

            return CacheMetrics(
                total_requests=total,
                exact_cache_hits=self._exact_hits,
                semantic_cache_hits=self._semantic_hits,
                cache_misses=self._cache_misses,
                cache_hit_rate=rate,
                estimated_groq_calls_saved=hits,
                total_cached_entries=total_entries,
                average_cached_response_time=avg_cached,
                average_rag_response_time=avg_rag,
                knowledge_base_version=settings.KNOWLEDGE_BASE_VERSION,
            )

    def _deserialize_entry(self, cache_id: str, document: str, meta: Dict[str, Any]) -> Optional[CacheEntry]:
        """Safely deserialize Chroma metadata dictionary into CacheEntry model."""
        try:
            sources_raw = meta.get("sources", "[]")
            sources = json.loads(sources_raw) if isinstance(sources_raw, str) else sources_raw

            return CacheEntry(
                id=cache_id,
                question=meta.get("question", document),
                normalized_question=meta.get("normalized_question", document),
                answer=meta.get("answer", ""),
                sources=sources,
                grounded=bool(meta.get("grounded", True)),
                hallucination_score=float(meta.get("hallucination_score", 0.0)),
                created_at=float(meta.get("created_at", time.time())),
                knowledge_base_version=str(meta.get("knowledge_base_version", "v1")),
                model_name=str(meta.get("model_name", "unknown")),
                cache_type=str(meta.get("cache_type", "verified_rag")),
                policy_category=str(meta.get("policy_category", "general")),
                metadata=meta,
            )
        except Exception as e:
            logger.error(f"Error deserializing cache entry '{cache_id}': {str(e)}")
            return None


_response_cache: Optional[ResponseCacheService] = None


def get_response_cache() -> ResponseCacheService:
    """Factory helper to obtain the ResponseCacheService singleton."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCacheService()
    return _response_cache
