"""Data models and validation structures for Semantic Response Cache."""
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """Full representation of a cached response item."""

    id: str
    question: str
    normalized_question: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    grounded: bool = True
    hallucination_score: float = 0.0
    created_at: float = Field(default_factory=time.time)
    knowledge_base_version: str = "v1"
    model_name: str = "llama-3.3-70b-versatile"
    cache_type: str = "verified_rag"
    policy_category: Optional[str] = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Calculate age of this entry in seconds."""
        return max(0.0, time.time() - self.created_at)

    @property
    def age_hours(self) -> float:
        """Calculate age of this entry in hours."""
        return self.age_seconds / 3600.0


class CacheSearchResult(BaseModel):
    """Encapsulates the result of checking the cache."""

    entry: Optional[CacheEntry] = None
    similarity: float = 0.0
    cache_type: Optional[str] = None  # "exact", "semantic", or None
    is_valid: bool = False
    invalidation_reason: Optional[str] = None


class CacheMetrics(BaseModel):
    """Aggregated operational metrics for the semantic response cache."""

    total_requests: int = 0
    exact_cache_hits: int = 0
    semantic_cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    estimated_groq_calls_saved: int = 0
    total_cached_entries: int = 0
    average_cached_response_time: float = 0.0
    average_rag_response_time: float = 0.0
    knowledge_base_version: str = "v1"
