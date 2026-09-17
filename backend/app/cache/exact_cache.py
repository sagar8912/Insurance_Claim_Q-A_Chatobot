"""Exact question normalization and in-memory cache lookup."""
import re
import hashlib
import threading
from typing import Optional, Dict
from collections import OrderedDict

from .cache_models import CacheEntry
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.exact_cache")


def normalize_question(question: str) -> str:
    """Normalize user question for exact cache matching.

    Rules:
    1. Convert to lowercase.
    2. Strip leading/trailing whitespace.
    3. Collapse consecutive spaces to single space.
    4. Remove unnecessary punctuation (?, !, ., ,, ;, :, ", ', `, etc.)
       while preserving meaningful hyphens and slashes in insurance terms (e.g., 'pre-existing', 'in-patient').
    """
    if not question:
        return ""

    # 1. Lowercase and strip
    text = question.strip().lower()

    # 2. Remove standard terminal and enclosing punctuation
    # Replace non-word chars that aren't hyphens or slashes with space
    text = re.sub(r"[^\w\s\-\/]", " ", text)

    # 3. Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_cache_id(normalized_question: str) -> str:
    """Generate deterministic SHA-256 cache ID for a normalized question."""
    return f"cache_{hashlib.sha256(normalized_question.encode('utf-8')).hexdigest()[:24]}"


class ExactCacheManager:
    """Thread-safe LRU in-memory cache for ultra-fast sub-millisecond exact matches."""

    def __init__(self, max_capacity: int = 1000) -> None:
        self.max_capacity = max_capacity
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, normalized_q: str) -> Optional[CacheEntry]:
        """Retrieve entry by normalized question."""
        with self._lock:
            if normalized_q in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(normalized_q)
                return self._cache[normalized_q]
        return None

    def put(self, normalized_q: str, entry: CacheEntry) -> None:
        """Add or update entry in memory cache."""
        with self._lock:
            if normalized_q in self._cache:
                self._cache.move_to_end(normalized_q)
            self._cache[normalized_q] = entry
            if len(self._cache) > self.max_capacity:
                # Pop oldest
                self._cache.popitem(last=False)

    def remove(self, normalized_q: str) -> None:
        """Remove entry if present."""
        with self._lock:
            self._cache.pop(normalized_q, None)

    def clear(self) -> None:
        """Clear all in-memory cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("In-memory exact cache cleared.")

    def count(self) -> int:
        with self._lock:
            return len(self._cache)
