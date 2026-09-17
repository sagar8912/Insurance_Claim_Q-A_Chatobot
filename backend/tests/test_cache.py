"""Comprehensive test suite for Semantic Response Cache in Insurance RAG Chatbot."""
import time
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.config import settings
from app.cache import (
    ResponseCacheService,
    get_response_cache,
    normalize_question,
    generate_cache_id,
    CacheEntry,
)
from app.rag.hallucination_checker import HallucinationChecker, ValidationResult
from app.services.chat_service import ChatService


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure response cache is cleared before and after each test."""
    cache = get_response_cache()
    cache.clear_cache()
    # Reset KB version to v1
    settings.KNOWLEDGE_BASE_VERSION = "v1"
    yield
    cache.clear_cache()
    settings.KNOWLEDGE_BASE_VERSION = "v1"


# ---------------------------------------------------------------------------
# Test 1, 2, 3: Normalization & Exact Cache (Capitalization, Punctuation, Spaces)
# ---------------------------------------------------------------------------

def test_normalization_variations():
    """Test question normalization handles capitalization, punctuation, and extra spaces."""
    q1 = "What is the waiting period for pre-existing diseases?"
    q2 = "what is the waiting period for pre-existing diseases"
    q3 = "   WHAT   IS   THE   WAITING   PERIOD   FOR   PRE-EXISTING   DISEASES???   "
    q4 = "What is the waiting period for pre-existing diseases?!"

    n1 = normalize_question(q1)
    n2 = normalize_question(q2)
    n3 = normalize_question(q3)
    n4 = normalize_question(q4)

    assert n1 == "what is the waiting period for pre-existing diseases"
    assert n1 == n2
    assert n1 == n3
    assert n1 == n4


def test_exact_cache_hit_and_variations():
    """Scenario 1, 2, 3: Exact same question, different capitalization, extra spaces & punctuation."""
    cache = get_response_cache()
    question = "What is the waiting period for pre-existing diseases?"
    answer = "The waiting period for pre-existing diseases is 24 months of continuous coverage."
    sources = [{
        "filename": "health_policy.txt",
        "file_type": "txt",
        "document_type": "health",
        "preview": "Pre-existing diseases have a waiting period of 24 months.",
        "similarity_score": 0.95,
    }]

    # 1. Initially miss
    hit = cache.check_exact_cache(question)
    assert hit is None

    # Save verified answer
    saved = cache.save_response(
        question=question,
        answer=answer,
        sources=sources,
        grounded=True,
        hallucination_score=0.0,
    )
    assert saved is True

    # 2. Exact same question matches
    hit_exact = cache.check_exact_cache(question)
    assert hit_exact is not None
    assert hit_exact.cache_type == "exact"
    assert hit_exact.similarity == 1.0
    assert hit_exact.entry.answer == answer

    # 3. Capitalization variation matches
    hit_upper = cache.check_exact_cache("WHAT IS THE WAITING PERIOD FOR PRE-EXISTING DISEASES?")
    assert hit_upper is not None
    assert hit_upper.cache_type == "exact"
    assert hit_upper.entry.answer == answer

    # 4. Extra spaces and punctuation match
    hit_spaces = cache.check_exact_cache("  what   is   the   waiting   period   for   pre-existing   diseases???  ")
    assert hit_spaces is not None
    assert hit_spaces.cache_type == "exact"
    assert hit_spaces.entry.answer == answer


# ---------------------------------------------------------------------------
# Test 4: Similar Semantic Question
# ---------------------------------------------------------------------------

def test_semantic_cache_hit_similar_question():
    """Scenario 4: Similar semantic question passes threshold and returns cached answer."""
    cache = get_response_cache()
    q_original = "What is the waiting period for pre-existing diseases?"
    answer = "The waiting period for pre-existing conditions is 24 months."
    sources = [{
        "filename": "health_policy.txt",
        "file_type": "txt",
        "document_type": "health",
        "preview": "Waiting period for pre-existing diseases is 24 months.",
    }]

    # Save into cache
    cache.save_response(
        question=q_original,
        answer=answer,
        sources=sources,
        grounded=True,
        hallucination_score=0.0,
        policy_category="health",
    )

    # Ask semantically very similar question
    q_similar = "How long is the waiting period for pre-existing conditions?"
    
    # Verify exact cache misses
    exact_hit = cache.check_exact_cache(q_similar)
    assert exact_hit is None

    # Check semantic cache (model-sensitive threshold testing)
    prev_threshold = settings.SEMANTIC_CACHE_THRESHOLD
    try:
        settings.SEMANTIC_CACHE_THRESHOLD = 0.68
        semantic_hit = cache.check_semantic_cache(q_similar)
        assert semantic_hit is not None
        assert semantic_hit.cache_type == "semantic"
        assert semantic_hit.similarity >= 0.68
        assert semantic_hit.entry.answer == answer
    finally:
        settings.SEMANTIC_CACHE_THRESHOLD = prev_threshold


# ---------------------------------------------------------------------------
# Test 5: Different Insurance Category Question (Safety Check)
# ---------------------------------------------------------------------------

def test_different_insurance_category_cache_miss():
    """Scenario 5: Unrelated or different insurance category question causes cache miss."""
    cache = get_response_cache()
    q_health = "What is the waiting period for pre-existing diseases?"
    answer_health = "The waiting period for pre-existing conditions is 24 months."
    sources = [{
        "filename": "health_policy.txt",
        "file_type": "txt",
        "document_type": "health",
        "preview": "Waiting period is 24 months.",
    }]

    cache.save_response(
        question=q_health,
        answer=answer_health,
        sources=sources,
        grounded=True,
        hallucination_score=0.0,
        policy_category="health",
    )

    # Completely different question (motor insurance)
    q_motor = "Does motor insurance cover vehicle theft?"
    semantic_hit = cache.check_semantic_cache(q_motor)
    assert semantic_hit is None


# ---------------------------------------------------------------------------
# Test 6: Knowledge Base Version Invalidation
# ---------------------------------------------------------------------------

def test_old_knowledge_base_version_rejected():
    """Scenario 6: Cache entry stored under v1 is rejected when active version is v2."""
    cache = get_response_cache()
    settings.KNOWLEDGE_BASE_VERSION = "v1"

    question = "What is the deductible amount?"
    answer = "The deductible is $500 per claim."
    sources = [{"filename": "doc.txt", "file_type": "txt", "document_type": "general", "preview": "Deductible is $500."}]

    cache.save_response(
        question=question,
        answer=answer,
        sources=sources,
        grounded=True,
    )

    # Verify hit on v1
    assert cache.check_exact_cache(question) is not None

    # Change active version to v2
    settings.KNOWLEDGE_BASE_VERSION = "v2"

    # Exact check should fail validation and return None
    hit_after = cache.check_exact_cache(question)
    assert hit_after is None


# ---------------------------------------------------------------------------
# Test 7: Expired Cache (TTL)
# ---------------------------------------------------------------------------

def test_expired_cache_entry_rejected():
    """Scenario 7: Cache entry past TTL is treated as cache miss."""
    cache = get_response_cache()
    question = "What is the grace period for premium payment?"
    answer = "The grace period is 30 days."
    sources = [{"filename": "doc.txt", "file_type": "txt", "document_type": "general", "preview": "30 days grace period."}]

    cache.save_response(
        question=question,
        answer=answer,
        sources=sources,
        grounded=True,
    )

    # Simulate aged entry by modifying created_at in memory and ChromaDB
    normalized_q = normalize_question(question)
    entry = cache.exact_cache.get(normalized_q)
    assert entry is not None
    # Set created_at to 200 hours ago (TTL is 168 hours)
    entry.created_at = time.time() - (200 * 3600)

    # Validation should reject due to expiration
    is_valid, reason = cache.validate_cache_entry(entry, similarity=1.0)
    assert is_valid is False
    assert "expired" in reason.lower()


# ---------------------------------------------------------------------------
# Test 8: Hallucinated / Ungrounded Response Never Cached
# ---------------------------------------------------------------------------

def test_hallucination_and_ungrounded_never_cached():
    """Scenario 8: Ungrounded or hallucinated answers are strictly rejected from cache."""
    cache = get_response_cache()
    checker = HallucinationChecker()

    # 1. Explicit refusal / ungrounded phrase
    refusal_answer = "I could not find this information in the available insurance policy documents."
    doc = Document(page_content="Some policy terms about dental care.")
    val_refusal = checker.validate(refusal_answer, [doc], sources=[{"filename": "doc.txt"}])
    assert val_refusal.is_cacheable is False

    saved_refusal = cache.save_response(
        question="What is the waiting period?",
        answer=refusal_answer,
        sources=[{"filename": "doc.txt"}],
        grounded=False,
        hallucination_score=1.0,
    )
    assert saved_refusal is False

    # 2. Fabricated numbers (not in source document)
    context_doc = Document(page_content="Health policy provides basic outpatient consultation benefits.")
    hallucinated_answer = "The waiting period is 48 months and surgery coverage is $25000."
    val_hallucinated = checker.validate(hallucinated_answer, [context_doc], sources=[{"filename": "doc.txt"}])
    assert val_hallucinated.hallucination_score > 0.3

    saved_hallucinated = cache.save_response(
        question="What is surgery coverage?",
        answer=hallucinated_answer,
        sources=[{"filename": "doc.txt"}],
        grounded=True,
        hallucination_score=val_hallucinated.hallucination_score,
    )
    assert saved_hallucinated is False


# ---------------------------------------------------------------------------
# Test 9: Failed API / Error Response Never Cached
# ---------------------------------------------------------------------------

def test_failed_api_response_never_cached():
    """Scenario 9: API errors or blank answers are never cached."""
    cache = get_response_cache()

    # Empty answer
    assert cache.save_response("question?", "", [{"filename": "doc.txt"}], grounded=True) is False

    # Ungrounded error
    assert cache.save_response("question?", "Something went wrong.", [], grounded=False) is False

    # Empty sources
    assert cache.save_response("question?", "Valid answer text.", [], grounded=True) is False


# ---------------------------------------------------------------------------
# Test 10: Re-index Knowledge Base Clears Response Cache
# ---------------------------------------------------------------------------

def test_reindex_clears_response_cache():
    """Scenario 10: Re-indexing knowledge base automatically clears response cache."""
    cache = get_response_cache()
    question = "What is the claim notification window?"
    answer = "Claims must be notified within 48 hours."
    sources = [{"filename": "claims.txt", "file_type": "txt", "document_type": "general", "preview": "48 hours notification."}]

    cache.save_response(question, answer, sources, grounded=True)
    assert cache.check_exact_cache(question) is not None

    # Simulate ChatService run_ingestion
    mock_rag = MagicMock()
    mock_vec = MagicMock()
    mock_vec.count_chunks.return_value = 10
    mock_vec.get_indexed_document_count.return_value = 2

    chat_service = ChatService(
        rag_pipeline=mock_rag,
        vector_store_manager=mock_vec,
        response_cache=cache,
    )

    with patch("app.services.chat_service.InsuranceDocumentLoader") as mock_loader_cls, \
         patch("app.services.chat_service.InsuranceTextSplitter") as mock_splitter_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="Policy text", metadata={"filename": "doc.txt"})]
        mock_loader_cls.return_value = mock_loader

        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [Document(page_content="Chunk 1", metadata={"filename": "doc.txt"})]
        mock_splitter_cls.return_value = mock_splitter

        result = chat_service.run_ingestion(reset=False, clear_cache=True)
        assert result["status"] == "success"

    # Verify cache was cleared
    assert cache.check_exact_cache(question) is None


# ---------------------------------------------------------------------------
# Test 11: Manual Cache Invalidation
# ---------------------------------------------------------------------------

def test_manual_cache_invalidation():
    """Scenario 11: clear_cache() purges cache completely."""
    cache = get_response_cache()
    cache.save_response(
        "Is dental care covered?",
        "Dental care is covered only following accidental injury.",
        [{"filename": "health.txt", "file_type": "txt", "document_type": "health", "preview": "Accidental injury only."}],
        grounded=True,
    )
    assert cache.check_exact_cache("Is dental care covered?") is not None

    clear_result = cache.clear_cache()
    assert clear_result["success"] is True

    assert cache.check_exact_cache("Is dental care covered?") is None


# ---------------------------------------------------------------------------
# Test 12: Concurrent Repeated Requests & Groq Bypass Verification
# ---------------------------------------------------------------------------

def test_concurrent_requests_and_groq_bypass():
    """Scenario 12: Repeated requests bypass Groq LLM and are thread-safe."""
    import concurrent.futures

    cache = get_response_cache()
    mock_rag = MagicMock()
    mock_rag.ask.return_value = {
        "success": True,
        "answer": "Maternity coverage requires 9 months continuous coverage.",
        "sources": [{"filename": "health.txt", "file_type": "txt", "document_type": "health", "preview": "9 months continuous coverage."}],
        "grounded": True,
        "processing_time": 1.25,
    }
    mock_rag.retrieve_context.return_value = [
        Document(page_content="Maternity coverage requires 9 months continuous coverage.", metadata={"filename": "health.txt"})
    ]

    chat_service = ChatService(
        rag_pipeline=mock_rag,
        response_cache=cache,
    )

    question = "What is the waiting period for maternity coverage?"

    # Request 1: Cache Miss -> calls Groq
    res1 = chat_service.process_message(question)
    assert res1["cache_hit"] is False
    assert mock_rag.ask.call_count == 1

    # Concurrent 10 requests with identical or normalized question
    questions = [
        question,
        "what is the waiting period for maternity coverage?",
        "  WHAT IS THE WAITING PERIOD FOR MATERNITY COVERAGE???  ",
    ] * 4

    def run_query(q):
        return chat_service.process_message(q)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_query, questions))

    # All repeated requests must be cache hits
    for r in results:
        assert r["cache_hit"] is True
        assert r["cache_type"] in ("exact", "semantic")

    # Groq should NOT have been called again! (call_count remains 1)
    assert mock_rag.ask.call_count == 1

    # Check metrics
    metrics = cache.get_statistics()
    assert metrics.total_requests >= 13
    assert metrics.exact_cache_hits >= 12
    assert metrics.estimated_groq_calls_saved >= 12
