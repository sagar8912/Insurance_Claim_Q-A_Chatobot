"""Unit tests for prompt formatting and RAG pipeline components."""
import pytest
from app.rag.prompt import get_rag_prompt_template, SYSTEM_PROMPT
from app.rag.rag_pipeline import InsuranceRAGPipeline
from langchain_core.documents import Document


def test_prompt_template_structure():
    """Verify prompt template contains required compliance phrases."""
    prompt = get_rag_prompt_template()
    assert prompt is not None
    assert "Final coverage and claim eligibility depend" in SYSTEM_PROMPT
    assert "I could not find this information" in SYSTEM_PROMPT
    assert "STRICT OPERATIONAL RULES" in SYSTEM_PROMPT


def test_format_sources():
    """Verify source document formatting and deduplication."""
    pipeline = InsuranceRAGPipeline.__new__(InsuranceRAGPipeline)
    docs = [
        Document(
            page_content="Coverage for health hospitalization up to 500,000.",
            metadata={"filename": "health_policy.txt", "document_type": "health", "similarity_score": 0.85},
        ),
        Document(
            page_content="Additional health conditions and waiting period rules.",
            metadata={"filename": "health_policy.txt", "document_type": "health", "similarity_score": 0.80},
        ),
        Document(
            page_content="Motor policy comprehensive terms.",
            metadata={"filename": "motor_policy.txt", "document_type": "motor", "similarity_score": 0.75},
        ),
    ]

    sources = pipeline.format_sources(docs)
    # Deduplication should keep only 2 sources (one for health_policy.txt, one for motor_policy.txt)
    assert len(sources) == 2
    assert sources[0]["filename"] == "health_policy.txt"
    assert sources[0]["document_type"] == "health"
    assert "hospitalization" in sources[0]["preview"]
    assert sources[1]["filename"] == "motor_policy.txt"


def test_missing_documents_fallback():
    """Verify answer fallback when no documents are retrieved."""
    pipeline = InsuranceRAGPipeline.__new__(InsuranceRAGPipeline)
    pipeline.llm = None

    answer = pipeline.generate_answer("How to claim?", [])
    assert "could not find this information" in answer
