"""Unit tests for document loading and text splitting."""
import pytest
from pathlib import Path
from app.rag.document_loader import InsuranceDocumentLoader, determine_document_type
from app.rag.text_splitter import InsuranceTextSplitter
from langchain_core.documents import Document


def test_determine_document_type():
    """Test classification of document type based on filename."""
    assert determine_document_type("health_policy_overview.txt") == "health"
    assert determine_document_type("motor_comprehensive.txt") == "motor"
    assert determine_document_type("life_insurance.txt") == "life"
    assert determine_document_type("travel_insurance.txt") == "travel"
    assert determine_document_type("home_insurance.txt") == "home"
    assert determine_document_type("claims_process.txt") == "claims"
    assert determine_document_type("claim_rejection.txt") == "claims"
    assert determine_document_type("customer_faq.txt") == "faq"
    assert determine_document_type("general_exclusions.txt") == "general"


def test_document_loader(tmp_path: Path):
    """Test loading documents from a directory."""
    # Create sample files
    sample_file_1 = tmp_path / "health_test.txt"
    sample_file_1.write_text("SecureLife health policy covers hospitalization.", encoding="utf-8")

    sample_file_2 = tmp_path / "motor_test.txt"
    sample_file_2.write_text("Comprehensive motor insurance covers vehicle theft.", encoding="utf-8")

    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    loader = InsuranceDocumentLoader(tmp_path)
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].metadata["document_type"] in ["health", "motor"]
    assert "SecureLife" in docs[0].page_content or "Comprehensive" in docs[0].page_content


def test_text_splitter():
    """Test splitting documents into chunks with IDs and metadata."""
    sample_doc = Document(
        page_content="This is section one. " * 30 + "\n\n" + "This is section two. " * 30,
        metadata={"filename": "test_doc.txt", "document_type": "test", "source": "test_doc.txt"},
    )

    splitter = InsuranceTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents([sample_doc])

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_id"] == f"test_doc.txt_chunk_{i}"
        assert chunk.metadata["filename"] == "test_doc.txt"
        assert len(chunk.page_content) > 0
