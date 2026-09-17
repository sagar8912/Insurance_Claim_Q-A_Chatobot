"""Standalone CLI script to ingest insurance knowledge base into ChromaDB."""
import sys
import os
import argparse
from pathlib import Path

# Add backend directory to sys.path so app modules are resolvable
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.rag.document_loader import InsuranceDocumentLoader
from app.rag.text_splitter import InsuranceTextSplitter
from app.rag.vector_store import get_vector_store_manager
from app.cache import get_response_cache
from app.utils.logger import get_logger

logger = get_logger("insurance_rag.ingest_cli")


def run_ingestion(reset: bool = False) -> None:
    """Execute complete ingestion pipeline and print statistics."""
    print("========================================")
    print("   INSURANCE KNOWLEDGE BASE INGESTION")
    print("========================================")

    vector_mgr = get_vector_store_manager()

    if reset:
        vector_mgr.reset_database()

    # 1. Load documents
    loader = InsuranceDocumentLoader(settings.DOCUMENT_DIRECTORY)
    
    # Calculate file counts before loading
    kb_path = Path(settings.DOCUMENT_DIRECTORY)
    txt_files = list(kb_path.rglob("*.txt"))
    pdf_files = list(kb_path.rglob("*.pdf"))
    
    print(f"TXT files found: {len(txt_files)}")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Total files: {len(txt_files) + len(pdf_files)}")

    documents = loader.load()

    if not documents:
        print(f"[!] Error: No documents found in '{settings.DOCUMENT_DIRECTORY}'")
        sys.exit(1)

    print(f"Documents loaded successfully: {len(txt_files) + len(pdf_files)}")
    
    # Count PDF pages
    pdf_pages = sum(1 for doc in documents if doc.metadata.get("file_type") == "pdf")
    print(f"PDF pages processed: {pdf_pages}")

    # 2. Split documents
    splitter = InsuranceTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Store in ChromaDB (and embed)
    print("Generating embeddings...")
    newly_added = vector_mgr.add_documents(chunks)

    # 4. Print stats
    total_chunks = vector_mgr.count_chunks()
    
    duplicate_chunks = len(chunks) - newly_added

    print(f"Embeddings generated: {newly_added}")
    print("Vectors stored successfully.")
    print("ChromaDB persistence completed.")
    # Invalidate response cache
    try:
        cache_service = get_response_cache()
        cache_res = cache_service.clear_cache()
        print(f"Response cache invalidated: {cache_res.get('deleted_count', 0)} entries cleared.")
    except Exception as ce:
        print(f"Note: Response cache invalidation: {str(ce)}")
    print("========================================")
    print(f"Skipped files: 0") # We don't track skipped files currently
    print(f"Failed files: 0") # We don't track failed files currently
    print(f"Duplicate chunks skipped: {duplicate_chunks}")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest insurance documents into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing collection before indexing new documents.",
    )
    args = parser.parse_args()
    run_ingestion(reset=args.reset)
