"""Document loader for insurance knowledge base."""
import os
import hashlib
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from ..utils.logger import get_logger

logger = get_logger("insurance_rag.loader")


def determine_document_type(filename: str) -> str:
    """Determine the insurance document type based on the file name prefix or keywords.

    Examples:
        health_* -> health
        motor_* -> motor
        life_* -> life
        travel_* -> travel
        home_* -> home
        claims_* / claim_* -> claims
        business_* -> business
        personal_* -> personal_accident
        policy_* -> policy_management
        complaints_* -> complaints
        customer_* / faq_* -> faq
        data_* / privacy_* -> privacy
        fraud_* -> fraud_prevention
        general_* -> general
        glossary_* -> glossary
    """
    lower = filename.lower()
    if lower.startswith("health_") or "health" in lower:
        return "health"
    if lower.startswith("motor_") or "motor" in lower:
        return "motor"
    if lower.startswith("life_") or "life" in lower:
        return "life"
    if lower.startswith("travel_") or "travel" in lower:
        return "travel"
    if lower.startswith("home_") or "home" in lower:
        return "home"
    if lower.startswith("business_") or "business" in lower:
        return "business"
    if lower.startswith("personal_") or "accident" in lower:
        return "personal_accident"
    if lower.startswith("claim_") or lower.startswith("claims_") or "claim" in lower:
        return "claims"
    if lower.startswith("policy_") or "renewal" in lower or "cancellation" in lower:
        return "policy_management"
    if "complaint" in lower or "grievance" in lower:
        return "complaints"
    if "faq" in lower or "customer" in lower:
        return "faq"
    if "privacy" in lower or "data" in lower or "security" in lower:
        return "privacy"
    if "fraud" in lower:
        return "fraud_prevention"
    if "glossary" in lower:
        return "glossary"
    if lower.startswith("general_") or "general" in lower:
        return "general"
    return "general"


def get_file_hash(file_path: Path) -> str:
    """Return a SHA256 hash of the file contents."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


class InsuranceDocumentLoader:
    """Loads and preprocesses text and PDF documents from the knowledge base directory."""

    def __init__(self, directory_path: Path) -> None:
        self.directory_path = Path(directory_path)

    def load(self) -> List[Document]:
        """Recursively scan directory for TXT and PDF files, load, and return Documents."""
        if not self.directory_path.exists():
            logger.error(f"Directory not found: {self.directory_path}")
            return []

        documents: List[Document] = []
        
        # Get all TXT and PDF files
        txt_files = list(self.directory_path.rglob("*.txt"))
        pdf_files = list(self.directory_path.rglob("*.pdf"))
        all_files = sorted(txt_files + pdf_files)

        logger.info(f"Discovered {len(txt_files)} TXT files and {len(pdf_files)} PDF files in {self.directory_path}")

        for file_path in all_files:
            try:
                # Check file size to avoid empty files
                if file_path.stat().st_size == 0:
                    logger.warning(f"Skipping empty file: {file_path.name}")
                    continue

                file_ext = file_path.suffix.lower()
                doc_type = determine_document_type(file_path.name)
                file_hash = get_file_hash(file_path)

                if file_ext == ".txt":
                    self._load_txt(file_path, doc_type, file_hash, documents)
                elif file_ext == ".pdf":
                    self._load_pdf(file_path, doc_type, file_hash, documents)
                else:
                    logger.warning(f"Unsupported file type: {file_path.name}")

            except Exception as e:
                logger.error(f"Error loading file {file_path.name}: {str(e)}", exc_info=True)

        logger.info(f"Successfully loaded {len(documents)} document pages/files from knowledge base")
        return documents

    def _load_txt(self, file_path: Path, doc_type: str, file_hash: str, documents: List[Document]) -> None:
        content: Optional[str] = None
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    text = f.read()
                content = text
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            logger.error(f"Failed to decode file with tested encodings: {file_path.name}")
            return

        # Text Cleaning: Remove excessive newlines/spaces
        cleaned_content = "\n".join([line.strip() for line in content.split("\n") if line.strip()])
        if not cleaned_content:
            logger.warning(f"Skipping blank content file: {file_path.name}")
            return

        metadata = {
            "source": file_path.name,
            "filename": file_path.name,
            "file_type": "txt",
            "document_type": doc_type,
            "file_hash": file_hash,
            "file_path": str(file_path),
        }

        doc = Document(page_content=cleaned_content, metadata=metadata)
        documents.append(doc)
        logger.debug(f"Loaded TXT {file_path.name} (type: {doc_type}, {len(cleaned_content)} chars)")

    def _load_pdf(self, file_path: Path, doc_type: str, file_hash: str, documents: List[Document]) -> None:
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            if not pages:
                logger.warning(f"Skipping empty PDF file: {file_path.name}")
                return
                
            for i, page in enumerate(pages):
                # Text Cleaning: Remove excessive newlines/spaces
                cleaned_content = "\n".join([line.strip() for line in page.page_content.split("\n") if line.strip()])
                if len(cleaned_content) < 10:  # Skip mostly empty pages
                    continue
                    
                page_num = page.metadata.get("page", i) + 1  # 1-indexed

                metadata = {
                    "source": file_path.name,
                    "filename": file_path.name,
                    "file_type": "pdf",
                    "document_type": doc_type,
                    "file_hash": file_hash,
                    "file_path": str(file_path),
                    "page": page_num,
                }

                doc = Document(page_content=cleaned_content, metadata=metadata)
                documents.append(doc)
                
            logger.debug(f"Loaded PDF {file_path.name} (type: {doc_type}, {len(pages)} pages)")
        except Exception as e:
            logger.error(f"Failed to extract PDF {file_path.name}: {str(e)}")
