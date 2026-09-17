"""Text splitter module for chunking insurance policy documents."""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.logger import get_logger

logger = get_logger("insurance_rag.splitter")


class InsuranceTextSplitter:
    """Splits documents into semantic chunks while maintaining metadata and tracking chunk IDs."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, min_chunk_len: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into chunked documents with unique chunk IDs and enriched metadata."""
        if not documents:
            logger.warning("No documents provided to text splitter.")
            return []

        all_chunks: List[Document] = []

        for doc in documents:
            filename = doc.metadata.get("filename", "unknown")
            doc_type = doc.metadata.get("document_type", "general")
            file_type = doc.metadata.get("file_type", "txt")
            file_hash = doc.metadata.get("file_hash", "")
            page = doc.metadata.get("page")
            source = doc.metadata.get("source", filename)

            raw_chunks = self.splitter.split_documents([doc])
            valid_chunk_idx = 0

            for chunk in raw_chunks:
                content = chunk.page_content.strip()
                # Discard tiny fragments unless it's the only chunk for this doc
                if len(content) < self.min_chunk_len and len(raw_chunks) > 1:
                    continue

                if file_type == "pdf" and page is not None:
                    chunk_id = f"{filename}_page_{page}_chunk_{valid_chunk_idx}"
                else:
                    chunk_id = f"{filename}_chunk_{valid_chunk_idx}"

                metadata = {
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": valid_chunk_idx,
                    "source": source,
                    "filename": filename,
                    "file_type": file_type,
                    "document_type": doc_type,
                    "file_hash": file_hash,
                    "char_count": len(content),
                }
                
                if page is not None:
                    metadata["page"] = page

                chunk_doc = Document(page_content=content, metadata=metadata)
                all_chunks.append(chunk_doc)
                valid_chunk_idx += 1

        logger.info(
            f"Split {len(documents)} documents into {len(all_chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks
