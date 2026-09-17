"""Insurance document retriever with similarity threshold filtering."""
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

from ..config import settings
from .vector_store import get_vector_store_manager, ChromaVectorStoreManager
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.retriever")


class InsuranceRetriever:
    """Retrieves relevant insurance policy chunks from ChromaDB based on semantic similarity."""

    def __init__(
        self,
        top_k: int = None,
        similarity_threshold: float = None,
        vector_store_manager: ChromaVectorStoreManager = None,
    ) -> None:
        self.top_k = top_k if top_k is not None else settings.RETRIEVAL_CANDIDATES
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.SIMILARITY_THRESHOLD
        )
        self.vector_store_manager = vector_store_manager or get_vector_store_manager()

    def retrieve(self, query: str) -> List[Document]:
        """Execute similarity search against ChromaDB, filter by similarity threshold,

        and return matching Document objects.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            logger.warning("Empty query passed to retriever.")
            return []

        logger.info(
            f"Retrieving top {self.top_k} documents for query: '{cleaned_query}' "
            f"(threshold: {self.similarity_threshold})"
        )

        raw_results: List[Tuple[Document, float]] = (
            self.vector_store_manager.similarity_search_with_score(cleaned_query, k=self.top_k)
        )

        filtered_documents: List[Document] = []

        for doc, distance in raw_results:
            # For normalized embeddings in ChromaDB, cosine distance is in [0, 2]
            # Convert distance to similarity score in range [0, 1]
            similarity_score = max(0.0, 1.0 - (distance / 2.0))
            doc.metadata["similarity_score"] = round(similarity_score, 4)
            doc.metadata["distance"] = round(distance, 4)

            # Ensure source and filename keys exist
            filename = doc.metadata.get("filename") or doc.metadata.get("source", "policy_doc.txt")
            doc.metadata["filename"] = filename
            doc.metadata["source"] = filename

            logger.debug(
                f"Candidate chunk: {doc.metadata.get('chunk_id')} | "
                f"Score: {similarity_score:.4f} | Dist: {distance:.4f}"
            )

            if similarity_score >= self.similarity_threshold:
                filtered_documents.append(doc)
            else:
                logger.debug(
                    f"Discarding chunk {doc.metadata.get('chunk_id')} - "
                    f"score {similarity_score:.4f} below threshold {self.similarity_threshold}"
                )

        logger.info(
            f"Retrieved {len(filtered_documents)} documents passing similarity threshold "
            f"out of {len(raw_results)} candidates."
        )
        return filtered_documents


def get_retriever() -> InsuranceRetriever:
    """Factory helper to obtain retriever instance."""
    return InsuranceRetriever()
