"""ChromaDB persistent vector store manager."""
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config import settings
from .embeddings import get_embedding_service
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.vector_store")


class ChromaVectorStoreManager:
    """Manages persistent ChromaDB vector store, collection creation, duplicate prevention,

    and similarity querying.
    """

    def __init__(self) -> None:
        self.persist_directory = Path(settings.CHROMA_PERSIST_DIRECTORY)
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self.embedding_service = get_embedding_service()

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client: Optional[chromadb.PersistentClient] = None
        self._vector_store: Optional[Chroma] = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize or reconnect to the persistent Chroma client and LangChain wrapper."""
        try:
            logger.info(
                f"Initializing ChromaDB at '{self.persist_directory}' "
                f"with collection '{self.collection_name}'"
            )
            try:
                from chromadb.api.client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._vector_store = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_service.embedder,
            )
            logger.info("ChromaDB vector store connected successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}", exc_info=True)
            raise

    def _is_client_alive(self) -> bool:
        """Verify if Chroma client and rust bindings are active and healthy."""
        if self._client is None or self._vector_store is None:
            return False
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False

    def _ensure_connected(self) -> None:
        """Reconnect to ChromaDB if client was stopped or disconnected."""
        if not self._is_client_alive():
            logger.info("ChromaDB connection lost or closed. Reconnecting...")
            self._initialize_store()

    @property
    def vector_store(self) -> Chroma:
        self._ensure_connected()
        return self._vector_store

    def _get_collection(self):
        """Get or create collection with auto-reconnect on stale bindings."""
        self._ensure_connected()
        try:
            return self._client.get_or_create_collection(self.collection_name)
        except Exception:
            self._initialize_store()
            return self._client.get_or_create_collection(self.collection_name)

    def count_chunks(self) -> int:
        """Return total number of stored vectors/chunks in the collection."""
        try:
            col = self._get_collection()
            return col.count()
        except Exception as e:
            logger.error(f"Error fetching chunk count: {str(e)}")
            return 0

    def get_indexed_document_count(self) -> int:
        """Return the number of unique documents indexed based on metadata."""
        try:
            col = self._get_collection()
            total = col.count()
            if total == 0:
                return 0
            results = col.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            unique_sources = {m.get("filename") for m in metadatas if m and m.get("filename")}
            return len(unique_sources)
        except Exception as e:
            logger.error(f"Error counting indexed documents: {str(e)}")
            return 0

    def get_existing_ids(self) -> set:
        """Return set of all chunk IDs currently in the collection."""
        try:
            col = self._client.get_or_create_collection(self.collection_name)
            total = col.count()
            if total == 0:
                return set()
            results = col.get()
            return set(results.get("ids", []))
        except Exception as e:
            logger.error(f"Error fetching existing IDs: {str(e)}")
            return set()

    def get_existing_file_hashes(self) -> dict:
        """Return dict of {filename: file_hash} for currently indexed documents."""
        try:
            col = self._client.get_or_create_collection(self.collection_name)
            total = col.count()
            if total == 0:
                return {}
            results = col.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            file_hashes = {}
            for m in metadatas:
                if m and m.get("filename") and m.get("file_hash"):
                    file_hashes[m["filename"]] = m["file_hash"]
            return file_hashes
        except Exception as e:
            logger.error(f"Error fetching existing hashes: {str(e)}")
            return {}

    def delete_by_filename(self, filename: str) -> None:
        """Delete all chunks associated with a specific filename."""
        try:
            col = self._client.get_or_create_collection(self.collection_name)
            results = col.get(where={"filename": filename})
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} outdated chunks for '{filename}'")
        except Exception as e:
            logger.error(f"Failed to delete chunks for {filename}: {str(e)}")

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> int:
        """Add documents to ChromaDB, preventing duplicates using file_hash.

        Returns number of newly added documents.
        """
        if not documents:
            logger.warning("No documents to add to vector store.")
            return 0

        existing_hashes = self.get_existing_file_hashes()
        docs_to_add: List[Document] = []
        ids_to_add: List[str] = []
        
        # Group incoming documents by filename to process them file-by-file
        from collections import defaultdict
        docs_by_file = defaultdict(list)
        for doc in documents:
            filename = doc.metadata.get("filename", "unknown")
            docs_by_file[filename].append(doc)

        for filename, file_docs in docs_by_file.items():
            file_hash = file_docs[0].metadata.get("file_hash", "")
            
            if filename in existing_hashes:
                if existing_hashes[filename] == file_hash:
                    # File exists and hasn't changed, skip entirely
                    logger.debug(f"Skipping unchanged file: {filename}")
                    continue
                else:
                    # File exists but hash differs (was modified), delete old chunks first
                    logger.info(f"File modified, updating vectors: {filename}")
                    self.delete_by_filename(filename)
            
            # Add all chunks for this new/modified file
            for doc in file_docs:
                chunk_id = doc.metadata.get("chunk_id")
                if not chunk_id:
                    chunk_id = f"{filename}_{hash(doc.page_content)}"
                    doc.metadata["chunk_id"] = chunk_id
                
                docs_to_add.append(doc)
                ids_to_add.append(chunk_id)

        if not docs_to_add:
            logger.info("All documents are already indexed and unchanged. No new chunks added.")
            return 0

        logger.info(f"Adding {len(docs_to_add)} new chunks to collection '{self.collection_name}'")
        try:
            # Add in batches to handle large datasets smoothly
            for i in range(0, len(docs_to_add), batch_size):
                batch_docs = docs_to_add[i : i + batch_size]
                batch_ids = ids_to_add[i : i + batch_size]
                self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                logger.debug(f"Indexed batch {i // batch_size + 1} ({len(batch_docs)} items)")

            logger.info(f"Successfully indexed {len(docs_to_add)} chunks into ChromaDB.")
            return len(docs_to_add)
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}", exc_info=True)
            raise

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search returning (Document, score)."""
        try:
            self._ensure_connected()
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.warning(f"ChromaDB similarity search error ({str(e)}). Reconnecting and retrying...")
            try:
                self._initialize_store()
                return self.vector_store.similarity_search_with_score(query, k=k)
            except Exception as retry_e:
                logger.error(f"ChromaDB similarity search failed after reconnect: {str(retry_e)}", exc_info=True)
                return []

    def reset_database(self) -> None:
        """Reset the vector store collection and remove persisted data."""
        logger.warning(f"Resetting vector store collection '{self.collection_name}'...")
        try:
            self._ensure_connected()
            try:
                self._client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'.")
            except Exception:
                pass

            # Re-initialize collection
            self._vector_store = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_service.embedder,
            )
            logger.info("Vector store reset complete.")
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}", exc_info=True)
            raise


# Global singleton instance
_store_manager: Optional[ChromaVectorStoreManager] = None


def get_vector_store_manager() -> ChromaVectorStoreManager:
    global _store_manager
    if _store_manager is None or not _store_manager._is_client_alive():
        _store_manager = ChromaVectorStoreManager()
    return _store_manager
