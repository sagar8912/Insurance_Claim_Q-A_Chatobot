"""RAG Pipeline integrating Retriever, Prompt, and Groq LLM."""
import re
import time
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from ..config import settings
from .retriever import get_retriever, InsuranceRetriever
from .prompt import get_rag_prompt_template
from .context_manager import ContextBudgetManager
from .query_rewriter import get_query_rewriter, QueryRewriter
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.pipeline")


class InsuranceRAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline for Insurance Q&A."""

    def __init__(
        self,
        retriever: Optional[InsuranceRetriever] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self.retriever = retriever or get_retriever()
        self.query_rewriter = query_rewriter or get_query_rewriter()
        self.llm: Optional[ChatGroq] = None
        self.prompt_template = get_rag_prompt_template()
        self.output_parser = StrOutputParser()
        self.initialized = False
        self.initialize()

    def initialize(self) -> None:
        """Initialize the Groq LLM client and verify configuration."""
        if not settings.has_groq_key:
            logger.warning(
                "GROQ_API_KEY is not set or empty. "
                "LLM generation will fall back to context summary mode until an API key is provided."
            )
            self.llm = None
            self.initialized = True
            return

        try:
            logger.info(f"Initializing ChatGroq with model '{settings.GROQ_MODEL}' (temp: 0.1)")
            self.llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=0.1,
                max_retries=2,
            )
            self.initialized = True
            logger.info("ChatGroq LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq: {str(e)}", exc_info=True)
            self.llm = None
            self.initialized = False

    def retrieve_context(self, question: str) -> List[Document]:
        """Retrieve relevant document chunks for the question."""
        return self.retriever.retrieve(question)

    def format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format retrieved documents into structured source citation objects,

        deduplicating by filename and page while preserving key previews.
        """
        seen_files = set()
        sources: List[Dict[str, Any]] = []

        for doc in documents:
            filename = doc.metadata.get("filename", "unknown.txt")
            page = doc.metadata.get("page")
            
            # Deduplicate by filename + page so multiple pages from the same PDF can be cited
            dedup_key = f"{filename}_page_{page}" if page else filename
            
            if dedup_key in seen_files:
                continue
            seen_files.add(dedup_key)

            doc_type = doc.metadata.get("document_type", "general")
            file_type = doc.metadata.get("file_type", "txt")
            content = doc.page_content.strip()
            # Clean up preview text
            preview = " ".join(content.split())
            if len(preview) > 220:
                preview = preview[:217] + "..."

            source_dict = {
                "filename": filename,
                "file_type": file_type,
                "document_type": doc_type,
                "preview": preview,
                "similarity_score": doc.metadata.get("similarity_score", 0.0),
            }
            if page is not None:
                source_dict["page"] = page
                
            sources.append(source_dict)

        return sources

    def generate_answer(
        self, question: str, documents: List[Document], chat_history: str = ""
    ) -> str:
        """Generate answer via Groq LLM using retrieved context and recent conversation history."""
        if not documents:
            return (
                "I could not find this information in the available insurance policy documents.\n\n"
                "Final coverage and claim eligibility depend on the specific policy terms and conditions. "
                "Please contact customer support for detailed assistance."
            )

        # Build context string
        context_parts = []
        for i, doc in enumerate(documents, 1):
            fname = doc.metadata.get("filename", "document")
            context_parts.append(f"[Document {i}: {fname}]\n{doc.page_content}\n")
        context_str = "\n".join(context_parts)

        # Check LLM availability
        if not self.llm:
            logger.warning("Groq LLM is not configured. Providing context summary.")
            return (
                "**[Notice: GROQ_API_KEY is not configured in backend/.env]**\n\n"
                "The relevant policy documents were successfully retrieved from the knowledge base:\n\n"
                f"{documents[0].page_content[:400]}...\n\n"
                "To enable full AI response generation, please add your Groq API key to `backend/.env` "
                "and restart the backend.\n\n"
                "*Final coverage and claim eligibility depend on the specific policy terms and conditions.*"
            )

        chain = self.prompt_template | self.llm | self.output_parser
        logger.info(f"Sending prompt to Groq model {settings.GROQ_MODEL}")
        history_val = chat_history.strip() if chat_history else "None (First turn in conversation)"
        answer = chain.invoke({
            "context": context_str,
            "question": question,
            "chat_history": history_val,
        })
        raw_answer = answer.strip()
        clean_answer = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", raw_answer)
        clean_answer = re.sub(r"[\u2018\u2019]", "'", clean_answer)
        clean_answer = re.sub(r"[\u201C\u201D]", '"', clean_answer)
        clean_answer = re.sub(r"[\u202f\u00a0]", " ", clean_answer)
        return clean_answer

    def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        chat_history_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute conversational RAG pipeline: rewrite query -> retrieve -> generate -> format sources."""
        start_time = time.perf_counter()
        clean_question = question.strip()
        history_text = (chat_history_str or "").strip()

        logger.info(f"[CHAT REQUEST] conversation_id: {conversation_id or 'none'}")
        logger.info(f"[USER QUERY] original_query: '{clean_question}'")
        logger.info(f"[CHAT HISTORY] length: {len(history_text)} chars")

        # Step 1: Query Contextualization / Rewriting for follow-up questions
        standalone_query = clean_question
        if history_text:
            standalone_query = self.query_rewriter.rewrite(clean_question, history_text)

        logger.info(f"[QUERY REWRITE] rewritten_query: '{standalone_query}'")

        # Step 2: Context Retrieval with fallback retry
        raw_documents = self.retrieve_context(standalone_query)

        # Fallback logic: if rewritten query retrieval returns no documents, retry with original query
        if not raw_documents and standalone_query != clean_question:
            logger.info("Rewritten query yielded 0 documents. Retrying retrieval with original query...")
            raw_documents = self.retrieve_context(clean_question)

        # Step 3: Apply strict context budget
        budget_manager = ContextBudgetManager()
        documents = budget_manager.apply_budget(raw_documents)
        sources = self.format_sources(documents)

        scores = [round(d.metadata.get("similarity_score", 0.0), 3) for d in documents]
        logger.info(f"[RETRIEVAL] documents_found: {len(documents)}")
        logger.info(f"[RETRIEVAL SCORES] scores: {scores}")

        try:
            answer = self.generate_answer(clean_question, documents, chat_history=history_text)
            logger.info("[LLM RESPONSE] generated successfully")
        except Exception as e:
            error_str = str(e).lower()
            if "413" in error_str or "request_too_large" in error_str or "entity too large" in error_str:
                if settings.MAX_CONTEXT_RETRY > 0:
                    logger.warning("Groq API returned 413. Attempting context reduction retry...")
                    # Halve the context budget constraints to force a smaller payload
                    reduced_top_k = max(1, settings.FINAL_TOP_K // 2)
                    reduced_chars = max(1000, settings.MAX_CONTEXT_CHARS // 2)

                    documents = budget_manager.apply_budget(
                        raw_documents,
                        char_limit_override=reduced_chars,
                        top_k_override=reduced_top_k,
                    )
                    sources = self.format_sources(documents)

                    try:
                        answer = self.generate_answer(clean_question, documents, chat_history=history_text)
                    except Exception as retry_e:
                        logger.error(f"Fallback retry failed: {str(retry_e)}", exc_info=True)
                        return {
                            "success": False,
                            "answer": None,
                            "error": "The retrieved policy information was too large to process. Please try again. The system has limited the context and will continue processing future requests safely.",
                            "sources": [],
                            "grounded": False,
                            "processing_time": round(time.perf_counter() - start_time, 2),
                            "original_query": clean_question,
                            "standalone_query": standalone_query,
                            "conversation_id": conversation_id,
                        }
                else:
                    logger.error("Groq API returned 413 and retries are disabled.", exc_info=True)
                    return {
                        "success": False,
                        "answer": None,
                        "error": "The retrieved policy information was too large to process. Please try again.",
                        "sources": [],
                        "grounded": False,
                        "processing_time": round(time.perf_counter() - start_time, 2),
                        "original_query": clean_question,
                        "standalone_query": standalone_query,
                        "conversation_id": conversation_id,
                    }
            else:
                logger.error(f"Error during Groq LLM generation: {str(e)}", exc_info=True)
                return {
                    "success": False,
                    "answer": None,
                    "error": "Something went wrong while processing your question.",
                    "sources": [],
                    "grounded": False,
                    "processing_time": round(time.perf_counter() - start_time, 2),
                    "original_query": clean_question,
                    "standalone_query": standalone_query,
                    "conversation_id": conversation_id,
                }

        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info(f"RAG query completed in {elapsed}s with {len(sources)} sources cited.")

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "grounded": len(sources) > 0,
            "processing_time": elapsed,
            "original_query": clean_question,
            "rewritten_query": standalone_query if standalone_query != clean_question else None,
            "standalone_query": standalone_query,
            "conversation_id": conversation_id,
        }


# Global singleton instance
_rag_pipeline: Optional[InsuranceRAGPipeline] = None


def get_rag_pipeline() -> InsuranceRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = InsuranceRAGPipeline()
    return _rag_pipeline
