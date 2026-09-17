"""LLM-powered Query Contextualization and Rewriting for Conversational RAG."""
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.query_rewriter")

QUERY_REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant for an insurance knowledge base.

Your task is to convert the user's latest question into a complete standalone question.

Use the conversation history to resolve pronouns, references, and follow-up questions.

CRITICAL RULES:
1. Do not answer the question.
2. Do not invent information.
3. Only rewrite the question into a single, complete, clear question suitable for vector search.
4. If the latest question is already a complete, standalone question, return it unchanged.
5. If the latest question cannot be understood even with the conversation history, return the latest question unchanged.
6. Output ONLY the rewritten question with no introductory or explanatory text."""

QUERY_REWRITE_USER_PROMPT = """Conversation History:
{chat_history}

Latest User Question:
{user_question}

Standalone Question:"""


class QueryRewriter:
    """Reformulates conversational follow-up questions into standalone queries using prior context."""

    def __init__(self, llm: Optional[ChatGroq] = None) -> None:
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", QUERY_REWRITE_SYSTEM_PROMPT),
            ("human", QUERY_REWRITE_USER_PROMPT),
        ])
        self.output_parser = StrOutputParser()
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize ChatGroq client if configured."""
        if self.llm is not None:
            return

        if not settings.has_groq_key:
            logger.warning("GROQ_API_KEY is not configured. Query contextualization will be bypassed.")
            self.llm = None
            return

        try:
            self.llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=0.0,
                max_retries=2,
            )
            logger.info("QueryRewriter ChatGroq client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq for QueryRewriter: {str(e)}", exc_info=True)
            self.llm = None

    def rewrite(self, user_question: str, chat_history: str) -> str:
        """Rewrite user question into a standalone query using conversation history.

        Returns original question if history is empty, question is already standalone,
        or if rewriting encounters any error.
        """
        trimmed_question = user_question.strip()
        trimmed_history = chat_history.strip() if chat_history else ""

        # Step 1: If there is no previous conversation history, the question is already standalone
        if not trimmed_history:
            return trimmed_question

        # Step 2: Ensure LLM is available
        if not self.llm:
            self._initialize_llm()
            if not self.llm:
                logger.info("LLM unavailable for query rewriting; using original question.")
                return trimmed_question

        # Step 3: Invoke rewriting chain
        try:
            chain = self.prompt_template | self.llm | self.output_parser
            logger.info(f"Rewriting follow-up question: '{trimmed_question}' with history length {len(trimmed_history)}")
            raw_output = chain.invoke({
                "chat_history": trimmed_history,
                "user_question": trimmed_question,
            })

            cleaned = self._clean_rewritten_query(raw_output, fallback=trimmed_question)
            logger.info(f"Query rewrite complete: '{trimmed_question}' -> '{cleaned}'")
            return cleaned

        except Exception as e:
            logger.warning(f"Query rewriting failed with error: {str(e)}. Falling back to original question.")
            return trimmed_question

    def _clean_rewritten_query(self, raw_output: str, fallback: str) -> str:
        """Clean prefixes, quotes, and markdown from LLM output."""
        if not raw_output or not raw_output.strip():
            return fallback

        cleaned = raw_output.strip()

        # Remove "Standalone Question:", "Rewritten Question:", etc.
        cleaned = re.sub(
            r"^(standalone\s*question|rewritten\s*question|question)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        # Remove outer quotation marks
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        # Normalize unicode dashes/hyphens to ASCII hyphen
        cleaned = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", cleaned)
        # Normalize smart quotes to standard quotes
        cleaned = re.sub(r"[\u2018\u2019]", "'", cleaned)
        cleaned = re.sub(r"[\u201C\u201D]", '"', cleaned)

        # Discard multi-line explanations if the model generated any extra text
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            cleaned = lines[0]

        if not cleaned or len(cleaned) < 2:
            return fallback

        return cleaned


# Global singleton
_query_rewriter: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter
