"""Hallucination detection and answer grounding validation service."""
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from ..utils.logger import get_logger

logger = get_logger("insurance_rag.hallucination_checker")

# Explicit error / refusal phrases that indicate ungrounded or fallback responses
UNCACHEABLE_PATTERNS = [
    r"could not find this information",
    r"not configured in backend/\.env",
    r"too large to process",
    r"something went wrong",
    r"unable to generate",
    r"i don'?t have enough information",
    r"not mentioned in the (provided|available) context",
    r"no relevant documents? found",
    r"error occurred while processing",
]

COMPILED_UNCACHEABLE = [re.compile(p, re.IGNORECASE) for p in UNCACHEABLE_PATTERNS]


class ValidationResult:
    """Result of answer grounding and hallucination validation."""

    def __init__(
        self,
        is_grounded: bool,
        is_hallucination: bool,
        hallucination_score: float,
        reason: str = "",
    ) -> None:
        self.is_grounded = is_grounded
        self.is_hallucination = is_hallucination
        self.hallucination_score = round(hallucination_score, 4)
        self.reason = reason

    @property
    def is_cacheable(self) -> bool:
        """Only grounded answers without hallucination are safe to cache."""
        return self.is_grounded and not self.is_hallucination

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "is_hallucination": self.is_hallucination,
            "hallucination_score": self.hallucination_score,
            "is_cacheable": self.is_cacheable,
            "reason": self.reason,
        }


class HallucinationChecker:
    """Validates that generated insurance answers are grounded strictly in retrieved context."""

    def __init__(self, hallucination_threshold: float = 0.3) -> None:
        self.hallucination_threshold = hallucination_threshold

    def validate(
        self,
        answer: Optional[str],
        documents: List[Document],
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidationResult:
        """Validate an answer against retrieved context documents."""
        # 1. Empty or non-string check
        if not answer or not isinstance(answer, str) or not answer.strip():
            logger.debug("Validation failed: Answer is empty.")
            return ValidationResult(
                is_grounded=False,
                is_hallucination=True,
                hallucination_score=1.0,
                reason="Empty or blank answer",
            )

        trimmed_answer = answer.strip()
        if len(trimmed_answer) < 10:
            logger.debug("Validation failed: Answer is too short.")
            return ValidationResult(
                is_grounded=False,
                is_hallucination=True,
                hallucination_score=1.0,
                reason="Answer too brief (<10 chars)",
            )

        # 2. Check for explicit fallback / uncacheable refusal phrases
        for pattern in COMPILED_UNCACHEABLE:
            if pattern.search(trimmed_answer):
                logger.info(f"Refusal/fallback phrase detected in answer: '{pattern.pattern}'")
                return ValidationResult(
                    is_grounded=False,
                    is_hallucination=True,
                    hallucination_score=1.0,
                    reason=f"Detected refusal or fallback phrase matching {pattern.pattern}",
                )

        # 3. Source documents check
        if not documents and not sources:
            logger.debug("Validation failed: No source documents provided.")
            return ValidationResult(
                is_grounded=False,
                is_hallucination=True,
                hallucination_score=1.0,
                reason="No source documents provided",
            )

        # Build combined context text
        context_text = " ".join([doc.page_content for doc in documents]).lower()
        if not context_text.strip():
            return ValidationResult(
                is_grounded=False,
                is_hallucination=True,
                hallucination_score=1.0,
                reason="Source documents contain empty context",
            )

        answer_lower = trimmed_answer.lower()

        # 4. Numeric & waiting period grounding check
        # Extract numerical quantities and periods (e.g. 24 months, 30 days, 50%, $500)
        num_patterns = re.findall(r"\b\d+(?:[\.,]\d+)?(?:\s*(?:months?|days?|years?|%|percent|inr|rs|usd|\$))?\b", answer_lower)
        unsupported_facts = 0
        total_facts = len(num_patterns)

        for fact in num_patterns:
            fact_clean = fact.strip()
            # If the specific number or period isn't mentioned anywhere in context
            if fact_clean and fact_clean not in context_text:
                unsupported_facts += 1
                logger.debug(f"Fact '{fact_clean}' from answer not found in source context.")

        fact_hallucination_ratio = (unsupported_facts / total_facts) if total_facts > 0 else 0.0

        # 5. Lexical overlap of meaningful keywords (length >= 4, excluding standard stop words)
        stop_words = {
            "this", "that", "with", "from", "have", "were", "what", "when",
            "where", "which", "your", "their", "about", "would", "could",
            "should", "please", "insurance", "policy", "terms", "conditions"
        }
        answer_words = [
            w for w in re.findall(r"\b[a-z]{4,}\b", answer_lower)
            if w not in stop_words
        ]

        if answer_words:
            matched_words = sum(1 for w in answer_words if w in context_text)
            overlap_ratio = matched_words / len(answer_words)
            lexical_hallucination_ratio = max(0.0, 1.0 - overlap_ratio)
        else:
            lexical_hallucination_ratio = 0.0

        # Weighted hallucination score: 60% factual/numerical, 40% lexical
        hallucination_score = (0.6 * fact_hallucination_ratio) + (0.4 * lexical_hallucination_ratio)
        is_hallucination = hallucination_score > self.hallucination_threshold
        is_grounded = not is_hallucination

        reason = "Passes grounding and hallucination check" if is_grounded else (
            f"Hallucination score {hallucination_score:.2f} exceeded threshold {self.hallucination_threshold:.2f}"
        )

        logger.info(
            f"Hallucination check: score={hallucination_score:.2f}, "
            f"grounded={is_grounded}, hallucination={is_hallucination}"
        )

        return ValidationResult(
            is_grounded=is_grounded,
            is_hallucination=is_hallucination,
            hallucination_score=hallucination_score,
            reason=reason,
        )


_checker: Optional[HallucinationChecker] = None


def get_hallucination_checker() -> HallucinationChecker:
    """Singleton getter for HallucinationChecker."""
    global _checker
    if _checker is None:
        _checker = HallucinationChecker()
    return _checker
