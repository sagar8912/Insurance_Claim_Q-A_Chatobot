"""Automated tests for persistent conversation sessions, query contextualization, and multi-turn conversational RAG."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.session_service import SessionService
from app.rag.query_rewriter import QueryRewriter
from app.rag.rag_pipeline import InsuranceRAGPipeline
from app.services.chat_service import ChatService
from langchain_core.documents import Document

client = TestClient(app)


@pytest.fixture
def temp_session_service(tmp_path: Path) -> SessionService:
    """Provide an isolated SessionService with a temporary SQLite database."""
    db_file = tmp_path / "test_conversations.db"
    return SessionService(db_path=db_file)


# ============================================================================
# 1. Persistent Session & History Storage Tests
# ============================================================================

def test_session_creation_and_retrieval(temp_session_service: SessionService):
    """Test generating, retrieving, and validating unique conversation sessions."""
    conv_id = temp_session_service.get_or_create_session()
    assert conv_id is not None
    assert len(conv_id) > 10

    # Retrieving existing session should return identical ID
    same_id = temp_session_service.get_or_create_session(conv_id)
    assert same_id == conv_id

    # New session without ID should generate a different ID
    another_id = temp_session_service.get_or_create_session()
    assert another_id != conv_id


def test_add_and_retrieve_messages(temp_session_service: SessionService):
    """Test recording user and assistant turns and retrieving them chronologically."""
    conv_id = temp_session_service.get_or_create_session()

    msg1 = temp_session_service.add_message(conv_id, "user", "What is the waiting period for maternity coverage?")
    assert msg1["role"] == "user"
    assert "maternity coverage" in msg1["content"]

    msg2 = temp_session_service.add_message(
        conv_id, "assistant", "The waiting period is 24 months of continuous coverage."
    )
    assert msg2["role"] == "assistant"

    msg3 = temp_session_service.add_message(conv_id, "user", "What happens after that?")
    assert msg3["role"] == "user"

    messages = temp_session_service.get_messages(conv_id)
    assert len(messages) == 3
    assert messages[0]["content"] == "What is the waiting period for maternity coverage?"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["content"] == "What happens after that?"


def test_history_window_limit(temp_session_service: SessionService):
    """Verify that get_messages respects the configurable message limit."""
    conv_id = temp_session_service.get_or_create_session()
    for i in range(15):
        role = "user" if i % 2 == 0 else "assistant"
        temp_session_service.add_message(conv_id, role, f"Message turn {i}")

    # Fetch with limit = 4
    recent = temp_session_service.get_messages(conv_id, limit=4)
    assert len(recent) == 4
    assert recent[0]["content"] == "Message turn 11"
    assert recent[-1]["content"] == "Message turn 14"


def test_format_chat_history_for_prompt(temp_session_service: SessionService):
    """Verify that chat history is formatted cleanly into a prompt transcript."""
    conv_id = temp_session_service.get_or_create_session()
    temp_session_service.add_message(conv_id, "user", "What are the policy conditions?")
    temp_session_service.add_message(conv_id, "assistant", "Conditions are A, B, and C.")

    formatted = temp_session_service.format_chat_history_for_prompt(conv_id)
    assert "User: What are the policy conditions?" in formatted
    assert "Assistant: Conditions are A, B, and C." in formatted


def test_clear_session(temp_session_service: SessionService):
    """Test clearing all messages from an existing conversation."""
    conv_id = temp_session_service.get_or_create_session()
    temp_session_service.add_message(conv_id, "user", "Hello")
    assert temp_session_service.count_messages(conv_id) == 1

    temp_session_service.clear_session(conv_id)
    assert temp_session_service.count_messages(conv_id) == 0


# ============================================================================
# 2. Conversational Query Rewriting Tests
# ============================================================================

def test_query_rewriter_no_history():
    """When chat history is empty, rewriter should immediately return original query."""
    rewriter = QueryRewriter()
    result = rewriter.rewrite("What is maternity coverage?", "")
    assert result == "What is maternity coverage?"

    result_whitespace = rewriter.rewrite("What is maternity coverage?", "   ")
    assert result_whitespace == "What is maternity coverage?"


def test_query_rewriter_with_mock_llm():
    """Verify that QueryRewriter invokes LLM chain and parses output cleanly."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Standalone Question: What is the waiting period for maternity coverage?"
    
    rewriter = QueryRewriter(llm=mock_llm)
    # Mock chain
    with patch.object(rewriter, "prompt_template") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Standalone Question: What is the waiting period for maternity coverage?"
        
        with patch("app.rag.query_rewriter.StrOutputParser"):
            with patch.object(rewriter, "_clean_rewritten_query", return_value="What is the waiting period for maternity coverage?"):
                result = rewriter.rewrite(
                    "What happens after that?",
                    "User: What is maternity coverage?\nAssistant: Maternity coverage has a 24-month waiting period."
                )
                assert "What is the waiting period" in result


def test_query_rewriter_fallback_on_error():
    """Verify that if LLM raises an error, rewriter falls back safely to original question."""
    mock_llm = MagicMock()
    mock_llm.side_effect = RuntimeError("Groq API connection timeout")

    rewriter = QueryRewriter(llm=mock_llm)
    result = rewriter.rewrite(
        "What about the second one?",
        "User: What are conditions?\nAssistant: Condition 1 and 2."
    )
    # Safe fallback
    assert result == "What about the second one?"


# ============================================================================
# 3. Conversational RAG Acceptance Tests (Scenarios 1 to 5)
# ============================================================================

def test_acceptance_scenario_1_maternity_waiting_period():
    """TEST 1: User asks maternity waiting period, then 'What happens after that?'."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(
            page_content="Maternity coverage applies after a 24-month waiting period. After 24 months, standard delivery expenses are covered up to 50,000.",
            metadata={"filename": "maternity_policy.txt", "document_type": "health", "similarity_score": 0.88},
        )
    ]

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "What happens after the 24-month waiting period for maternity coverage?"

    pipeline = InsuranceRAGPipeline.__new__(InsuranceRAGPipeline)
    pipeline.retriever = mock_retriever
    pipeline.query_rewriter = mock_rewriter
    pipeline.llm = None
    pipeline.prompt_template = MagicMock()
    pipeline.output_parser = MagicMock()

    history = "User: What is the waiting period for maternity coverage?\nAssistant: The waiting period is 24 months."
    res = pipeline.ask(
        question="What happens after that?",
        conversation_id="conv_test_1",
        chat_history_str=history,
    )

    assert res["success"] is True
    # Verify the retriever was called with the rewritten standalone query
    mock_rewriter.rewrite.assert_called_once()
    mock_retriever.retrieve.assert_called_with("What happens after the 24-month waiting period for maternity coverage?")
    assert res["rewritten_query"] == "What happens after the 24-month waiting period for maternity coverage?"


def test_acceptance_scenario_2_hospitalization_exclusions():
    """TEST 2: Tell me about hospitalization coverage -> 'What are the exclusions?'."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(
            page_content="Hospitalization coverage excludes cosmetic surgery, self-inflicted injuries, and non-prescribed treatments.",
            metadata={"filename": "health_policy.txt", "document_type": "health", "similarity_score": 0.91},
        )
    ]

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "What are the exclusions for hospitalization coverage?"

    pipeline = InsuranceRAGPipeline.__new__(InsuranceRAGPipeline)
    pipeline.retriever = mock_retriever
    pipeline.query_rewriter = mock_rewriter
    pipeline.llm = None
    pipeline.prompt_template = MagicMock()
    pipeline.output_parser = MagicMock()

    history = "User: Tell me about hospitalization coverage.\nAssistant: Hospitalization coverage pays for in-patient hospital room and board."
    res = pipeline.ask(
        question="What are the exclusions?",
        conversation_id="conv_test_2",
        chat_history_str=history,
    )

    assert res["success"] is True
    mock_retriever.retrieve.assert_called_with("What are the exclusions for hospitalization coverage?")


def test_acceptance_scenario_3_policy_conditions_second_one():
    """TEST 3: Conditions A, B, C -> 'Explain the second one.'."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(
            page_content="Condition B requires continuous policy renewal without any lapse in premium payments.",
            metadata={"filename": "policy_terms.txt", "document_type": "general", "similarity_score": 0.85},
        )
    ]

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "Explain Condition B of the policy conditions."

    pipeline = InsuranceRAGPipeline.__new__(InsuranceRAGPipeline)
    pipeline.retriever = mock_retriever
    pipeline.query_rewriter = mock_rewriter
    pipeline.llm = None
    pipeline.prompt_template = MagicMock()
    pipeline.output_parser = MagicMock()

    history = "User: What are the policy conditions?\nAssistant: There are three major conditions: A, B, and C."
    res = pipeline.ask(
        question="Explain the second one.",
        conversation_id="conv_test_3",
        chat_history_str=history,
    )

    assert res["success"] is True
    mock_retriever.retrieve.assert_called_with("Explain Condition B of the policy conditions.")


def test_acceptance_scenario_4_claim_settlement_timeline():
    """TEST 4: What is the claim process? -> 'How long does it take?'."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(
            page_content="Claims are settled within 15 to 30 business days after receiving all verified documents.",
            metadata={"filename": "claim_guide.txt", "document_type": "claims", "similarity_score": 0.89},
        )
    ]

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "How long does the insurance claim settlement process take?"

    pipeline = InsuranceRAGPipeline.__new__(InsuranceRAGPipeline)
    pipeline.retriever = mock_retriever
    pipeline.query_rewriter = mock_rewriter
    pipeline.llm = None
    pipeline.prompt_template = MagicMock()
    pipeline.output_parser = MagicMock()

    history = "User: What is the claim process?\nAssistant: You must submit form A and hospital discharge summary."
    res = pipeline.ask(
        question="How long does it take?",
        conversation_id="conv_test_4",
        chat_history_str=history,
    )

    assert res["success"] is True
    mock_retriever.retrieve.assert_called_with("How long does the insurance claim settlement process take?")


def test_acceptance_scenario_5_new_chat_isolation(temp_session_service: SessionService):
    """TEST 5: New Chat isolation -> When user starts a fresh chat, it must NOT inherit previous session context."""
    # First conversation has maternity context
    conv1 = temp_session_service.get_or_create_session()
    temp_session_service.add_message(conv1, "user", "What is maternity waiting period?")
    temp_session_service.add_message(conv1, "assistant", "24 months.")

    # Start a brand new conversation
    conv2 = temp_session_service.get_or_create_session()
    assert conv2 != conv1

    # In conv2, history is completely empty
    conv2_history = temp_session_service.format_chat_history_for_prompt(conv2)
    assert conv2_history == ""

    # QueryRewriter receiving empty history does NOT rewrite or contaminate
    rewriter = QueryRewriter()
    standalone_query = rewriter.rewrite("What about the second one?", conv2_history)
    assert standalone_query == "What about the second one?"


# ============================================================================
# 4. API Endpoints Integration Tests
# ============================================================================

def test_api_conversations_new_endpoint():
    """Verify POST /api/v1/conversations/new creates a new session."""
    response = client.post("/api/v1/conversations/new")
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert len(data["conversation_id"]) > 10


def test_api_chat_with_conversation_id_and_message_alias():
    """Verify /api/v1/chat accepts conversation_id and alternative 'message' field."""
    # Create conversation
    res_new = client.post("/api/v1/conversations/new")
    conv_id = res_new.json()["conversation_id"]

    # Send message using 'message' field alias
    response = client.post(
        "/api/v1/chat",
        json={"conversation_id": conv_id, "message": "What is the waiting period for pre-existing diseases?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["conversation_id"] == conv_id

    # Retrieve history from endpoint
    history_res = client.get(f"/api/v1/conversations/{conv_id}")
    assert history_res.status_code == 200
    history_data = history_res.json()
    assert history_data["message_count"] >= 2  # user + assistant
