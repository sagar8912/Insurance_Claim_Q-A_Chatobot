"""API route tests using FastAPI TestClient."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test GET / endpoint returns running message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Insurance RAG API is running" in data["message"]


def test_health_endpoint():
    """Test GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data


def test_chat_validation_empty():
    """Test POST /api/v1/chat fails on whitespace-only or empty question."""
    response = client.post("/api/v1/chat", json={"question": "   "})
    assert response.status_code == 422


def test_chat_validation_too_short():
    """Test POST /api/v1/chat fails on 1 character question."""
    response = client.post("/api/v1/chat", json={"question": "a"})
    assert response.status_code == 422


def test_chat_validation_missing():
    """Test POST /api/v1/chat fails on missing question key."""
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422


def test_chat_validation_too_long():
    """Test POST /api/v1/chat fails on excessively long question."""
    response = client.post("/api/v1/chat", json={"question": "x" * 1001})
    assert response.status_code == 422


def test_cache_stats_endpoint():
    """Test GET /api/v1/cache/stats returns valid metrics."""
    response = client.get("/api/v1/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "exact_cache_hits" in data
    assert "semantic_cache_hits" in data
    assert "cache_misses" in data
    assert "knowledge_base_version" in data


def test_cache_clear_endpoint():
    """Test POST /api/v1/cache/clear purges cache."""
    response = client.post("/api/v1/cache/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "deleted_count" in data
