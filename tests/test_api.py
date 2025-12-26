"""
API endpoint tests.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check and system endpoints."""
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_ping(self):
        """Test ping endpoint."""
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_health(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "vector_store_initialized" in data
    
    def test_stats(self):
        """Test statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "vector_store_initialized" in data


class TestQAEndpoint:
    """Test question-answering endpoint."""
    
    def test_ask_valid_question(self):
        """Test asking a valid question."""
        response = client.post(
            "/ask",
            json={"question": "What is RAG?", "use_llm": False}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "context_used" in data
    
    def test_ask_empty_question(self):
        """Test asking an empty question."""
        response = client.post(
            "/ask",
            json={"question": "", "use_llm": False}
        )
        assert response.status_code == 422  # Validation error
    
    def test_ask_invalid_payload(self):
        """Test with invalid payload."""
        response = client.post(
            "/ask",
            json={"invalid": "payload"}
        )
        assert response.status_code == 422


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_page(self):
        """Test Swagger UI documentation page."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_page(self):
        """Test ReDoc documentation page."""
        response = client.get("/redoc")
        assert response.status_code == 200