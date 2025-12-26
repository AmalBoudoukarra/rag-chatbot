"""
Service layer tests.
"""
import pytest
from pathlib import Path
from app.services.ingestion import IngestionService
from app.services.qa import QAService


class TestIngestionService:
    """Test ingestion service."""
    
    def test_service_initialization(self):
        """Test that service can be initialized."""
        service = IngestionService()
        assert service is not None
        assert service.text_splitter is not None
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        service = IngestionService()
        assert '.pdf' in service.SUPPORTED_EXTENSIONS
        assert '.txt' in service.SUPPORTED_EXTENSIONS
        assert '.md' in service.SUPPORTED_EXTENSIONS


class TestQAService:
    """Test QA service."""
    
    def test_service_initialization(self):
        """Test that service can be initialized."""
        service = QAService()
        assert service is not None
        assert service.vectorstore_manager is not None
    
    def test_extract_sources(self):
        """Test source extraction from documents."""
        from langchain.schema import Document
        
        service = QAService()
        docs = [
            Document(page_content="test1", metadata={"source": "doc1.pdf"}),
            Document(page_content="test2", metadata={"source": "doc2.pdf"}),
            Document(page_content="test3", metadata={"source": "doc1.pdf"}),
        ]
        
        sources = service._extract_sources(docs)
        assert len(sources) == 2
        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources


class TestEmbeddingConfiguration:
    """Test embedding configuration."""
    
    def test_embedding_model_config(self):
        """Test that embedding model is configured."""
        from app.config import settings
        assert settings.EMBEDDING_MODEL is not None
        assert len(settings.EMBEDDING_MODEL) > 0
    
    def test_chunk_config(self):
        """Test chunking configuration."""
        from app.config import settings
        assert settings.CHUNK_SIZE > 0
        assert settings.CHUNK_OVERLAP >= 0
        assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE