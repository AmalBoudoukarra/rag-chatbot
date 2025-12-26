"""
Embedding model management for document vectorization.
"""
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
from app.utils.logger import get_logger
from app.utils.exceptions import EmbeddingException

logger = get_logger(__name__)


class EmbeddingManager:
    """Manages embedding model initialization and usage."""
    
    def __init__(self):
        """Initialize embedding model."""
        self._embeddings = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the embedding model."""
        try:
            logger.info(
                "Initializing embedding model",
                model=settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            )
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': settings.EMBEDDING_DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info("Embedding model initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize embedding model", error=str(e))
            raise EmbeddingException(f"Failed to initialize embeddings: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self._embeddings.embed_query(text)
        except Exception as e:
            logger.error("Failed to embed query", error=str(e))
            raise EmbeddingException(f"Failed to embed query: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embedding vectors
        """
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as e:
            logger.error("Failed to embed documents", error=str(e))
            raise EmbeddingException(f"Failed to embed documents: {e}")
    
    def get_embeddings(self):
        """Get the underlying embeddings object for LangChain."""
        return self._embeddings


# Singleton instance
_embedding_manager = None


def get_embedding_manager() -> EmbeddingManager:
    """Get or create the singleton embedding manager."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager