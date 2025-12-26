"""
Custom exceptions for the RAG application.
"""


class RAGException(Exception):
    """Base exception for RAG application."""
    pass


class VectorStoreNotInitializedException(RAGException):
    """Raised when vector store is accessed before initialization."""
    pass


class DocumentIngestionException(RAGException):
    """Raised when document ingestion fails."""
    pass


class EmbeddingException(RAGException):
    """Raised when embedding generation fails."""
    pass


class LLMException(RAGException):
    """Raised when LLM request fails."""
    pass


class InvalidDocumentException(RAGException):
    """Raised when document format is invalid or unsupported."""
    pass