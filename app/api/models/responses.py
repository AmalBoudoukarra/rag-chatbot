"""
API response models using Pydantic.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AnswerResponse(BaseModel):
    """Response model for question answering."""
    
    answer: str = Field(
        ...,
        description="The generated answer"
    )
    
    sources: List[str] = Field(
        default_factory=list,
        description="List of source documents used"
    )
    
    context_used: bool = Field(
        default=False,
        description="Whether context was found and used"
    )
    
    retrieved_chunks: Optional[int] = Field(
        default=None,
        description="Number of chunks retrieved"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if any"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "La Schatzinsel (L'Île au trésor) est un roman...",
                "sources": ["Schatzinsel_E.pdf"],
                "context_used": True,
                "retrieved_chunks": 4
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    vector_store_initialized: bool = Field(
        ...,
        description="Whether vector store is ready"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "vector_store_initialized": True
            }
        }


class StatsResponse(BaseModel):
    """Response model for statistics."""
    
    vector_store_initialized: bool
    total_chunks: Optional[int] = None
    embedding_model: Optional[str] = None
    llm_provider: Optional[str] = None
    search_k: Optional[int] = None
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "vector_store_initialized": True,
                "total_chunks": 245,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_provider": "openai",
                "search_k": 4
            }
        }


class IngestionResponse(BaseModel):
    """Response model for ingestion operations."""
    
    status: str
    message: str
    documents_loaded: Optional[int] = None
    chunks_created: Optional[int] = None
    total_in_db: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Documents ingested successfully",
                "documents_loaded": 5,
                "chunks_created": 245,
                "total_in_db": 245
            }
        }