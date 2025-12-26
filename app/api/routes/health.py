"""
Health check and system status routes.
"""
from fastapi import APIRouter, Depends
from app.api.models.responses import HealthResponse, StatsResponse
from app.services.qa import get_qa_service, QAService
from app.core.vector_store import get_vectorstore_manager
from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns the status of the service and vector store.
    """
    vectorstore_manager = get_vectorstore_manager()
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        vector_store_initialized=vectorstore_manager.is_initialized
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(qa_service: QAService = Depends(get_qa_service)):
    """
    Get system statistics.
    Returns information about the vector store and configuration.
    """
    stats = qa_service.get_statistics()
    return StatsResponse(**stats)