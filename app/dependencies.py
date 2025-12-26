"""
dependencies.py
Fournit toutes les dépendances injectables pour FastAPI.
"""

from fastapi import Depends
from app.services.qa import get_qa_service, QAService
from app.services.ingestion import get_ingestion_service, IngestionService
from app.core.embeddings import get_embedding_manager, EmbeddingManager
from app.core.vector_store import get_vectorstore_manager, VectorStoreManager
from app.core.llm import get_llm_manager, LLMManager

# -----------------------------
# QA Service
# -----------------------------
def qa_service() -> QAService:
    """
    Dépendance FastAPI pour QAService (RAG).
    """
    return get_qa_service()


# -----------------------------
# Ingestion Service
# -----------------------------
def ingestion_service() -> IngestionService:
    """
    Dépendance FastAPI pour IngestionService.
    """
    return get_ingestion_service()


# -----------------------------
# Embedding Manager
# -----------------------------
def embedding_manager() -> EmbeddingManager:
    """
    Dépendance FastAPI pour EmbeddingManager.
    """
    return get_embedding_manager()


# -----------------------------
# Vector Store Manager
# -----------------------------
def vectorstore_manager() -> VectorStoreManager:
    """
    Dépendance FastAPI pour VectorStoreManager.
    """
    return get_vectorstore_manager()


# -----------------------------
# LLM Manager
# -----------------------------
def llm_manager() -> LLMManager:
    """
    Dépendance FastAPI pour LLMManager.
    """
    return get_llm_manager()


# -----------------------------
# Exemple d'utilisation dans FastAPI
# -----------------------------
# from fastapi import APIRouter, Depends
# from app.dependencies import qa_service
#
# router = APIRouter()
#
# @router.get("/ask")
# async def ask_question(
#     question: str, 
#     service: QAService = Depends(qa_service)
# ):
#     return service.answer_question(question)
