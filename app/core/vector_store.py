"""
Vector store management using FAISS.
"""
from typing import List, Optional
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from app.config import settings
from app.core.embeddings import get_embedding_manager
from app.utils.logger import get_logger
from app.utils.exceptions import VectorStoreNotInitializedException

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages vector store operations."""
    
    def __init__(self):
        """Initialize vector store manager."""
        self._vectorstore: Optional[FAISS] = None
        self._is_initialized = False
    
    def initialize(self) -> None:
        """
        Initialize or load existing vector store.
        """
        try:
            embedding_manager = get_embedding_manager()
            embeddings = embedding_manager.get_embeddings()
            
            # Ensure directory exists
            settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            
            # Check if vector store already exists
            if self._vectorstore_exists():
                logger.info(
                    "Loading existing vector store",
                    path=str(settings.VECTOR_DB_DIR)
                )
                self._vectorstore = FAISS.load_local(
                    str(settings.VECTOR_DB_DIR),
                    embeddings
                )
            else:
                logger.warning(
                    "No existing vector store found",
                    path=str(settings.VECTOR_DB_DIR)
                )
                # Don't create empty vector store, wait for documents
                self._vectorstore = None
            
            self._is_initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize vector store", error=str(e))
            raise VectorStoreNotInitializedException(
                f"Failed to initialize vector store: {e}"
            )
    
    def _vectorstore_exists(self) -> bool:
        """Check if vector store directory contains data."""
        if not settings.VECTOR_DB_DIR.exists():
            return False
        
        # Check for FAISS index files
        index_file = settings.VECTOR_DB_DIR / "index.faiss"
        pkl_file = settings.VECTOR_DB_DIR / "index.pkl"
        return index_file.exists() and pkl_file.exists()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
        """
        if not self._is_initialized:
            raise VectorStoreNotInitializedException(
                "Vector store not initialized. Call initialize() first."
            )
        
        try:
            logger.info("Adding documents to vector store", count=len(documents))
            
            embedding_manager = get_embedding_manager()
            embeddings = embedding_manager.get_embeddings()
            
            if self._vectorstore is None:
                # Create new vector store from documents
                self._vectorstore = FAISS.from_documents(documents, embeddings)
                logger.info("Created new vector store with documents")
            else:
                # Add to existing vector store
                new_vectorstore = FAISS.from_documents(documents, embeddings)
                self._vectorstore.merge_from(new_vectorstore)
                logger.info("Merged documents into existing vector store")
            
            # Save after adding documents
            self._save_vectorstore()
            logger.info("Documents added successfully")
            
        except Exception as e:
            logger.error("Failed to add documents", error=str(e))
            raise
    
    def _save_vectorstore(self) -> None:
        """Save the vector store to disk."""
        if self._vectorstore is not None:
            try:
                settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
                self._vectorstore.save_local(str(settings.VECTOR_DB_DIR))
                logger.info("Vector store saved", path=str(settings.VECTOR_DB_DIR))
            except Exception as e:
                logger.error("Failed to save vector store", error=str(e))
                raise
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return (default from settings)
            
        Returns:
            List of similar documents
        """
        self._ensure_initialized()
        
        k = k or settings.SEARCH_K
        
        try:
            logger.info("Performing similarity search", query=query, k=k)
            results = self._vectorstore.similarity_search(query, k=k)
            logger.info("Search completed", results_count=len(results))
            return results
            
        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            raise
    
    def get_retriever(self, k: Optional[int] = None):
        """
        Get a retriever for the vector store.
        
        Args:
            k: Number of results to return
            
        Returns:
            LangChain retriever
        """
        self._ensure_initialized()
        k = k or settings.SEARCH_K
        return self._vectorstore.as_retriever(search_kwargs={"k": k})
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._ensure_initialized()
        
        try:
            logger.warning("Deleting vector store", path=str(settings.VECTOR_DB_DIR))
            
            # Delete FAISS index files
            index_file = settings.VECTOR_DB_DIR / "index.faiss"
            pkl_file = settings.VECTOR_DB_DIR / "index.pkl"
            
            if index_file.exists():
                index_file.unlink()
            if pkl_file.exists():
                pkl_file.unlink()
            
            self._vectorstore = None
            self._is_initialized = False
            logger.info("Vector store deleted successfully")
            
        except Exception as e:
            logger.error("Failed to delete vector store", error=str(e))
            raise
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        self._ensure_initialized()
        
        try:
            if self._vectorstore is None:
                return 0
            # FAISS doesn't have a direct count method, so we count the index
            return self._vectorstore.index.ntotal
        except Exception as e:
            logger.error("Failed to get document count", error=str(e))
            return 0
    
    def _ensure_initialized(self) -> None:
        """Ensure vector store is initialized."""
        if not self._is_initialized:
            raise VectorStoreNotInitializedException(
                "Vector store not initialized. Call initialize() first or run ingestion."
            )
        
        if self._vectorstore is None:
            raise VectorStoreNotInitializedException(
                "Vector store is empty. Please ingest documents first."
            )
    
    @property
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self._is_initialized and self._vectorstore is not None


# Singleton instance
_vectorstore_manager = None


def get_vectorstore_manager() -> VectorStoreManager:
    """Get or create the singleton vector store manager."""
    global _vectorstore_manager
    if _vectorstore_manager is None:
        _vectorstore_manager = VectorStoreManager()
    return _vectorstore_manager