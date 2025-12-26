"""
Document ingestion service.
Handles document loading, chunking, and storage in vector database.
"""
import os
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document
from app.config import settings
from app.core.vector_store import get_vectorstore_manager
from app.utils.logger import get_logger
from app.utils.exceptions import DocumentIngestionException, InvalidDocumentException

logger = get_logger(__name__)


class IngestionService:
    """Service for ingesting documents into the vector store."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}
    
    def __init__(self):
        """Initialize ingestion service."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def ingest_all_documents(self) -> dict:
        """
        Ingest all documents from the data directory.
        
        Returns:
            Statistics about the ingestion process
        """
        try:
            logger.info(
                "Starting document ingestion",
                data_dir=str(settings.DATA_DIR)
            )
            
            # Ensure data directory exists
            if not settings.DATA_DIR.exists():
                raise DocumentIngestionException(
                    f"Data directory not found: {settings.DATA_DIR}"
                )
            
            # Load all documents
            documents = self._load_all_documents()
            
            if not documents:
                logger.warning("No documents found to ingest")
                return {
                    "status": "warning",
                    "message": "No documents found",
                    "documents_loaded": 0,
                    "chunks_created": 0
                }
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Split into chunks
            chunks = self._split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Store in vector database
            vectorstore_manager = get_vectorstore_manager()
            
            # Initialize vector store if needed
            if not vectorstore_manager.is_initialized:
                vectorstore_manager.initialize()
            
            # Add documents
            vectorstore_manager.add_documents(chunks)
            
            # Get final count
            total_docs = vectorstore_manager.get_document_count()
            
            logger.info(
                "Ingestion completed successfully",
                total_documents=total_docs
            )
            
            return {
                "status": "success",
                "message": "Documents ingested successfully",
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "total_in_db": total_docs
            }
            
        except Exception as e:
            logger.error("Document ingestion failed", error=str(e))
            raise DocumentIngestionException(f"Ingestion failed: {e}")
    
    def _load_all_documents(self) -> List[Document]:
        """
        Load all supported documents from data directory.
        
        Returns:
            List of loaded documents
        """
        documents = []
        
        for filename in os.listdir(settings.DATA_DIR):
            filepath = settings.DATA_DIR / filename
            
            # Skip directories
            if filepath.is_dir():
                continue
            
            # Check extension
            ext = filepath.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                logger.warning(
                    "Skipping unsupported file",
                    filename=filename,
                    extension=ext
                )
                continue
            
            # Load document
            try:
                docs = self._load_single_document(filepath)
                documents.extend(docs)
                logger.info(f"Loaded {filename}", pages=len(docs))
            except Exception as e:
                logger.error(
                    "Failed to load document",
                    filename=filename,
                    error=str(e)
                )
        
        return documents
    
    def _load_single_document(self, filepath: Path) -> List[Document]:
        """
        Load a single document based on its extension.
        
        Args:
            filepath: Path to the document
            
        Returns:
            List of Document objects (may be multiple pages)
        """
        ext = filepath.suffix.lower()
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(str(filepath))
            elif ext == '.txt':
                loader = TextLoader(str(filepath), encoding='utf-8')
            elif ext == '.md':
                loader = UnstructuredMarkdownLoader(str(filepath))
            else:
                raise InvalidDocumentException(
                    f"Unsupported file extension: {ext}"
                )
            
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source'] = filepath.name
                doc.metadata['file_type'] = ext
            
            return documents
            
        except Exception as e:
            raise InvalidDocumentException(
                f"Failed to load {filepath.name}: {e}"
            )
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
            
            return chunks
            
        except Exception as e:
            logger.error("Failed to split documents", error=str(e))
            raise DocumentIngestionException(f"Failed to split documents: {e}")
    
    def ingest_single_file(self, filepath: Path) -> dict:
        """
        Ingest a single document file.
        
        Args:
            filepath: Path to the document
            
        Returns:
            Ingestion statistics
        """
        try:
            logger.info("Ingesting single file", filepath=str(filepath))
            
            # Load document
            documents = self._load_single_document(filepath)
            
            # Split into chunks
            chunks = self._split_documents(documents)
            
            # Store in vector database
            vectorstore_manager = get_vectorstore_manager()
            
            if not vectorstore_manager.is_initialized:
                vectorstore_manager.initialize()
            
            vectorstore_manager.add_documents(chunks)
            
            return {
                "status": "success",
                "filename": filepath.name,
                "chunks_created": len(chunks)
            }
            
        except Exception as e:
            logger.error(
                "Failed to ingest file",
                filepath=str(filepath),
                error=str(e)
            )
            raise DocumentIngestionException(f"Failed to ingest file: {e}")


# Singleton instance
_ingestion_service = None


def get_ingestion_service() -> IngestionService:
    """Get or create the singleton ingestion service."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service