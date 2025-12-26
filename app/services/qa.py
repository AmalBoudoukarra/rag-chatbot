"""
Question-Answering service using RAG.
"""
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from app.config import settings
from app.core.vector_store import get_vectorstore_manager
from app.core.llm import get_llm_manager
from app.utils.logger import get_logger
from app.utils.exceptions import VectorStoreNotInitializedException, LLMException

logger = get_logger(__name__)


class QAService:
    """Service for answering questions using RAG."""
    
    def __init__(self):
        """Initialize QA service."""
        self.vectorstore_manager = get_vectorstore_manager()
        self.llm_manager: Optional[Any] = None
        self.llm_available = False
    
    def _ensure_llm_initialized(self) -> bool:
        """
        Ensure LLM is initialized (lazy loading).
        
        Returns:
            True if LLM is available, False otherwise
        """
        # If already initialized and available, return True
        if self.llm_manager is not None and self.llm_available:
            logger.info("LLM already initialized and available")
            return True
        
        # Try to initialize
        try:
            logger.info("=== ATTEMPTING TO INITIALIZE LLM ===")
            self.llm_manager = get_llm_manager()
            self.llm_available = True
            logger.info("=== LLM INITIALIZED SUCCESSFULLY IN QA SERVICE ===")
            return True
            
        except LLMException as e:
            logger.error(
                "LLM initialization failed - will use context-only mode",
                error=str(e)
            )
            self.llm_available = False
            self.llm_manager = None
            return False
            
        except Exception as e:
            logger.error(
                "Unexpected error during LLM initialization",
                error=str(e),
                exception_type=type(e).__name__
            )
            self.llm_available = False
            self.llm_manager = None
            return False
    
    def answer_question(
        self,
        question: str,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            use_llm: Whether to use LLM for generation (if False, returns raw context)
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            logger.info("Processing question", question=question[:100], use_llm=use_llm)
            
            # Ensure vector store is initialized
            if not self.vectorstore_manager.is_initialized:
                logger.info("Initializing vector store")
                self.vectorstore_manager.initialize()
            
            # Retrieve relevant documents
            relevant_docs = self._retrieve_documents(question)
            
            if not relevant_docs:
                logger.warning("No relevant documents found")
                return {
                    "answer": "Je n'ai pas trouvé d'informations pertinentes dans les documents pour répondre à cette question.",
                    "sources": [],
                    "context_used": False,
                    "retrieved_chunks": 0
                }
            
            # Extract sources
            sources = self._extract_sources(relevant_docs)
            
            # Generate answer
            if use_llm:
                logger.info("=== USER REQUESTED LLM ANSWER ===")
                answer = self._generate_llm_answer(question, relevant_docs)
            else:
                logger.info("Generating context-only answer (LLM disabled by request)")
                answer = self._generate_context_answer(relevant_docs)
            
            logger.info(
                "Question answered successfully",
                sources_count=len(sources),
                used_llm=use_llm,
                chunks_retrieved=len(relevant_docs)
            )
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": True,
                "retrieved_chunks": len(relevant_docs)
            }
            
        except VectorStoreNotInitializedException:
            logger.error("Vector store not initialized")
            return {
                "answer": "La base de données n'est pas initialisée. Veuillez d'abord ingérer des documents.",
                "sources": [],
                "context_used": False,
                "retrieved_chunks": 0,
                "error": "Vector store not initialized"
            }
        except Exception as e:
            logger.error("Failed to answer question", error=str(e), exception_type=type(e).__name__)
            return {
                "answer": f"Une erreur s'est produite lors du traitement de votre question: {str(e)}",
                "sources": [],
                "context_used": False,
                "retrieved_chunks": 0,
                "error": str(e)
            }
    
    def _retrieve_documents(self, question: str) -> List[Document]:
        """
        Retrieve relevant documents for the question.
        
        Args:
            question: User's question
            
        Returns:
            List of relevant documents
        """
        try:
            docs = self.vectorstore_manager.similarity_search(
                query=question,
                k=settings.SEARCH_K
            )
            
            logger.info(
                "Documents retrieved",
                count=len(docs),
                question=question[:50]
            )
            
            return docs
            
        except Exception as e:
            logger.error("Document retrieval failed", error=str(e))
            raise
    
    def _generate_llm_answer(
        self,
        question: str,
        documents: List[Document]
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            question: User's question
            documents: Context documents
            
        Returns:
            Generated answer
        """
        # Ensure LLM is initialized
        logger.info("Checking if LLM can be initialized...")
        llm_ready = self._ensure_llm_initialized()
        
        logger.info(
            "LLM readiness check",
            llm_ready=llm_ready,
            llm_available=self.llm_available,
            llm_manager_exists=self.llm_manager is not None
        )
        
        # If LLM is not available, fall back to context
        if not llm_ready:
            logger.warning("=== LLM NOT AVAILABLE - FALLING BACK TO CONTEXT ===")
            return self._generate_context_answer(documents)
        
        # Try to generate with LLM
        try:
            logger.info("=== CALLING LLM MANAGER TO GENERATE RESPONSE ===")
            answer = self.llm_manager.generate_response(question, documents)
            
            logger.info("=== LLM RESPONSE GENERATED SUCCESSFULLY ===")
            logger.info("Answer preview from QA Service", preview=answer[:200])
            
            # Verify we got a real LLM response, not context fallback
            if answer.startswith("Voici les informations pertinentes"):
                logger.error("❌ QA Service received context-only answer instead of LLM response!")
                logger.error("This should NOT happen - check LLM Manager")
            
            return answer
            
        except LLMException as e:
            logger.error("=== LLM GENERATION FAILED WITH LLMException ===", error=str(e))
            logger.info("Falling back to context-only answer")
            return self._generate_context_answer(documents)
            
        except Exception as e:
            logger.error(
                "=== LLM GENERATION FAILED WITH UNEXPECTED ERROR ===",
                error=str(e),
                exception_type=type(e).__name__
            )
            logger.info("Falling back to context-only answer")
            return self._generate_context_answer(documents)
    
    def _generate_context_answer(self, documents: List[Document]) -> str:
        """
        Generate answer from context without LLM.
        
        Args:
            documents: Context documents
            
        Returns:
            Formatted context
        """
        logger.info("=== GENERATING CONTEXT-ONLY ANSWER ===", num_docs=len(documents))
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Source inconnue')
            # Extract just filename from full path
            if '/' in source:
                source = source.split('/')[-1]
            elif '\\' in source:
                source = source.split('\\')[-1]
            
            content = doc.page_content.strip()
            context_parts.append(
                f"[Extrait {i} - {source}]\n{content}"
            )
        
        answer = (
            "Voici les informations pertinentes trouvées dans les documents:\n\n"
            + "\n\n".join(context_parts)
        )
        
        return answer
    
    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """
        Extract unique sources from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of unique source filenames
        """
        sources = set()
        
        for doc in documents:
            source = doc.metadata.get('source')
            if source:
                # Extract just filename from full path
                if '/' in source:
                    source = source.split('/')[-1]
                elif '\\' in source:
                    source = source.split('\\')[-1]
                sources.add(source)
        
        return sorted(list(sources))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the QA system.
        
        Returns:
            Dictionary with statistics
        """
        try:
            if not self.vectorstore_manager.is_initialized:
                self.vectorstore_manager.initialize()
            
            doc_count = self.vectorstore_manager.get_document_count()
            
            # Try to initialize LLM to check availability
            self._ensure_llm_initialized()
            
            return {
                "vector_store_initialized": True,
                "total_chunks": doc_count,
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_provider": settings.LLM_PROVIDER if self.llm_available else "none",
                "llm_available": self.llm_available,
                "search_k": settings.SEARCH_K
            }
            
        except Exception as e:
            logger.error("Failed to get statistics", error=str(e))
            return {
                "vector_store_initialized": False,
                "llm_available": False,
                "error": str(e)
            }


# Singleton instance
_qa_service = None


def get_qa_service() -> QAService:
    """Get or create the singleton QA service."""
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service