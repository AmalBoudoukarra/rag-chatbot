"""
LLM (Large Language Model) management.
Supports OpenAI and HuggingFace models.
"""
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from app.config import settings
from app.utils.logger import get_logger
from app.utils.exceptions import LLMException

logger = get_logger(__name__)


# Custom prompt template for RAG - Version améliorée
RAG_PROMPT_TEMPLATE = """Tu es un assistant qui répond aux questions en utilisant uniquement le contexte fourni.

Contexte:
{context}

Question: {question}

Instructions:
- Réponds de manière claire et concise
- Utilise UNIQUEMENT les informations du contexte ci-dessus
- Si la réponse n'est pas dans le contexte, dis-le clairement
- Ne cite pas directement de longs extraits, synthétise l'information
- Réponds en français

Réponse:"""


class LLMManager:
    """Manages LLM initialization and usage."""
    
    def __init__(self):
        """Initialize LLM manager."""
        self._llm = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the LLM based on configuration."""
        try:
            logger.info("Starting LLM initialization", provider=settings.LLM_PROVIDER)
            
            if settings.LLM_PROVIDER == "openai":
                self._initialize_openai()
            elif settings.LLM_PROVIDER == "huggingface":
                self._initialize_huggingface()
            else:
                raise LLMException(
                    f"Unknown LLM provider: {settings.LLM_PROVIDER}"
                )
            
            logger.info(
                "LLM initialized successfully",
                provider=settings.LLM_PROVIDER,
                model=settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.HUGGINGFACE_MODEL
            )
            
        except Exception as e:
            logger.error("Failed to initialize LLM", error=str(e), exception_type=type(e).__name__)
            raise LLMException(f"Failed to initialize LLM: {e}")
    
    def _initialize_openai(self) -> None:
        """Initialize OpenAI LLM."""
        if not settings.OPENAI_API_KEY:
            raise LLMException(
                "OpenAI API key not configured. Set OPENAI_API_KEY in .env"
            )
        
        logger.info("Initializing OpenAI LLM", model=settings.OPENAI_MODEL, api_key_present=bool(settings.OPENAI_API_KEY))
        
        try:
            self._llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                api_key=settings.OPENAI_API_KEY,
            )
            logger.info("OpenAI ChatOpenAI instance created successfully")
        except Exception as e:
            logger.error("Failed to create ChatOpenAI instance", error=str(e), exception_type=type(e).__name__)
            raise
    
    def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace LLM."""
        if not settings.HUGGINGFACE_API_KEY:
            raise LLMException(
                "HuggingFace API key not configured. Set HUGGINGFACE_API_KEY in .env"
            )
        
        logger.info(
            "Initializing HuggingFace LLM",
            model=settings.HUGGINGFACE_MODEL
        )
        
        self._llm = HuggingFaceHub(
            repo_id=settings.HUGGINGFACE_MODEL,
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
    
    def create_qa_chain(self, retriever):
        """
        Create a RetrievalQA chain.
        
        Args:
            retriever: LangChain retriever
            
        Returns:
            RetrievalQA chain
        """
        try:
            prompt = PromptTemplate(
                template=RAG_PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            return qa_chain
            
        except Exception as e:
            logger.error("Failed to create QA chain", error=str(e))
            raise LLMException(f"Failed to create QA chain: {e}")
    
    def generate_response(
        self, 
        query: str, 
        context_documents: list
    ) -> str:
        """
        Generate a response using the LLM with context.
        
        Args:
            query: User question
            context_documents: Retrieved context documents
            
        Returns:
            Generated response from LLM
        """
        try:
            logger.info("=== STARTING LLM RESPONSE GENERATION ===")
            logger.info("Generating LLM response", question=query[:50], num_docs=len(context_documents))
            
            # Build context from documents
            context = "\n\n".join([
                f"[Extrait {i+1}]\n{doc.page_content}" 
                for i, doc in enumerate(context_documents)
            ])
            
            logger.info("Context built", context_length=len(context))
            
            # Format prompt with context and question
            formatted_prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )
            
            logger.info("Prompt formatted", prompt_length=len(formatted_prompt))
            
            # Verify LLM is initialized
            if self._llm is None:
                raise LLMException("LLM is not initialized")
            
            logger.info("LLM instance type", llm_type=type(self._llm).__name__)
            
            # Generate response using the appropriate method
            if isinstance(self._llm, ChatOpenAI):
                logger.info("Using ChatOpenAI to generate response")
                
                try:
                    # Create message for chat model
                    message = HumanMessage(content=formatted_prompt)
                    
                    logger.info("Calling ChatOpenAI with message...")
                    # Invoke with list of messages - THIS ACTUALLY GENERATES TEXT
                    response = self._llm.invoke([message])
                    
                    logger.info("ChatOpenAI response received", response_type=type(response).__name__)
                    
                    # Extract content from response
                    if hasattr(response, 'content'):
                        answer = response.content
                        logger.info("Extracted answer from response.content")
                    else:
                        answer = str(response)
                        logger.info("Converted response to string")
                    
                    logger.info("=== LLM RESPONSE GENERATION COMPLETED ===")
                    logger.info("Answer preview", answer_preview=answer[:200])
                    
                    return answer
                    
                except Exception as api_error:
                    logger.error(
                        "OpenAI API call failed",
                        error=str(api_error),
                        error_type=type(api_error).__name__
                    )
                    raise LLMException(f"OpenAI API error: {str(api_error)}")
                    
            else:
                # HuggingFace model
                logger.info("Using HuggingFace to generate response")
                
                try:
                    # HuggingFaceHub uses __call__ to generate
                    response = self._llm(formatted_prompt)
                    logger.info("HuggingFace response received")
                    
                    logger.info("=== LLM RESPONSE GENERATION COMPLETED ===")
                    logger.info("Answer preview", answer_preview=response[:200])
                    
                    return response
                    
                except Exception as api_error:
                    logger.error(
                        "HuggingFace API call failed",
                        error=str(api_error),
                        error_type=type(api_error).__name__
                    )
                    raise LLMException(f"HuggingFace API error: {str(api_error)}")
                
        except LLMException:
            # Re-raise LLMException as is
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in generate_response", 
                error=str(e),
                exception_type=type(e).__name__
            )
            raise LLMException(f"Failed to generate response: {e}")
    
    def get_llm(self):
        """Get the underlying LLM object."""
        return self._llm


# Singleton instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get or create the singleton LLM manager."""
    global _llm_manager
    if _llm_manager is None:
        logger.info("Creating new LLMManager instance")
        _llm_manager = LLMManager()
    return _llm_manager