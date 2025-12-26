"""
Question-Answering API routes.
"""
from fastapi import APIRouter, Depends, HTTPException
from app.api.models.requests import QuestionRequest
from app.api.models.responses import AnswerResponse
from app.services.qa import get_qa_service, QAService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ask", tags=["Q&A"])


@router.post("", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    qa_service: QAService = Depends(get_qa_service)
) -> AnswerResponse:
    """
    Answer a question using RAG.
    
    - **question**: The question to answer
    - **use_llm**: Whether to use LLM for generation (default: True)
    
    Returns the answer with sources.
    """
    try:
        logger.info(
            "Received question",
            question=request.question[:100],
            use_llm=request.use_llm
        )
        
        result = qa_service.answer_question(
            question=request.question,
            use_llm=request.use_llm
        )
        
        return AnswerResponse(**result)
        
    except Exception as e:
        logger.error("Error processing question", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )