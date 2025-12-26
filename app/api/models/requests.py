"""
API request models using Pydantic.
"""
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to answer",
        examples=["What is RAG?"]
    )
    
    use_llm: bool = Field(
        default=True,
        description="Whether to use LLM for answer generation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Qu'est-ce que la Schatzinsel?",
                "use_llm": True
            }
        }