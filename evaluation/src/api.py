#!/usr/bin/env python3
"""
FastAPI web service for entity evaluation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
from .easy_eval import evaluate_entities, evaluate_entities_by_type

app = FastAPI(
    title="Medical NER and Semi-Structured Extraction Evaluation API",
    description="API for evaluating NER and Semi-Structured entity extraction results",
    version="1.0.0"
)


class Entity(BaseModel):
    """Entity model for API requests."""
    start: int
    end: int
    type: Optional[str] = None
    text: Optional[str] = None


class Document(BaseModel):
    """Document model containing entities."""
    entities: List[Entity]


class EvaluationRequest(BaseModel):
    """Request model for evaluation."""
    y_true: List[Document]
    y_pred: List[Document]
    mode: Literal["ner", "semi_structured"] = "ner"
    strict_match: bool = True
    enable_validation: bool = False
    # Semi-structured specific parameters
    matching_method: Optional[Literal["position", "text"]] = None
    text_match_mode: Optional[Literal["exact", "overlap", "similarity"]] = None
    similarity_threshold: Optional[float] = 0.8
    similarity_model: Optional[Literal["tiny", "small", "medium", "large", "offline"]] = "tiny"


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    true_negatives: int


class TypeEvaluationResponse(BaseModel):
    """Response model for type-specific evaluation."""
    overall: EvaluationResponse
    by_type: Dict[str, Dict[str, float]]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Entity Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "/evaluate": "Evaluate entities with full options",
            "/quick-evaluate": "Quick evaluation with defaults",
            "/evaluate-by-type": "Evaluate NER entities by type",
            "/health": "Health check"
        },
        "modes": {
            "ner": "Named Entity Recognition evaluation",
            "semi_structured": "Semi-structured entity extraction evaluation"
        },
        "matching_methods": {
            "position": "Exact position matching (start/end)",
            "text": "Text-based matching with modes: exact, overlap, similarity"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_entities_api(request: EvaluationRequest):
    """
    Evaluate entity extraction results with full options.
    
    Args:
        request: Evaluation request containing y_true, y_pred, and parameters
    
    Returns:
        Evaluation results with precision, recall, F1 score, and counts
    """
    try:
        # Convert Pydantic models to dictionaries
        y_true = [{"entities": [entity.dict() for entity in doc.entities]} for doc in request.y_true]
        y_pred = [{"entities": [entity.dict() for entity in doc.entities]} for doc in request.y_pred]
        
        # Prepare evaluation parameters
        eval_params = {
            "y_true": y_true,
            "y_pred": y_pred,
            "mode": request.mode,
            "strict_match": request.strict_match,
            "enable_validation": request.enable_validation
        }
        
        # Add semi-structured specific parameters if provided
        if request.mode == "semi_structured":
            if request.matching_method:
                eval_params["matching_method"] = request.matching_method
            if request.text_match_mode:
                eval_params["text_match_mode"] = request.text_match_mode
            if request.similarity_threshold is not None:
                eval_params["similarity_threshold"] = request.similarity_threshold
            if request.similarity_model:
                eval_params["similarity_model"] = request.similarity_model
        
        # Perform evaluation
        results = evaluate_entities(**eval_params)
        
        return EvaluationResponse(**results)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/evaluate-by-type", response_model=TypeEvaluationResponse)
async def evaluate_entities_by_type_api(request: EvaluationRequest):
    """
    Evaluate NER entities by type (only works for NER mode).
    
    Args:
        request: Evaluation request containing y_true, y_pred, and parameters
    
    Returns:
        Overall evaluation results and results by entity type
    """
    if request.mode != "ner":
        raise HTTPException(status_code=400, detail="Type-specific evaluation only works for NER mode")
    
    try:
        # Convert Pydantic models to dictionaries
        y_true = [{"entities": [entity.dict() for entity in doc.entities]} for doc in request.y_true]
        y_pred = [{"entities": [entity.dict() for entity in doc.entities]} for doc in request.y_pred]
        
        # Perform evaluation
        overall_results = evaluate_entities(
            y_true=y_true,
            y_pred=y_pred,
            mode=request.mode,
            strict_match=request.strict_match,
            enable_validation=request.enable_validation
        )
        
        type_results = evaluate_entities_by_type(
            y_true=y_true,
            y_pred=y_pred,
            strict_match=request.strict_match,
            enable_validation=request.enable_validation
        )
        
        return TypeEvaluationResponse(
            overall=EvaluationResponse(**overall_results),
            by_type=type_results
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/quick-evaluate", response_model=EvaluationResponse)
async def quick_evaluate_api(request: EvaluationRequest):
    """
    Quick evaluation with default settings.
    
    Args:
        request: Evaluation request containing y_true, y_pred, and mode
    
    Returns:
        Evaluation results with default settings
    """
    try:
        # Convert Pydantic models to dictionaries
        y_true = [{"entities": [entity.dict() for entity in doc.entities]} for doc in request.y_true]
        y_pred = [{"entities": [entity.dict() for entity in doc.entities]} for doc in request.y_pred]
        
        # Prepare evaluation parameters with defaults
        eval_params = {
            "y_true": y_true,
            "y_pred": y_pred,
            "mode": request.mode,
            "strict_match": True,
            "enable_validation": False
        }
        
        # Add default semi-structured parameters if needed
        if request.mode == "semi_structured":
            eval_params["matching_method"] = "position"
        
        # Perform quick evaluation
        results = evaluate_entities(**eval_params)
        
        return EvaluationResponse(**results)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 