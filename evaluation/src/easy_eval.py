#!/usr/bin/env python3
"""
Easy-to-use evaluation functions for NER and Semi-Structured entity evaluation.
"""

from typing import List, Dict, Any, Literal
from .ner_metrics import NERMetrics, SemiStructuredMetrics


def evaluate_entities(
    y_true: List[Dict[str, Any]], 
    y_pred: List[Dict[str, Any]], 
    mode: Literal["ner", "semi_structured"] = "ner",
    strict_match: bool = True,
    matching_method: str = "position",
    text_match_mode: str = "exact",
    similarity_threshold: float = 0.8,
    similarity_model: str = "tiny",
    enable_validation: bool = False
) -> Dict[str, float]:
    """
    Easy-to-use function for entity evaluation.
    
    Args:
        y_true: List of ground truth entity dictionaries
        y_pred: List of predicted entity dictionaries
        mode: "ner" for Named Entity Recognition, "semi_structured" for entity extraction with position
        strict_match: If True, entities must match exactly; If False, allow overlap matching (for NER mode only)
        matching_method: For semi_structured mode, "position" (compare start/end) or "text" (compare content)
        text_match_mode: For semi_structured mode with text matching, "exact", "overlap", or "similarity"
        similarity_threshold: Similarity threshold for similarity matching (0-1, default 0.8)
        similarity_model: BERT model to use for similarity ("tiny", "small", "medium", "large", or custom model name)
        enable_validation: If True, validate entity consistency (position and type/content)
    
    Returns:
        Dictionary containing evaluation metrics:
        - precision: Precision score
        - recall: Recall score  
        - f1_score: F1 score
        - true_positives: Number of true positives
        - false_positives: Number of false positives
        - false_negatives: Number of false negatives
        - accuracy: Accuracy (always 0.0 for NER, not meaningful)
        - true_negatives: True negatives (always 0 for NER, not meaningful)
    
    Example:
        # NER evaluation
        results = evaluate_entities(y_true, y_pred, mode="ner")
        
        # Semi-structured evaluation - position matching
        results = evaluate_entities(y_true, y_pred, mode="semi_structured", matching_method="position")
        
        # Semi-structured evaluation - text exact matching
        results = evaluate_entities(y_true, y_pred, mode="semi_structured", matching_method="text", text_match_mode="exact")
        
        # Semi-structured evaluation - text overlap matching
        results = evaluate_entities(y_true, y_pred, mode="semi_structured", matching_method="text", text_match_mode="overlap")
        
        # Semi-structured evaluation - text similarity matching
        results = evaluate_entities(y_true, y_pred, mode="semi_structured", matching_method="text", text_match_mode="similarity", similarity_threshold=0.8, similarity_model="small")
        

    """
    
    if mode == "ner":
        metrics = NERMetrics(strict_match=strict_match, enable_validation=enable_validation)
    elif mode == "semi_structured":
        metrics = SemiStructuredMetrics(matching_method=matching_method, text_match_mode=text_match_mode, 
                                      similarity_threshold=similarity_threshold, similarity_model=similarity_model, enable_validation=enable_validation)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'ner' or 'semi_structured'")
    
    return metrics.calculate_metrics(y_true, y_pred)


def evaluate_entities_by_type(
    y_true: List[Dict[str, Any]], 
    y_pred: List[Dict[str, Any]], 
    strict_match: bool = True,
    enable_validation: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate NER entities by type (only works for NER mode).
    
    Args:
        y_true: List of ground truth entity dictionaries
        y_pred: List of predicted entity dictionaries
        strict_match: If True, entities must match exactly; If False, allow overlap matching
        enable_validation: If True, validate entity consistency
    
    Returns:
        Dictionary mapping entity types to their metrics
    
    Example:
        type_results = evaluate_entities_by_type(y_true, y_pred)
        for entity_type, metrics in type_results.items():
            print(f"{entity_type}: F1={metrics['f1_score']:.3f}")
    """
    
    metrics = NERMetrics(strict_match=strict_match, enable_validation=enable_validation)
    return metrics.calculate_metrics_by_type(y_true, y_pred)


def quick_eval(
    y_true: List[Dict[str, Any]], 
    y_pred: List[Dict[str, Any]], 
    mode: Literal["ner", "semi_structured"] = "ner"
) -> Dict[str, float]:
    """
    Quick evaluation with default settings (strict matching, no validation).
    
    Args:
        y_true: List of ground truth entity dictionaries
        y_pred: List of predicted entity dictionaries
        mode: "ner" or "semi_structured"
    
    Returns:
        Dictionary with precision, recall, f1_score, and counts
    
    Example:
        results = quick_eval(y_true, y_pred, mode="ner")
        print(f"F1 Score: {results['f1_score']:.3f}")
    """
    
    return evaluate_entities(y_true, y_pred, mode=mode, strict_match=True, enable_validation=False)


# Convenience functions for specific use cases
def ner_eval(y_true: List[Dict[str, Any]], y_pred: List[Dict[str, Any]], **kwargs) -> Dict[str, float]:
    """NER evaluation with default settings."""
    return evaluate_entities(y_true, y_pred, mode="ner", **kwargs)


def semi_structured_eval(y_true: List[Dict[str, Any]], y_pred: List[Dict[str, Any]], **kwargs) -> Dict[str, float]:
    """Semi-structured evaluation with default settings."""
    return evaluate_entities(y_true, y_pred, mode="semi_structured", **kwargs)


 