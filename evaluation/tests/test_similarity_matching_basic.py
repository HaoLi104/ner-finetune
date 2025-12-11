#!/usr/bin/env python3
"""
Basic tests for similarity-based matching functionality (without BERT model).
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ner_metrics import SemiStructuredMetrics
from src.similarity_utils import ChineseMedicalSimilarity

class TestSimilarityMatchingBasic:
    """Test similarity-based matching functionality without BERT model."""
    
    def test_similarity_calculator_initialization(self):
        """Test similarity calculator initialization."""
        calculator = ChineseMedicalSimilarity(model_name="offline", load_model=False)  # Don't load model for testing
        assert calculator is not None
        assert calculator.model_name == "offline"
        assert calculator.model is None  # Model should not be loaded
    
    def test_basic_similarity_fallback(self):
        """Test basic similarity calculation when BERT model is not available."""
        # Create calculator without model
        calculator = ChineseMedicalSimilarity(load_model=False)  # Don't load model
        
        # Test exact match
        assert calculator.calculate_similarity("患者姓名", "患者姓名") == 1.0
        
        # Test partial match (using predefined similarities)
        similarity = calculator.calculate_similarity("患者姓名", "患者名")
        assert similarity == 0.80  # Predefined similarity value
        
        # Test no match
        assert calculator.calculate_similarity("患者姓名", "糖尿病") < 0.5
    
    def test_similarity_threshold_check(self):
        """Test similarity threshold checking."""
        calculator = ChineseMedicalSimilarity(load_model=False)  # Don't load model
        
        # Test with different thresholds
        assert calculator.is_similar("患者姓名", "患者姓名", threshold=0.8) == True
        assert calculator.is_similar("患者姓名", "患者名", threshold=0.8) == True  # Predefined similarity is 0.80
        assert calculator.is_similar("患者姓名", "患者名", threshold=0.1) == True   # Low threshold
    
    def test_semi_structured_similarity_metrics(self):
        """Test semi-structured metrics with similarity matching."""
        metrics = SemiStructuredMetrics(
            matching_method="text", 
            text_match_mode="similarity", 
            similarity_threshold=0.8
        )
        
        y_true = [
            {
                "entities": [
                    {"text": "患者姓名", "start": 10, "end": 15},
                    {"text": "急性阑尾炎", "start": 20, "end": 25}
                ]
            }
        ]
        
        y_pred = [
            {
                "entities": [
                    {"text": "患者姓名", "start": 10, "end": 15},  # Exact match
                    {"text": "阑尾炎", "start": 20, "end": 23}     # Similar but not exact
                ]
            }
        ]
        
        results = metrics.calculate_metrics(y_true, y_pred)
        
        # Should have at least one true positive (exact match)
        assert results["true_positives"] >= 1
        assert results["precision"] > 0
        assert results["recall"] > 0
    
    def test_similarity_threshold_validation(self):
        """Test similarity threshold validation."""
        # Valid thresholds
        metrics1 = SemiStructuredMetrics(similarity_threshold=0.5)
        metrics2 = SemiStructuredMetrics(similarity_threshold=1.0)
        
        assert metrics1.similarity_threshold == 0.5
        assert metrics2.similarity_threshold == 1.0
        
        # Invalid thresholds should raise ValueError
        with pytest.raises(ValueError):
            SemiStructuredMetrics(similarity_threshold=1.5)
        
        with pytest.raises(ValueError):
            SemiStructuredMetrics(similarity_threshold=-0.1)
    
    def test_text_match_mode_validation(self):
        """Test text match mode validation."""
        # Valid modes
        SemiStructuredMetrics(text_match_mode="exact")
        SemiStructuredMetrics(text_match_mode="overlap")
        SemiStructuredMetrics(text_match_mode="similarity")
        
        # Invalid mode should raise ValueError
        with pytest.raises(ValueError):
            SemiStructuredMetrics(text_match_mode="invalid")
    
    def test_similarity_with_empty_texts(self):
        """Test similarity calculation with empty texts."""
        calculator = ChineseMedicalSimilarity(load_model=False)  # Don't load model
        
        # Empty texts should return 0 similarity
        assert calculator.calculate_similarity("", "患者姓名") == 0.0
        assert calculator.calculate_similarity("患者姓名", "") == 0.0
        assert calculator.calculate_similarity("", "") == 0.0
    
    def test_batch_similarity_calculation(self):
        """Test batch similarity calculation."""
        calculator = ChineseMedicalSimilarity(load_model=False)  # Don't load model
        
        texts1 = ["患者姓名", "急性阑尾炎", "入院日期"]
        texts2 = ["患者姓名", "阑尾炎", "入院时间"]
        
        similarities = calculator.calculate_batch_similarity(texts1, texts2)
        
        assert len(similarities) == 3
        assert similarities[0] == 1.0  # Exact match
        assert 0.0 < similarities[1] < 1.0  # Partial match
        assert 0.0 < similarities[2] < 1.0  # Partial match
    
    def test_similarity_matching_edge_cases(self):
        """Test similarity matching with edge cases."""
        metrics = SemiStructuredMetrics(
            matching_method="text", 
            text_match_mode="similarity", 
            similarity_threshold=0.8
        )
        
        # Empty predictions
        y_true = [{"entities": [{"text": "患者姓名", "start": 10, "end": 15}]}]
        y_pred = [{"entities": []}]
        
        results = metrics.calculate_metrics(y_true, y_pred)
        assert results["f1_score"] == 0.0
        assert results["true_positives"] == 0
        assert results["false_negatives"] == 1
        
        # Perfect predictions
        results = metrics.calculate_metrics(y_true, y_true)
        assert results["f1_score"] == 1.0
        assert results["true_positives"] == 1
        assert results["false_positives"] == 0
        assert results["false_negatives"] == 0 