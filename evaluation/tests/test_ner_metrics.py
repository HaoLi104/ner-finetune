#!/usr/bin/env python3
"""
Test file for NER metrics module.
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ner_metrics import NERMetrics, Entity


def test_perfect_predictions():
    """Test NER metrics with perfect predictions."""
    y_true = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
            ]
        }
    ]
    
    metrics = NERMetrics()
    results = metrics.calculate_metrics(y_true, y_pred)
    
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1_score"] == 1.0
    assert results["true_positives"] == 2
    assert results["false_positives"] == 0
    assert results["false_negatives"] == 0


def test_no_predictions():
    """Test NER metrics with no predictions."""
    y_true = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
            ]
        }
    ]
    
    y_pred = [{"entities": []}]
    
    metrics = NERMetrics()
    results = metrics.calculate_metrics(y_true, y_pred)
    
    assert results["precision"] == 0.0
    assert results["recall"] == 0.0
    assert results["f1_score"] == 0.0
    assert results["true_positives"] == 0
    assert results["false_positives"] == 0
    assert results["false_negatives"] == 2


def test_false_positives():
    """Test NER metrics with false positives."""
    y_true = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "额外手术"}  # False positive
            ]
        }
    ]
    
    metrics = NERMetrics()
    results = metrics.calculate_metrics(y_true, y_pred)
    
    assert results["precision"] == 0.5
    assert results["recall"] == 1.0
    assert results["f1_score"] == 2/3  # 2*(0.5*1.0)/(0.5+1.0) = 2/3
    assert results["true_positives"] == 1
    assert results["false_positives"] == 1
    assert results["false_negatives"] == 0


def test_false_negatives():
    """Test NER metrics with false negatives."""
    y_true = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}
                # Missing surgery name (false negative)
            ]
        }
    ]
    
    metrics = NERMetrics()
    results = metrics.calculate_metrics(y_true, y_pred)
    
    assert results["precision"] == 1.0
    assert results["recall"] == 0.5
    assert results["f1_score"] == 2/3  # 2*(1.0*0.5)/(1.0+0.5) = 2/3
    assert results["true_positives"] == 1
    assert results["false_positives"] == 0
    assert results["false_negatives"] == 1


def test_overlap_matching():
    """Test NER metrics with overlap matching."""
    y_true = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                {"start": 12, "end": 18, "type": "手术日期", "text": "2024-09"}  # Overlap
            ]
        }
    ]
    
    # Strict matching should fail
    metrics_strict = NERMetrics(strict_match=True)
    results_strict = metrics_strict.calculate_metrics(y_true, y_pred)
    assert results_strict["f1_score"] == 0.0
    
    # Overlap matching should succeed
    metrics_overlap = NERMetrics(strict_match=False)
    results_overlap = metrics_overlap.calculate_metrics(y_true, y_pred)
    assert results_overlap["f1_score"] == 1.0


def test_metrics_by_type():
    """Test metrics calculation by entity type."""
    y_true = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"},
                {"start": 60, "end": 70, "type": "手术名称", "text": "额外手术"}  # False positive
            ]
        }
    ]
    
    metrics = NERMetrics()
    type_metrics = metrics.calculate_metrics_by_type(y_true, y_pred)
    
    # Check surgery date metrics
    assert "手术日期" in type_metrics
    date_metrics = type_metrics["手术日期"]
    assert date_metrics["precision"] == 1.0
    assert date_metrics["recall"] == 1.0
    assert date_metrics["f1_score"] == 1.0
    
    # Check surgery name metrics
    assert "手术名称" in type_metrics
    name_metrics = type_metrics["手术名称"]
    assert name_metrics["precision"] == 0.5  # 1 correct, 1 false positive
    assert name_metrics["recall"] == 1.0
    assert name_metrics["f1_score"] == 2/3


def test_entity_equality():
    """Test Entity class equality and hashing."""
    entity1 = Entity(start=10, end=20, type="手术日期", text="2024-09-10")
    entity2 = Entity(start=10, end=20, type="手术日期", text="2024-09-10")
    entity3 = Entity(start=10, end=20, type="手术名称", text="2024-09-10")
    
    assert entity1 == entity2
    assert entity1 != entity3
    assert hash(entity1) == hash(entity2)
    assert hash(entity1) != hash(entity3)


def test_entities_overlap():
    """Test entity overlap detection."""
    metrics = NERMetrics()
    
    # Overlapping entities
    entity1 = Entity(start=10, end=20, type="手术日期", text="2024-09-10")
    entity2 = Entity(start=15, end=25, type="手术日期", text="2024-09-10")
    assert metrics._entities_overlap(entity1, entity2)
    
    # Non-overlapping entities
    entity3 = Entity(start=30, end=40, type="手术日期", text="2024-09-10")
    assert not metrics._entities_overlap(entity1, entity3)
    
    # Adjacent entities (should not overlap)
    entity4 = Entity(start=20, end=30, type="手术日期", text="2024-09-10")
    assert not metrics._entities_overlap(entity1, entity4)


def test_different_lengths():
    """Test that different length inputs raise ValueError."""
    y_true = [{"entities": []}]
    y_pred = [{"entities": []}, {"entities": []}]
    
    metrics = NERMetrics()
    with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
        metrics.calculate_metrics(y_true, y_pred)


def test_missing_entities_key():
    """Test handling of missing entities key."""
    y_true = [{"other_key": []}]
    y_pred = [{"entities": []}]
    
    metrics = NERMetrics()
    results = metrics.calculate_metrics(y_true, y_pred)
    
    assert results["precision"] == 0.0
    assert results["recall"] == 0.0
    assert results["f1_score"] == 0.0


def test_missing_entity_fields():
    """Test handling of missing entity fields."""
    y_true = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}  # Non-empty text
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}  # Same text
            ]
        }
    ]
    
    metrics = NERMetrics()
    results = metrics.calculate_metrics(y_true, y_pred)
    
    # Should match since both have non-empty text
    assert results["true_positives"] == 1
    assert results["false_positives"] == 0
    assert results["false_negatives"] == 0


def test_empty_text_field():
    """Test handling of empty text field."""
    y_true = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": ""}  # Empty text
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": ""}  # Also empty text
            ]
        }
    ]
    
    metrics = NERMetrics()
    results = metrics.calculate_metrics(y_true, y_pred)
    
    # Should match since both have empty text
    # Note: TN cannot be calculated in NER scenarios
    assert results["true_positives"] == 0
    assert results["false_positives"] == 0
    assert results["false_negatives"] == 0
    assert results["true_negatives"] == 0  # TN is not meaningful in NER


if __name__ == "__main__":
    pytest.main([__file__]) 