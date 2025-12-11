#!/usr/bin/env python3
"""
Tests for the new semi-structured data format.
"""

import pytest
from src.ner_metrics import SemiStructuredMetrics, SemiStructuredEntity


class TestNewSemiStructuredFormat:
    """Test cases for the new semi-structured data format."""
    
    def test_parse_entities_new_format(self):
        """Test parsing entities with new format (text field in entities array)."""
        metrics = SemiStructuredMetrics()
        
        # 新格式：嵌套在entities数组中，使用text字段
        data = {
            "entities": [
                {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
                {"text": "张三", "start": 10, "end": 12},
                {"text": "45岁", "start": 21, "end": 24}
            ]
        }
        
        entities = metrics.parse_entities(data)
        expected = {
            SemiStructuredEntity(52, 60, "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变"),
            SemiStructuredEntity(10, 12, "张三"),
            SemiStructuredEntity(21, 24, "45岁")
        }
        
        assert entities == expected
    

    
    def test_calculate_metrics_new_format(self):
        """Test metrics calculation with new format."""
        metrics = SemiStructuredMetrics()
        
        # 使用新格式的数据
        y_true = [
            {
                "entities": [
                    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
                    {"text": "张三", "start": 10, "end": 12},
                    {"text": "45岁", "start": 21, "end": 24}
                ]
            }
        ]
        
        y_pred = [
            {
                "entities": [
                    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
                    {"text": "张三", "start": 10, "end": 12},
                    {"text": "45岁", "start": 21, "end": 24},
                    {"text": "额外内容", "start": 100, "end": 105}  # False positive
                ]
            }
        ]
        
        results = metrics.calculate_metrics(y_true, y_pred)
        
        assert results["precision"] == 3/4  # 3 correct out of 4 predicted
        assert results["recall"] == 1.0     # All true entities found
        assert results["f1_score"] == 2 * (3/4 * 1.0) / (3/4 + 1.0)
        assert results["true_positives"] == 3
        assert results["false_positives"] == 1
        assert results["false_negatives"] == 0
    

    
    def test_parse_entities_missing_fields(self):
        """Test parsing with missing fields in new format."""
        metrics = SemiStructuredMetrics()
        
        # 缺少必需字段的数据
        data = {
            "entities": [
                {"text": "张三", "start": 10},  # Missing end
                {"text": "45岁", "end": 24},    # Missing start
                {"start": 30, "end": 40},       # Missing text/content
                {"text": "正常", "start": 50, "end": 52}  # Valid
            ]
        }
        
        entities = metrics.parse_entities(data)
        expected = {
            SemiStructuredEntity(50, 52, "正常")
        }
        
        assert entities == expected
    
    def test_parse_entities_empty_data(self):
        """Test parsing with empty data."""
        metrics = SemiStructuredMetrics()
        
        # 空数据
        data = {"entities": []}
        entities = metrics.parse_entities(data)
        assert entities == set()
        
        # 无效数据
        data = {"invalid": "data"}
        entities = metrics.parse_entities(data)
        assert entities == set()
    
    def test_overlap_matching_new_format(self):
        """Test overlap matching with new format."""
        metrics = SemiStructuredMetrics(matching_method="text", text_match_mode="overlap")
        
        y_true = [
            {
                "entities": [
                    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
                    {"text": "张三", "start": 10, "end": 12}
                ]
            }
        ]
        
        y_pred = [
            {
                "entities": [
                    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
                    {"text": "张三", "start": 10, "end": 12},
                    {"text": "重叠内容", "start": 11, "end": 13}  # Overlaps with "张三"
                ]
            }
        ]
        
        results = metrics.calculate_metrics(y_true, y_pred)
        
        # 在重叠匹配模式下，重叠的实体也会被计算
        assert results["true_positives"] >= 2
        assert results["precision"] > 0
        assert results["recall"] > 0
    
    def test_validation_with_new_format(self):
        """Test validation with new format."""
        metrics = SemiStructuredMetrics(enable_validation=True)
        
        y_true = [
            {
                "entities": [
                    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
                    {"text": "张三", "start": 10, "end": 12}
                ]
            }
        ]
        
        y_pred = [
            {
                "entities": [
                    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
                    {"text": "张三", "start": 10, "end": 12}
                ]
            }
        ]
        
        # 应该正常工作
        results = metrics.calculate_metrics(y_true, y_pred)
        assert results["f1_score"] == 1.0
        
        # 无效数据（不同的start位置）
        y_pred_invalid = [
            {
                "entities": [
                    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 53, "end": 60},  # Different start
                    {"text": "张三", "start": 10, "end": 12}
                ]
            }
        ]
        
        # 应该抛出错误
        with pytest.raises(ValueError, match="Position conflict"):
            metrics.calculate_metrics(y_true, y_pred_invalid) 