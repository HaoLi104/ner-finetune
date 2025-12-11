#!/usr/bin/env python3
"""
Comprehensive test for breast cancer admission report evaluation.
This test covers all evaluation modes with realistic medical data.
"""

import pytest
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.easy_eval import evaluate_entities, evaluate_entities_by_type, quick_eval
from src.ner_metrics import NERMetrics, SemiStructuredMetrics


class TestBreastCancerComprehensive:
    """Comprehensive test for breast cancer admission report evaluation."""
    
    def setup_method(self):
        """Set up test data by loading from JSON files."""
        
        # 获取数据文件路径
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        # 加载真实数据
        with open(os.path.join(data_dir, "breast_cancer_ground_truth.json"), "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
            self.y_true_ner = ground_truth["ner_data"]
            self.y_true_semi = ground_truth["semi_structured_data"]
        
        # 加载预测数据
        with open(os.path.join(data_dir, "breast_cancer_predictions.json"), "r", encoding="utf-8") as f:
            predictions = json.load(f)
            self.y_pred_ner = predictions["ner_data"]
            self.y_pred_semi = predictions["semi_structured_data"]
        
        # 加载边缘情况数据
        with open(os.path.join(data_dir, "breast_cancer_edge_cases.json"), "r", encoding="utf-8") as f:
            edge_cases = json.load(f)
            self.empty_data = edge_cases["empty_data"]
            self.conflict_data = edge_cases["conflict_data"]
            self.perfect_match = edge_cases["perfect_match"]
    
    def test_ner_strict_matching(self):
        """Test NER evaluation with strict matching."""
        print("\n=== NER Strict Matching Test ===")
        
        results = evaluate_entities(
            self.y_true_ner, self.y_pred_ner,
            mode="ner",
            strict_match=True,
            enable_validation=False
        )
        
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1 Score: {results['f1_score']:.3f}")
        print(f"True Positives: {results['true_positives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"False Negatives: {results['false_negatives']}")
        
        # 验证结果
        assert results['precision'] > 0
        assert results['recall'] > 0
        assert results['f1_score'] > 0
        assert results['true_positives'] == 4  # 完全正确的4个
        assert results['false_positives'] == 10  # 错误的10个
        assert results['false_negatives'] == 8  # 漏掉的8个
    
    def test_ner_overlap_matching(self):
        """Test NER evaluation with overlap matching."""
        print("\n=== NER Overlap Matching Test ===")
        
        results = evaluate_entities(
            self.y_true_ner, self.y_pred_ner,
            mode="ner",
            strict_match=False,
            enable_validation=False
        )
        
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1 Score: {results['f1_score']:.3f}")
        
        # 重叠匹配应该比严格匹配更宽松
        assert results['f1_score'] >= 0
    
    def test_ner_by_type(self):
        """Test NER evaluation by entity type."""
        print("\n=== NER By Type Test ===")
        
        type_results = evaluate_entities_by_type(
            self.y_true_ner, self.y_pred_ner,
            strict_match=True,
            enable_validation=False
        )
        
        for entity_type, metrics in type_results.items():
            print(f"{entity_type}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1_score']:.3f}")
        
        # 验证按类型分组的结果
        assert len(type_results) > 0
        for entity_type, metrics in type_results.items():
            assert metrics['precision'] >= 0
            assert metrics['recall'] >= 0
            assert metrics['f1_score'] >= 0
    
    def test_semi_structured_position_matching(self):
        """Test semi-structured evaluation with position matching."""
        print("\n=== Semi-Structured Position Matching Test ===")
        
        results = evaluate_entities(
            self.y_true_semi, self.y_pred_semi,
            mode="semi_structured",
            matching_method="position",
            enable_validation=False
        )
        
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1 Score: {results['f1_score']:.3f}")
        print(f"True Positives: {results['true_positives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"False Negatives: {results['false_negatives']}")
        
        # 位置匹配只考虑位置，不考虑文本
        # 真实数据：18个实体，预测数据：20个实体
        # 位置完全正确的：14个
        assert results['true_positives'] == 14  # 位置匹配的14个
        assert results['false_positives'] == 6  # 位置错误的6个
        assert results['false_negatives'] == 4  # 漏掉的4个
    
    def test_semi_structured_text_exact_matching(self):
        """Test semi-structured evaluation with text exact matching."""
        print("\n=== Semi-Structured Text Exact Matching Test ===")
        
        results = evaluate_entities(
            self.y_true_semi, self.y_pred_semi,
            mode="semi_structured",
            matching_method="text",
            text_match_mode="exact",
            enable_validation=False
        )
        
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1 Score: {results['f1_score']:.3f}")
        
        # 文本精确匹配只考虑文本，不考虑位置
        # 文本完全正确的：8个
        assert results['true_positives'] == 8  # 文本完全正确的8个
        assert results['false_positives'] == 12  # 文本错误的12个
        assert results['false_negatives'] == 10  # 漏掉的10个
    
    def test_semi_structured_text_overlap_matching(self):
        """Test semi-structured evaluation with text overlap matching."""
        print("\n=== Semi-Structured Text Overlap Matching Test ===")
        
        results = evaluate_entities(
            self.y_true_semi, self.y_pred_semi,
            mode="semi_structured",
            matching_method="text",
            text_match_mode="overlap",
            enable_validation=False
        )
        
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1 Score: {results['f1_score']:.3f}")
        
        # 重叠匹配应该比精确匹配更宽松
        assert results['f1_score'] >= 0
    
    def test_semi_structured_text_similarity_matching(self):
        """Test semi-structured evaluation with text similarity matching."""
        print("\n=== Semi-Structured Text Similarity Matching Test ===")
        
        results = evaluate_entities(
            self.y_true_semi, self.y_pred_semi,
            mode="semi_structured",
            matching_method="text",
            text_match_mode="similarity",
            similarity_threshold=0.8,
            similarity_model="tiny",
            enable_validation=False
        )
        
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1 Score: {results['f1_score']:.3f}")
        print(f"Similarity Threshold: 0.8")
        
        # 相似度匹配应该能识别语义相似的文本
        assert results['f1_score'] >= 0
    
    def test_quick_eval_functions(self):
        """Test quick evaluation functions."""
        print("\n=== Quick Evaluation Functions Test ===")
        
        # 快速NER评估
        ner_results = quick_eval(self.y_true_ner, self.y_pred_ner, mode="ner")
        print(f"Quick NER F1: {ner_results['f1_score']:.3f}")
        
        # 快速半结构化评估
        semi_results = quick_eval(self.y_true_semi, self.y_pred_semi, mode="semi_structured")
        print(f"Quick Semi-Structured F1: {semi_results['f1_score']:.3f}")
        
        assert ner_results['f1_score'] >= 0
        assert semi_results['f1_score'] >= 0
    
    def test_validation_enabled(self):
        """Test evaluation with validation enabled."""
        print("\n=== Validation Enabled Test ===")
        
        # 测试验证功能
        try:
            results = evaluate_entities(
                self.conflict_data["y_true"], self.conflict_data["y_pred"],
                mode="ner",
                enable_validation=True
            )
            print("Validation should have failed but passed")
            assert False
        except ValueError as e:
            print(f"Validation correctly failed: {e}")
            assert "Position conflict" in str(e)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n=== Edge Cases Test ===")
        
        # 空数据测试
        results = evaluate_entities(self.empty_data, self.empty_data, mode="ner")
        print(f"Empty data F1: {results['f1_score']:.3f}")
        assert results['f1_score'] == 0.0
        
        # 完美匹配测试
        perfect_results = evaluate_entities(self.perfect_match, self.perfect_match, mode="ner")
        print(f"Perfect match F1: {perfect_results['f1_score']:.3f}")
        assert perfect_results['f1_score'] == 1.0
    
    def test_comprehensive_comparison(self):
        """Comprehensive comparison of all evaluation methods."""
        print("\n=== Comprehensive Comparison ===")
        
        methods = [
            ("NER Strict", lambda: evaluate_entities(self.y_true_ner, self.y_pred_ner, mode="ner", strict_match=True)),
            ("NER Overlap", lambda: evaluate_entities(self.y_true_ner, self.y_pred_ner, mode="ner", strict_match=False)),
            ("Semi-Structured Position", lambda: evaluate_entities(self.y_true_semi, self.y_pred_semi, mode="semi_structured", matching_method="position")),
            ("Semi-Structured Text Exact", lambda: evaluate_entities(self.y_true_semi, self.y_pred_semi, mode="semi_structured", matching_method="text", text_match_mode="exact")),
            ("Semi-Structured Text Overlap", lambda: evaluate_entities(self.y_true_semi, self.y_pred_semi, mode="semi_structured", matching_method="text", text_match_mode="overlap")),
            ("Semi-Structured Text Similarity", lambda: evaluate_entities(self.y_true_semi, self.y_pred_semi, mode="semi_structured", matching_method="text", text_match_mode="similarity", similarity_threshold=0.8))
        ]
        
        print("Method Comparison:")
        print(f"{'Method':<30} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 60)
        
        for method_name, method_func in methods:
            try:
                results = method_func()
                print(f"{method_name:<30} {results['precision']:<10.3f} {results['recall']:<10.3f} {results['f1_score']:<10.3f}")
            except Exception as e:
                print(f"{method_name:<30} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} - {e}")
        
        print("\nExpected Behavior:")
        print("- NER Strict: Most conservative, only exact matches")
        print("- NER Overlap: More lenient, allows partial overlaps")
        print("- Position Matching: Ignores text content")
        print("- Text Exact: Ignores position, requires exact text")
        print("- Text Overlap: Allows partial text matches")
        print("- Text Similarity: Uses BERT for semantic similarity")


def load_test_data():
    """Load test data for standalone execution."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # 加载真实数据
    with open(os.path.join(data_dir, "breast_cancer_ground_truth.json"), "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    
    # 加载预测数据
    with open(os.path.join(data_dir, "breast_cancer_predictions.json"), "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    return ground_truth, predictions


if __name__ == "__main__":
    # 运行综合测试
    test_instance = TestBreastCancerComprehensive()
    test_instance.setup_method()
    
    print("=== 乳腺癌入院报告综合评估测试 ===")
    print("测试数据说明：")
    print("- 真实数据：包含12个医疗实体（患者信息、诊断、治疗等）")
    print("- 预测数据：包含各种错误类型（位置错误、文本错误、类型错误等）")
    print("- 测试覆盖：NER严格匹配、重叠匹配、半结构化位置匹配、文本匹配、相似度匹配")
    print()
    
    # 运行所有测试
    test_instance.test_ner_strict_matching()
    test_instance.test_ner_overlap_matching()
    test_instance.test_ner_by_type()
    test_instance.test_semi_structured_position_matching()
    test_instance.test_semi_structured_text_exact_matching()
    test_instance.test_semi_structured_text_overlap_matching()
    test_instance.test_semi_structured_text_similarity_matching()
    test_instance.test_quick_eval_functions()
    test_instance.test_validation_enabled()
    test_instance.test_edge_cases()
    test_instance.test_comprehensive_comparison()
    
    print("\n=== 测试完成 ===")
    print("所有评估模式都已测试，展示了不同匹配策略的效果差异。") 