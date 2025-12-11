#!/usr/bin/env python3
"""
Main entry point for the evaluation project.
"""

from ner_metrics import NERMetrics, SemiStructuredMetrics

def main():
    """Main function."""
    print("Welcome to the Medical ENR Evaluation Project!")
    print("This project provides both NER and Semi-Structured Entity Extraction evaluation metrics.")
    
    # Run NER evaluation demo
    print("\n" + "="*50)
    print("Running Medical NER Evaluation Demo:")
    print("Demo functionality has been moved to dedicated test files.")
    
    # Example usage
    print("\n" + "="*50)
    print("Example MedicalNER Evaluation Usage:")
    
    # Example NER data
    y_true_ner = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
            ]
        }
    ]
    
    y_pred_ner = [
        {
            "entities": [
                {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
                {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"},
                {"start": 60, "end": 70, "type": "手术名称", "text": "额外手术"}  # False positive
            ]
        }
    ]
    
    # Example without validation (default behavior)
    ner_metrics = NERMetrics(enable_validation=False)
    ner_results = ner_metrics.calculate_metrics(y_true_ner, y_pred_ner)
    
    print(f"NER Precision: {ner_results['precision']:.3f}")
    print(f"NER Recall:    {ner_results['recall']:.3f}")
    print(f"NER F1 Score:  {ner_results['f1_score']:.3f}")
    
    # Example semi-structured data - 出院记录场景
    print("\n" + "="*50)
    print("Example Semi-Structured Evaluation Usage:")
    print("出院记录字段提取示例：")
    
    y_true_semi = [
        {
            "entities": [
                {"start": 5, "end": 9, "text": "患者姓名"},
                {"start": 10, "end": 12, "text": "张三"},
                {"start": 18, "end": 20, "text": "年龄"},
                {"start": 21, "end": 24, "text": "45岁"},
                {"start": 38, "end": 42, "text": "入院日期"},
                {"start": 43, "end": 53, "text": "2024-01-15"}
            ]
        }
    ]
    
    y_pred_semi = [
        {
            "entities": [
                {"start": 5, "end": 9, "text": "患者姓名"},
                {"start": 10, "end": 12, "text": "张三"},
                {"start": 18, "end": 20, "text": "年龄"},
                {"start": 21, "end": 24, "text": "45岁"},
                {"start": 38, "end": 42, "text": "入院日期"},
                {"start": 43, "end": 53, "text": "2024-01-15"},
                {"start": 60, "end": 64, "text": "床位号"},  # False positive
                {"start": 65, "end": 69, "text": "A101"}
            ]
        }
    ]
    
    # Example semi-structured evaluation
    semi_metrics = SemiStructuredMetrics(enable_validation=False)
    semi_results = semi_metrics.calculate_metrics(y_true_semi, y_pred_semi)
    
    print(f"Semi-Structured Precision: {semi_results['precision']:.3f}")
    print(f"Semi-Structured Recall:    {semi_results['recall']:.3f}")
    print(f"Semi-Structured F1 Score:  {semi_results['f1_score']:.3f}")
    
    print("\nProject is ready to use!")

if __name__ == "__main__":
    main() 