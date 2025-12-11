#!/usr/bin/env python3
"""
Demonstration of similarity-based matching for semi-structured entity evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.easy_eval import evaluate_entities
from src.similarity_utils import calculate_text_similarity, is_texts_similar

def main():
    print("=== Similarity-Based Matching Demo ===\n")
    
    # Example data with medical terms that are semantically similar
    y_true = [
        {
            "entities": [
                {"text": "急性阑尾炎", "start": 10, "end": 15},
                {"text": "患者姓名", "start": 20, "end": 25},
                {"text": "腹腔镜手术", "start": 30, "end": 35},
                {"text": "入院日期", "start": 40, "end": 45}
            ]
        }
    ]
    
    y_pred = [
        {
            "entities": [
                {"text": "阑尾炎", "start": 10, "end": 13},           # Similar to "急性阑尾炎"
                {"text": "患者名", "start": 20, "end": 23},           # Similar to "患者姓名"
                {"text": "腹腔镜阑尾切除术", "start": 30, "end": 38}, # Similar to "腹腔镜手术"
                {"text": "入院时间", "start": 40, "end": 45},         # Similar to "入院日期"
                {"text": "糖尿病", "start": 50, "end": 53}            # Not similar to anything
            ]
        }
    ]
    
    print("Ground Truth Entities:")
    for entity in y_true[0]["entities"]:
        print(f"  - {entity['text']} ({entity['start']}-{entity['end']})")
    
    print("\nPredicted Entities:")
    for entity in y_pred[0]["entities"]:
        print(f"  - {entity['text']} ({entity['start']}-{entity['end']})")
    
    print("\n" + "="*60)
    
    # Test different similarity thresholds
    thresholds = [0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"\n--- Similarity Threshold: {threshold} ---")
        
        # Calculate similarity scores for each pair
        print("Similarity Scores:")
        for true_entity in y_true[0]["entities"]:
            for pred_entity in y_pred[0]["entities"]:
                similarity = calculate_text_similarity(true_entity["text"], pred_entity["text"])
                is_match = is_texts_similar(true_entity["text"], pred_entity["text"], threshold)
                status = "✓" if is_match else "✗"
                print(f"  {status} '{true_entity['text']}' vs '{pred_entity['text']}': {similarity:.3f}")
        
        # Evaluate with similarity matching
        results = evaluate_entities(
            y_true, y_pred, 
            mode="semi_structured", 
            matching_method="text", 
            text_match_mode="similarity",
            similarity_threshold=threshold
        )
        
        print(f"\nResults (threshold={threshold}):")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall:    {results['recall']:.3f}")
        print(f"  F1 Score:  {results['f1_score']:.3f}")
        print(f"  True Positives:  {results['true_positives']}")
        print(f"  False Positives: {results['false_positives']}")
        print(f"  False Negatives: {results['false_negatives']}")
    
    print("\n" + "="*60)
    
    # Compare with other matching methods
    print("\n--- Comparison with Other Matching Methods ---")
    
    # Position matching
    results_position = evaluate_entities(y_true, y_pred, mode="semi_structured", matching_method="position")
    print(f"Position Matching:     F1={results_position['f1_score']:.3f}")
    
    # Text exact matching
    results_exact = evaluate_entities(y_true, y_pred, mode="semi_structured", matching_method="text", text_match_mode="exact")
    print(f"Text Exact Matching:   F1={results_exact['f1_score']:.3f}")
    
    # Text overlap matching
    results_overlap = evaluate_entities(y_true, y_pred, mode="semi_structured", matching_method="text", text_match_mode="overlap")
    print(f"Text Overlap Matching: F1={results_overlap['f1_score']:.3f}")
    
    # Text similarity matching
    results_similarity = evaluate_entities(y_true, y_pred, mode="semi_structured", matching_method="text", text_match_mode="similarity", similarity_threshold=0.8)
    print(f"Text Similarity Matching (0.8): F1={results_similarity['f1_score']:.3f}")
    
    print("\n" + "="*60)
    
    # Key insights
    print("\n--- Key Insights ---")
    print("• Similarity matching captures semantic relationships between medical terms")
    print("• Lower thresholds (0.7) are more lenient, higher thresholds (0.9) are stricter")
    print("• BERT embeddings understand medical terminology better than exact string matching")
    print("• Similarity matching can handle synonyms and related medical terms")
    print("• Position matching is strictest, similarity matching is most flexible")
    
    print("\n--- Medical Term Examples ---")
    medical_pairs = [
        ("急性阑尾炎", "阑尾炎"),
        ("患者姓名", "患者名"),
        ("腹腔镜手术", "腹腔镜阑尾切除术"),
        ("入院日期", "入院时间"),
        ("糖尿病", "血糖异常"),
        ("高血压", "血压升高"),
        ("心脏病", "心血管疾病")
    ]
    
    print("Medical Term Similarities:")
    for term1, term2 in medical_pairs:
        similarity = calculate_text_similarity(term1, term2)
        print(f"  '{term1}' vs '{term2}': {similarity:.3f}")

if __name__ == "__main__":
    main() 